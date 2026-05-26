"""
03_merge_dataset.py
────────────────────
Loads all features extracted by 02_feature_extraction.py, merges them
with metadata, applies PCA dimensionality reduction, and saves a single
Parquet file ready for model training.

Steps:
  1. Load metadata + all index CSVs
  2. Inner-join to keep only videos with ALL features
  3. Stack numpy arrays → dense feature matrix
  4. Fit PCA on embeddings (hook / full / text), save PCA objects
  5. Add acoustic + LLM features
  6. Save final_dataset.parquet + feature_names.json

Usage:
    python 03_merge_dataset.py
    python 03_merge_dataset.py --no-pca          # skip PCA, keep raw dims
    python 03_merge_dataset.py --pca-components 32 32 24

Requirements:
    pip install pandas numpy scikit-learn pyarrow
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# 1. Load indexes and merge


def load_and_merge_indexes(df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join metadata with all feature indexes so we keep only complete rows."""
    required = {
        "clip":     config.EMBEDDING_INDEX_CSV,
        "text":     config.TEXT_INDEX_CSV,
        "whisper":  config.WHISPER_INDEX_CSV,
        "acoustic": config.AUDIO_INDEX_CSV,
    }

    merged = df.copy()
    for name, path in required.items():
        if not os.path.isfile(path):
            log.warning("Index not found, skipping join: %s", path)
            continue
        idx = pd.read_csv(path)
        idx["video_id"] = idx["video_id"].astype(str)
        idx = idx.drop_duplicates(subset=["video_id"], keep="first")

        before = len(merged)
        merged = merged.merge(idx, on="video_id", how="inner")
        log.info("  After joining %s: %d rows", name, len(merged))
    return merged.reset_index(drop=True)


def _load_npy_stack(paths: pd.Series, label: str) -> tuple[np.ndarray, list[int]]:
    """Load numpy files and stack into matrix. Returns (matrix, valid_indices)."""
    vectors, valid = [], []
    for i, p in enumerate(paths):
        try:
            v = np.load(str(p))
            vectors.append(v)
            valid.append(i)
        except Exception as e:
            log.debug("Cannot load %s: %s", p, e)
    if not vectors:
        raise RuntimeError(f"No valid numpy files for {label}")
    mat = np.vstack(vectors)
    log.info("  %-6s embeddings: shape=%s", label, mat.shape)
    return mat.astype(np.float32), valid



def fit_and_apply_pca(X: np.ndarray, n_components: int, label: str, save: bool = True) -> tuple[np.ndarray, PCA]:
    """Fit PCA and return reduced matrix + fitted PCA object."""
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    log.info("  PCA %s: %d → %d dims", label, X.shape[1], n_components)
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_reduced = pca.fit_transform(X)

    evr = pca.explained_variance_ratio_.sum()
    log.info("    Explained variance: %.1f%%", evr * 100)

    if save:
        pca_path = os.path.join(config.PCA_DIR, f"{label}_pca.pkl")
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)
        log.info("    PCA saved → %s", pca_path)

    return X_reduced.astype(np.float32), pca



# 4. Acoustic features

def load_acoustic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Load per-video acoustic JSON files and return a flat DataFrame."""
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Acoustic features"):
        vid = str(row["video_id"])
        p = row.get("audio_feat_path", "")
        if pd.notna(p) and os.path.isfile(str(p)):
            try:
                with open(p, "r") as f:
                    feats = json.load(f)
                feats["video_id"] = vid
                rows.append(feats)
            except Exception:
                rows.append({"video_id": vid})
        else:
            rows.append({"video_id": vid})

    acdf = pd.DataFrame(rows).fillna(0.0)
    acdf["video_id"] = acdf["video_id"].astype(str)
    log.info("Acoustic feature columns: %d", acdf.shape[1] - 1)
    return acdf


# 5. LLM features


def load_llm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Load LLM JSON files, encode categoricals, return feature DataFrame."""
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM features"):
        vid = str(row["video_id"])
        p = os.path.join(config.LLM_DIR, f"{vid}_llm.json")
        if os.path.isfile(p):
            try:
                with open(p, "r") as f:
                    d = json.load(f)
                rows.append(d)
            except Exception:
                rows.append({"video_id": vid})
        else:
            rows.append({"video_id": vid})

    llm_df = pd.DataFrame(rows)
    if llm_df.empty:
        log.warning("No LLM features found — skipping")
        return pd.DataFrame({"video_id": df["video_id"].astype(str)})

    llm_df["video_id"] = llm_df["video_id"].astype(str)

    # Numeric scores 
    for col in ["hook_score", "clarity_score", "quality_score"]:
        if col in llm_df.columns:
            llm_df[col] = pd.to_numeric(llm_df[col], errors="coerce").fillna(0)

    # List features → counts 
        if col in llm_df.columns:
            llm_df[f"num_{col}"] = llm_df[col].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )

    # Categorical → one-hot 
    cat_cols = [c for c in ["hook_type", "tone", "emotion", "content_category"] if c in llm_df.columns]
    keep_cols = (
        ["video_id"]
        + [c for c in ["hook_score", "clarity_score", "quality_score"] if c in llm_df.columns]
        + [f"num_{c}" for c in ["strengths", "weaknesses", "engagement_triggers"] if f"num_{c}" in llm_df.columns]
        + cat_cols
    )
    llm_df = llm_df[[c for c in keep_cols if c in llm_df.columns]]
    llm_df = pd.get_dummies(llm_df, columns=cat_cols)

    log.info("LLM feature columns: %d", llm_df.shape[1] - 1)
    return llm_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cosine similarity feature
# ─────────────────────────────────────────────────────────────────────────────

def hook_full_cosine(hook_mat: np.ndarray, full_mat: np.ndarray) -> np.ndarray:
    """Scalar cosine similarity between hook and full embedding per video."""
    h_norm = hook_mat / (np.linalg.norm(hook_mat, axis=1, keepdims=True) + 1e-8)
    f_norm = full_mat / (np.linalg.norm(full_mat, axis=1, keepdims=True) + 1e-8)
    sim = (h_norm * f_norm).sum(axis=1, keepdims=True)
    return sim.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Merge features and build final dataset")
    p.add_argument("--no-pca", action="store_true", help="Skip PCA, keep raw embedding dims")
    p.add_argument(
        "--pca-components",
        nargs=3,
        type=int,
        default=[config.PCA_HOOK_COMPONENTS, config.PCA_FULL_COMPONENTS, config.PCA_TEXT_COMPONENTS],
        metavar=("HOOK", "FULL", "TEXT"),
        help="Number of PCA components for hook / full / text embeddings",
    )
    return p.parse_args()


def main():
    args = parse_args()
    config.make_dirs()

    # ── Load metadata ──────────────────────────────────────────────────────
    if not os.path.isfile(config.ALL_METADATA_CSV):
        log.error("Metadata CSV not found: %s", config.ALL_METADATA_CSV)
        sys.exit(1)

    df = pd.read_csv(config.ALL_METADATA_CSV)
    df["video_id"] = df["video_id"].astype(str)
    log.info("Metadata: %d rows", len(df))

    # ── Merge indexes ──────────────────────────────────────────────────────
    log.info("Merging feature indexes…")
    df = load_and_merge_indexes(df)
    if len(df) == 0:
        log.error("No videos survive all inner joins. Check that feature extraction ran successfully.")
        sys.exit(1)

    # ── Load embedding matrices ────────────────────────────────────────────
    log.info("Loading embedding arrays…")
    hook_mat, hook_valid = _load_npy_stack(df["hook_path"], "hook")
    full_mat, full_valid = _load_npy_stack(df["full_path"], "full")
    text_mat, text_valid = _load_npy_stack(df["text_path"], "text")

    # Keep only rows that are valid in ALL three
    valid_rows = sorted(set(hook_valid) & set(full_valid) & set(text_valid))
    if len(valid_rows) < len(df):
        log.warning("Dropping %d rows due to missing numpy files", len(df) - len(valid_rows))
    df = df.iloc[valid_rows].reset_index(drop=True)
    hook_mat = hook_mat[[hook_valid.index(i) for i in valid_rows]]
    full_mat = full_mat[[full_valid.index(i) for i in valid_rows]]
    text_mat = text_mat[[text_valid.index(i) for i in valid_rows]]

    log.info("Valid rows after loading: %d", len(df))

    # ── PCA ────────────────────────────────────────────────────────────────
    if not args.no_pca:
        log.info("Applying PCA…")
        h_comp, f_comp, t_comp = args.pca_components
        hook_mat, _ = fit_and_apply_pca(hook_mat, h_comp, "hook")
        full_mat, _ = fit_and_apply_pca(full_mat, f_comp, "full")
        text_mat, _ = fit_and_apply_pca(text_mat, t_comp, "text")
    else:
        log.info("PCA skipped (--no-pca flag).")

    # ── Build DataFrame columns ────────────────────────────────────────────
    hook_cols = [f"hook_{i}" for i in range(hook_mat.shape[1])]
    full_cols = [f"full_{i}" for i in range(full_mat.shape[1])]
    text_cols = [f"text_{i}" for i in range(text_mat.shape[1])]

    hook_df = pd.DataFrame(hook_mat, columns=hook_cols)
    full_df = pd.DataFrame(full_mat, columns=full_cols)
    text_df = pd.DataFrame(text_mat, columns=text_cols)

    # ── Hook-Full cosine similarity ────────────────────────────────────────
    # Use raw (before PCA) embeddings for better geometric accuracy
    sim = hook_full_cosine(hook_mat, full_mat)
    sim_df = pd.DataFrame(sim, columns=["hook_full_cosine_sim"])

    # ── Target variable ────────────────────────────────────────────────────
    df["log_views"] = np.log1p(df["views"])

    # ── Acoustic features ──────────────────────────────────────────────────
    acoustic_df = pd.DataFrame()
    if os.path.isfile(config.AUDIO_INDEX_CSV):
        acoustic_df = load_acoustic_features(df)
        acoustic_df = acoustic_df.set_index("video_id").reindex(df["video_id"]).reset_index()
        # Remove duration_s if already in metadata
        acoustic_df = acoustic_df.drop(columns=["duration_s"], errors="ignore")

    # ── LLM features ──────────────────────────────────────────────────────
    llm_df = load_llm_features(df)
    llm_df = llm_df.set_index("video_id").reindex(df["video_id"]).reset_index()
    llm_df = llm_df.fillna(0)

    # ── Assemble final dataset ─────────────────────────────────────────────
    log.info("Assembling final dataset…")
    meta_cols = [c for c in config.META_COLS if c in df.columns]

    parts = [
        df[meta_cols].reset_index(drop=True),
        hook_df,
        full_df,
        text_df,
        sim_df,
    ]
    if not acoustic_df.empty:
        acoustic_df = acoustic_df.drop(columns=["video_id"], errors="ignore").reset_index(drop=True)
        parts.append(acoustic_df)
    if not llm_df.empty:
        llm_df = llm_df.drop(columns=["video_id"], errors="ignore").reset_index(drop=True)
        parts.append(llm_df)

    final_df = pd.concat(parts, axis=1)
    # drop any boolean columns left from pd.get_dummies (convert to int)
    bool_cols = final_df.select_dtypes(include="bool").columns
    final_df[bool_cols] = final_df[bool_cols].astype(int)

    final_df = final_df.fillna(0)
    log.info("Final dataset: %s", final_df.shape)

    # ── Save ───────────────────────────────────────────────────────────────
    final_df.to_parquet(config.FINAL_DATASET_PATH, index=False)
    log.info("Saved → %s", config.FINAL_DATASET_PATH)

    # Save feature names for inference scripts
    feature_cols = [c for c in final_df.columns if c not in config.LEAKY_COLS + ["video_id", "platform"]]
    names_path = os.path.join(config.FEATURE_DIR, "feature_names.json")
    with open(names_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    log.info("Feature names → %s  (%d features)", names_path, len(feature_cols))

    print("\n📊 Dataset summary:")
    print(final_df[["platform", "duration_seconds", "log_views"]].describe())


if __name__ == "__main__":
    main()
