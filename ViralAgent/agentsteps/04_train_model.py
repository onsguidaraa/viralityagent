"""
04_train_model.py
──────────────────
Trains a viral-video regression model on the merged feature dataset
produced by 03_merge_dataset.py.

Supported model architectures
  --model xgb   → XGBoost  (default, best for tabular + PCA features)
  --model lgbm  → LightGBM (faster, similar accuracy)
  --model mlp   → PyTorch MLP (jointly learns from all modalities)
  --model rf    → Random Forest (legacy baseline)

Outputs
  models/best_model.pkl         (XGB / LGBM / RF)
  models/best_model_mlp.pt      (MLP state-dict)
  results/cv_scores.json        cross-validation metrics
  results/feature_importance.csv (XGB / LGBM / RF only)

Usage
  python 04_train_model.py
  python 04_train_model.py --model lgbm
  python 04_train_model.py --model mlp --epochs 150
  python 04_train_model.py --model xgb --tune     # Optuna HPO (optional)

Requirements
  pip install xgboost lightgbm scikit-learn optuna torch pandas pyarrow
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load final_dataset.parquet → X, y arrays and feature names."""
    if not os.path.isfile(config.FINAL_DATASET_PATH):
        log.error("Dataset not found: %s\nRun 03_merge_dataset.py first.", config.FINAL_DATASET_PATH)
        sys.exit(1)

    df = pd.read_parquet(config.FINAL_DATASET_PATH)
    log.info("Loaded dataset: %s", df.shape)

    # Load pre-saved feature names (respects leaky-col exclusion from step 03)
    names_path = os.path.join(config.FEATURE_DIR, "feature_names.json")
    if os.path.isfile(names_path):
        with open(names_path) as f:
            feature_cols = json.load(f)
        # Keep only columns that actually exist in this parquet
        feature_cols = [c for c in feature_cols if c in df.columns]
    else:
        # Fallback: exclude known leaky and meta columns
        feature_cols = [
            c for c in df.columns
            if c not in config.LEAKY_COLS + ["video_id", "platform",
                                              "title", "url", "uploader",
                                              "upload_date", "local_video_path",
                                              "hook_path", "full_path",
                                              "text_path", "audio_feat_path",
                                              "transcript_path"]
        ]

    if config.TARGET_COL not in df.columns:
        log.error("Target column '%s' not found in dataset.", config.TARGET_COL)
        sys.exit(1)

    X = df[feature_cols].values.astype(np.float32)
    y = df[config.TARGET_COL].values.astype(np.float32)

    log.info("Features: %d  |  Samples: %d", X.shape[1], X.shape[0])
    log.info("Target   mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
             y.mean(), y.std(), y.min(), y.max())

    return X, y, feature_cols


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def cross_validate(model_factory, X: np.ndarray, y: np.ndarray,
                   n_folds: int = config.CV_FOLDS) -> list[dict]:
    """k-fold CV returning per-fold metrics."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = model_factory()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)

        m = regression_metrics(y_va, preds)
        m["fold"] = fold
        fold_metrics.append(m)
        log.info("  Fold %d/%d  RMSE=%.4f  MAE=%.4f  R²=%.4f",
                 fold, n_folds, m["rmse"], m["mae"], m["r2"])

    return fold_metrics


def summarise_cv(fold_metrics: list[dict]) -> dict:
    summary = {}
    for key in ["rmse", "mae", "r2"]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Model builders
# ═════════════════════════════════════════════════════════════════════════════

def build_xgb(params: dict | None = None):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        log.error("xgboost not installed: pip install xgboost")
        sys.exit(1)
    p = {**config.XGB_PARAMS, **(params or {})}
    return XGBRegressor(**p)


def build_lgbm(params: dict | None = None):
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        log.error("lightgbm not installed: pip install lightgbm")
        sys.exit(1)
    p = {**config.LGBM_PARAMS, **(params or {})}
    return LGBMRegressor(**p)


def build_rf(params: dict | None = None):
    from sklearn.ensemble import RandomForestRegressor
    defaults = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": config.RANDOM_STATE,
    }
    p = {**defaults, **(params or {})}
    return RandomForestRegressor(**p)



# MLP (PyTorch)


class _MLP(object):
    """Thin sklearn-compatible wrapper around a PyTorch MLP regressor."""

    def __init__(self, input_dim: int, hidden_dims: list[int],
                 dropout: float, epochs: int, batch_size: int, lr: float,
                 device: str = "cpu"):
        import torch
        import torch.nn as nn

        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers).to(self.device)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs)
        self.criterion = nn.MSELoss()
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        X = self.scaler.fit_transform(X)
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.net.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(xb), yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            self.scheduler.step()
            if epoch % 10 == 0:
                log.info("    Epoch %3d/%d  loss=%.4f", epoch, self.epochs, epoch_loss / len(loader))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        X = self.scaler.transform(X)
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.net.eval()
        with torch.no_grad():
            preds = self.net(X_t).squeeze(1).cpu().numpy()
        return preds

    def save(self, path: str):
        import torch
        torch.save(self.net.state_dict(), path)
        log.info("MLP weights → %s", path)


def build_mlp(input_dim: int, epochs: int | None = None) -> _MLP:
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        log.error("torch not installed: pip install torch")
        sys.exit(1)

    log.info("MLP device: %s", device)
    return _MLP(
        input_dim=input_dim,
        hidden_dims=config.MLP_HIDDEN_DIMS,
        dropout=config.MLP_DROPOUT,
        epochs=epochs or config.MLP_EPOCHS,
        batch_size=config.MLP_BATCH_SIZE,
        lr=config.MLP_LR,
        device=device,
    )



# Optuna HPO  (optional, only if --tune is passed)
# ════════════════════════════════════════════════════════════════════════════

def tune_xgb(X_tr: np.ndarray, y_tr: np.ndarray, n_trials: int = 40) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("optuna not installed — skipping HPO. pip install optuna")
        return {}

    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": config.RANDOM_STATE,
            "n_jobs": -1,
        }
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X_tr, y_tr,
                                 cv=3, scoring="neg_root_mean_squared_error",
                                 n_jobs=1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info("Best HPO RMSE: %.4f", study.best_value)
    log.info("Best params:   %s", study.best_params)
    return study.best_params


# ═════════════════════════════════════════════════════════════════════════════
# Feature importance
# ═════════════════════════════════════════════════════════════════════════════

def save_feature_importance(model, feature_cols: list[str], model_name: str):
    """Save a feature importance CSV for tree-based models."""
    try:
        importances = model.feature_importances_
    except AttributeError:
        return  # MLP has no feature_importances_

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    path = os.path.join(config.RESULTS_DIR, f"{model_name}_feature_importance.csv")
    fi.to_csv(path, index=False)
    log.info("Feature importance → %s", path)
    log.info("Top 15 features:\n%s", fi.head(15).to_string(index=False))


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def reach_tiers(log_views: np.ndarray) -> np.ndarray:
    """Convert log-view regression values into reach classes."""
    bins = np.log1p([10_000, 100_000, 1_000_000])
    labels = np.array(["Low", "Moderate", "High", "Viral"])
    return labels[np.digitize(log_views, bins)]


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Save a confusion matrix for predicted reach tiers."""
    labels = ["Low", "Moderate", "High", "Viral"]
    true_tiers = reach_tiers(y_true)
    pred_tiers = reach_tiers(y_pred)
    cm = confusion_matrix(true_tiers, pred_tiers, labels=labels)

    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )
    csv_path = os.path.join(config.RESULTS_DIR, f"{model_name}_confusion_matrix.csv")
    cm_df.to_csv(csv_path)
    log.info("Confusion matrix CSV -> %s", csv_path)
    log.info("Reach-tier confusion matrix:\n%s", cm_df.to_string())

    png_path = os.path.join(config.RESULTS_DIR, f"{model_name}_confusion_matrix.png")
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"{model_name.upper()} Reach Tier Confusion Matrix")
        fig.tight_layout()
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        log.info("Confusion matrix PNG -> %s", png_path)
    except Exception as e:
        png_path = ""
        log.warning("Could not save confusion matrix PNG: %s", e)

    return {
        "labels": labels,
        "csv_path": csv_path,
        "png_path": png_path,
        "matrix": cm.tolist(),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train viral-video regression model")
    p.add_argument("--model",  choices=["xgb", "lgbm", "mlp", "rf"],
                   default="xgb",
                   help="Model architecture (default: xgb)")
    p.add_argument("--no-cv",  action="store_true",
                   help="Skip cross-validation, train on full train split only")
    p.add_argument("--tune",   action="store_true",
                   help="Run Optuna HPO before final training (XGB only)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Training epochs (MLP only, overrides config)")
    p.add_argument("--test-size", type=float, default=config.TEST_SIZE,
                   help="Fraction reserved for final hold-out test")
    return p.parse_args()


def main():
    args = parse_args()
    config.make_dirs()

    # ── Load data 
    X, y, feature_cols = load_dataset()

    # Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=config.RANDOM_STATE,
    )
    log.info("Train=%d  Test=%d", len(X_train), len(X_test))

    # Optional HPO 
    best_hpo_params: dict = {}
    if args.tune:
        if args.model != "xgb":
            log.warning("--tune is only supported for XGBoost; ignoring.")
        else:
            log.info("Running Optuna HPO (40 trials)…")
            best_hpo_params = tune_xgb(X_train, y_train, n_trials=40)

    # ── Cross-validation ───────────────────────────────────────────────────
    all_cv_results: dict = {}

    if not args.no_cv:
        log.info("▶ %d-fold cross-validation on training set…", config.CV_FOLDS)

        if args.model == "xgb":
            factory = lambda: build_xgb(best_hpo_params)
        elif args.model == "lgbm":
            factory = lambda: build_lgbm()
        elif args.model == "rf":
            factory = lambda: build_rf()
        else:
            # MLP — we skip CV to save time; it's done once below
            log.info("Skipping fold-CV for MLP (use --no-cv to suppress this notice).")
            factory = None

        if factory is not None:
            fold_metrics = cross_validate(factory, X_train, y_train)
            summary = summarise_cv(fold_metrics)
            all_cv_results["folds"] = fold_metrics
            all_cv_results["summary"] = summary
            log.info("CV summary: RMSE=%.4f±%.4f  MAE=%.4f±%.4f  R²=%.4f±%.4f",
                     summary["rmse"]["mean"], summary["rmse"]["std"],
                     summary["mae"]["mean"],  summary["mae"]["std"],
                     summary["r2"]["mean"],   summary["r2"]["std"])

    # Final model training 
    log.info("▶ Training final %s on full training set…", args.model.upper())
    t0 = time.time()

    if args.model == "xgb":
        model = build_xgb(best_hpo_params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

    elif args.model == "lgbm":
        from lightgbm import early_stopping, log_evaluation
        model = build_lgbm()
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  callbacks=[early_stopping(50, verbose=False),
                              log_evaluation(period=-1)])

    elif args.model == "rf":
        model = build_rf()
        model.fit(X_train, y_train)

    elif args.model == "mlp":
        model = build_mlp(input_dim=X_train.shape[1], epochs=args.epochs)
        model.fit(X_train, y_train)

    elapsed = time.time() - t0
    log.info("Training done in %.1f s", elapsed)

    # Hold-out evaluation 
    log.info("▶ Evaluating on hold-out test set…")
    test_preds = model.predict(X_test)
    test_m = regression_metrics(y_test, test_preds)
    log.info("Test  RMSE=%.4f  MAE=%.4f  R²=%.4f",
             test_m["rmse"], test_m["mae"], test_m["r2"])

    all_cv_results["test"] = test_m
    all_cv_results["model"] = args.model
    all_cv_results["n_features"] = X.shape[1]
    all_cv_results["n_train"] = len(X_train)
    all_cv_results["n_test"]  = len(X_test)

    # Save results JSON 
    results_path = os.path.join(config.RESULTS_DIR, "cv_scores.json")
    with open(results_path, "w") as f:
        json.dump(all_cv_results, f, indent=2)
    log.info("Results → %s", results_path)

    # Save model 
    if args.model == "mlp":
        model_path = os.path.join(config.MODEL_DIR, "best_model_mlp.pt")
        model.save(model_path)
        # Also pickle the scaler so inference can use it
        scaler_path = os.path.join(config.MODEL_DIR, "mlp_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(model.scaler, f)
        log.info("MLP scaler → %s", scaler_path)
    else:
        model_path = os.path.join(config.MODEL_DIR, "best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info("Model → %s", model_path)

    # Feature importance (tree models)
    save_feature_importance(model, feature_cols, args.model)

    # Save metadata needed by inference
    meta = {
        "model_type":   args.model,
        "feature_cols": feature_cols,
        "target_col":   config.TARGET_COL,
        "pca_dir":      config.PCA_DIR,
    }
    meta_path = os.path.join(config.MODEL_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Model meta → %s", meta_path)

    print("\n✅ Training complete.")
    print(f"   Model type : {args.model.upper()}")
    print(f"   Test RMSE  : {test_m['rmse']:.4f}")
    print(f"   Test R²    : {test_m['r2']:.4f}")


if __name__ == "__main__":
    main()
