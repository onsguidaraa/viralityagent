"""
05_agent_inference.py
──────────────────────
End-to-end inference agent with SHAP AI Coach.
Extracts features, runs the trained model, and uses SHAP to explain
exactly WHY the video got its score and what to improve.

Usage
  python 05_agent_inference.py --video path/to/video.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    log.warning("shap library not installed. Run `pip install shap` for the AI Coach features.")
    SHAP_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# 1. Download (if URL)
# ═══════════════════════════════════════════════════════════════════════════

def download_video(url: str, out_dir: str) -> str:
    cmd = [
        "yt-dlp", url,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", os.path.join(out_dir, "%(id)s.%(ext)s"),
        "--merge-output-format", "mp4",
        "--max-filesize", "100M",
        "--no-warnings", "--quiet",
    ]
    log.info("Downloading %s …", url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")
    for fname in os.listdir(out_dir):
        if fname.endswith(".mp4"):
            return os.path.join(out_dir, fname)
    raise FileNotFoundError("yt-dlp ran but produced no .mp4 file.")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Feature extraction (reuse step-02 functions)
# ═══════════════════════════════════════════════════════════════════════════

def _get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def extract_clip(video_path: str) -> tuple[np.ndarray, np.ndarray]:
    from transformers import CLIPModel, CLIPProcessor
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("fe", Path(__file__).parent / "02_feature_extraction.py")
    fe = module_from_spec(spec); spec.loader.exec_module(fe)  # type: ignore

    device = _get_device()
    model  = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(device)
    proc   = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    model.eval()

    hook_frames = fe.extract_frames(video_path, config.HOOK_MAX_FRAMES, config.HOOK_SECONDS)
    full_frames = fe.extract_frames(video_path, config.FULL_MAX_FRAMES)
    if not hook_frames: hook_frames = full_frames[:1]

    hook_emb = fe.encode_frames(hook_frames, model, proc)
    full_emb = fe.encode_frames(full_frames, model, proc)
    return hook_emb, full_emb

def extract_audio_features(video_path: str, tmp_dir: str) -> tuple[dict, str, str]:
    import whisper
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("fe", Path(__file__).parent / "02_feature_extraction.py")
    fe = module_from_spec(spec); spec.loader.exec_module(fe)  # type: ignore

    wav_path = os.path.join(tmp_dir, "audio.wav")
    if not fe.extract_audio_wav(video_path, wav_path):
        return {}, "", ""

    device = _get_device()
    wmodel = whisper.load_model(config.WHISPER_MODEL_SIZE)
    result = wmodel.transcribe(wav_path, fp16=(device == "cuda"))
    full_tr = result["text"].strip()
    hook_segs = [s for s in result.get("segments", []) if s["start"] < config.HOOK_SECONDS]
    hook_tr = " ".join(s["text"] for s in hook_segs).strip()

    acoustic = fe.extract_acoustic_features(wav_path)
    return acoustic, hook_tr, full_tr

def extract_text_embedding(title: str, hook_tr: str, full_tr: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.TEXT_MODEL_NAME)
    text = (f"Title: {title}\n\nHook (first {config.HOOK_SECONDS}s):\n{hook_tr}\n\nFull transcript:\n{full_tr}").strip()
    return model.encode(text, normalize_embeddings=True).astype(np.float32)

def run_llm_analysis(title: str, platform: str, duration: float, hook_tr: str, full_tr: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY", "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")
    if not api_key: return {}
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = (
            f"You are an expert viral video analyst.\nPlatform: {platform}\nTitle: {title}\nDuration: {duration}s\n"
            f"Hook transcript: {hook_tr or '(none)'}\nFull transcript: {full_tr or '(none)'}\n\n"
            "Return ONLY valid JSON. Include a short 'video_summary' explaining what the video is about based on the text.\n"
            '{"video_summary":"string (2-3 sentences)","hook_score":0,"clarity_score":0,"quality_score":0,"hook_type":"curiosity",'
            '"tone":"entertaining","emotion":"string","content_category":"string","engagement_triggers":[],'
            '"strengths":[],"weaknesses":[],"improvement_suggestion":"string"}'
        )
        resp = client.chat.completions.create(
            model=config.GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=700, response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        log.warning(f"LLM analysis failed: {e}")
        return {}

# ═══════════════════════════════════════════════════════════════════════════
# 3 & 4. PCA and Vector Assembly
# ═══════════════════════════════════════════════════════════════════════════

def apply_pca(emb: np.ndarray, label: str) -> np.ndarray:
    path = os.path.join(config.PCA_DIR, f"{label}_pca.pkl")
    if not os.path.isfile(path): return emb
    with open(path, "rb") as f:
        pca = pickle.load(f)
    return pca.transform(emb.reshape(1, -1)).flatten().astype(np.float32)

def build_feature_vector(hook_emb, full_emb, text_emb, acoustic, llm, duration_s, feature_cols):
    hook_r, full_r, text_r = apply_pca(hook_emb, "hook"), apply_pca(full_emb, "full"), apply_pca(text_emb, "text")
    sim = float(np.dot(hook_r / (np.linalg.norm(hook_r) + 1e-8), full_r / (np.linalg.norm(full_r) + 1e-8)))

    feat = {"hook_full_cosine_sim": sim, "duration_seconds": float(duration_s)}
    for i, v in enumerate(hook_r): feat[f"hook_{i}"] = float(v)
    for i, v in enumerate(full_r): feat[f"full_{i}"] = float(v)
    for i, v in enumerate(text_r): feat[f"text_{i}"] = float(v)
    for k, v in acoustic.items(): feat[k] = float(v)

    for col in ["hook_score", "clarity_score", "quality_score"]: feat[col] = float(llm.get(col, 0))
    for col in ["strengths", "weaknesses", "engagement_triggers"]:
        val = llm.get(col, [])
        feat[f"num_{col}"] = float(len(val) if isinstance(val, list) else 0)

    for cat_col in ["hook_type", "tone", "emotion", "content_category"]:
        val = llm.get(cat_col, "")
        for fc in feature_cols:
            if fc.startswith(f"{cat_col}_"): feat[fc] = 1.0 if fc == f"{cat_col}_{val}" else 0.0

    return np.array([feat.get(c, 0.0) for c in feature_cols], dtype=np.float32)

# ═══════════════════════════════════════════════════════════════════════════
# 5. SHAP AI Coach Explanation
# ═══════════════════════════════════════════════════════════════════════════

def _translate_feature(feat: str, val: float) -> str:
    """Translates raw ML feature names into human-readable diagnostic feedback."""
    if "duration" in feat: return f"Video Length ({val:.1f}s)"
    if "rms_max" in feat: return f"Peak Audio Energy/Volume"
    if "spec_centroid" in feat: return "Audio Brightness (EQ)"
    if "tempo" in feat: return f"Pacing/Tempo ({val:.0f} BPM)"
    if "silence_ratio" in feat: return f"Dead Air / Silence ({val*100:.0f}%)"
    if feat.startswith("hook_type_"): return f"Hook Strategy: {feat.split('_')[-1].title()}"
    if feat.startswith("tone_"): return f"Tone: {feat.split('_', 1)[-1].title()}"
    if feat.startswith("emotion_"): return f"Emotion: {feat.split('_', 1)[-1].title()}"
    if feat.startswith("content_category_"): return f"Category: {feat.split('_', 2)[-1].title()}"
    
    # --- UPDATED THESE LINES TO INCLUDE THE NUMBER ---
    if feat.startswith("hook_") and feat[5:].isdigit(): return f"Hook Visual Pattern #{feat[5:]}"
    if feat.startswith("full_") and feat[5:].isdigit(): return f"Overall Visual Pattern #{feat[5:]}"
    if feat.startswith("text_") and feat[5:].isdigit(): return f"Transcript Semantics #{feat[5:]}"
    # -----------------------------------------------
    
    if feat == "hook_full_cosine_sim": return "Match between Hook and Full Video"
    return feat.replace("_", " ").title()

def get_shap_feedback(model, X, feature_cols, model_type):
    if not SHAP_AVAILABLE or model_type not in ["xgb", "lgbm", "rf"]:
        return [], []
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)[0] # Get values for the single row
        
        contributions = []
        for i, col in enumerate(feature_cols):
            contributions.append({
                "feature": col,
                "raw_name": col,
                "value": X[0, i],
                "impact": shap_vals[i]
            })
            
        contributions.sort(key=lambda x: x["impact"])
        
        # Top 3 negative (dragging down)
        top_negative = [c for c in contributions if c["impact"] < 0][:3]
        # Top 3 positive (pushing up)
        top_positive = sorted([c for c in contributions if c["impact"] > 0], key=lambda x: x["impact"], reverse=True)[:3]
        
        up_feedback = [{"msg": _translate_feature(c['feature'], c['value']), "impact": c['impact']} for c in top_positive]
        down_feedback = [{"msg": _translate_feature(c['feature'], c['value']), "impact": c['impact']} for c in top_negative]
        
        return up_feedback, down_feedback
    except Exception as e:
        log.warning(f"SHAP failed: {e}")
        return [], []

# ═══════════════════════════════════════════════════════════════════════════
# 6. Report
# ═══════════════════════════════════════════════════════════════════════════

def build_report(log_views_pred, llm, hook_tr, full_tr, video_path, shap_up, shap_down) -> dict:
    predicted_views = int(np.expm1(log_views_pred))
    if log_views_pred >= np.log1p(1_000_000): tier = "🔥 VIRAL (1M+ predicted views)"
    elif log_views_pred >= np.log1p(100_000): tier = "📈 High potential (100K+ predicted views)"
    elif log_views_pred >= np.log1p(10_000):  tier = "📊 Moderate potential (10K+ predicted views)"
    else: tier = "📉 Low reach (< 10K predicted views)"

    return {
        "video": os.path.basename(video_path),
        "predicted_views_estimate": predicted_views,
        "viral_tier": tier,
        "shap_pushing_up": shap_up,
        "shap_dragging_down": shap_down,
        "llm_analysis": llm,
        "hook_transcript": hook_tr,
    }

def print_report(report: dict) -> None:
    print("\n" + "═" * 65)
    print("  🤖  VIRAL AGENT — AI COACH REPORT")
    print("═" * 65)
    print(f"  Video   : {report['video']}")
    print(f"  Tier    : {report['viral_tier']}")
    print(f"  Est.    : ~{report['predicted_views_estimate']:,} views")
    print("─" * 65)
    
    llm = report.get("llm_analysis", {})
    
    # NEW: Print the LLM Video Summary
    if llm.get("video_summary"):
        print(f"  📝 VIDEO SUMMARY:\n    {llm['video_summary']}")
        print("─" * 65)
    
    print("  🚀 WHAT'S WORKING WELL (Model Insights):")
    if not report['shap_pushing_up']: print("    (No strong positive signals detected)")
    for i in report['shap_pushing_up']: 
        print(f"    ✅ {i['msg']} (Score Boost: +{i['impact']:.2f})")
        
    print("\n  ⚠️ WHAT'S HOLDING IT BACK (Model Insights):")
    if not report['shap_dragging_down']: print("    (No strong negative signals detected)")
    for i in report['shap_dragging_down']: 
        print(f"    🔻 {i['msg']} (Score Penalty: {i['impact']:.2f})")
    
    print("─" * 65)
    if llm.get("improvement_suggestion"):
        print(f"  💡 CREATIVE ADVICE:\n    {llm['improvement_suggestion']}")
        
    print("═" * 65 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video", type=str, help="Path to local .mp4 file")
    grp.add_argument("--url",   type=str, help="YouTube/TikTok URL")
    p.add_argument("--title",    type=str, default="")
    p.add_argument("--platform", type=str, default="unknown")
    p.add_argument("--output",   type=str, default=None)
    args = p.parse_args()
    config.make_dirs()

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = download_video(args.url, tmp_dir) if args.url else args.video
        if not os.path.isfile(video_path):
            log.error("File not found: %s", video_path)
            sys.exit(1)

        log.info("Extracting features...")
        hook_emb, full_emb = extract_clip(video_path)
        acoustic, hook_tr, full_tr = extract_audio_features(video_path, tmp_dir)
        duration_s = acoustic.pop("duration_s", 0.0)
        text_emb = extract_text_embedding(args.title, hook_tr, full_tr)
        llm = run_llm_analysis(args.title, args.platform, duration_s, hook_tr, full_tr)

        meta_path = os.path.join(config.MODEL_DIR, "model_meta.json")
        with open(meta_path) as f: meta = json.load(f)
        feature_cols = meta["feature_cols"]

        X = build_feature_vector(hook_emb, full_emb, text_emb, acoustic, llm, duration_s, feature_cols).reshape(1, -1)

        model_path = os.path.join(config.MODEL_DIR, "best_model.pkl")
        with open(model_path, "rb") as f: model = pickle.load(f)

        log_views_pred = float(model.predict(X)[0])
        
        # Calculate SHAP Explanations
        log.info("Generating AI Coach insights (SHAP)...")
        shap_up, shap_down = get_shap_feedback(model, X, feature_cols, meta["model_type"])

        report = build_report(log_views_pred, llm, hook_tr, full_tr, video_path, shap_up, shap_down)
        print_report(report)

        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f: json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()