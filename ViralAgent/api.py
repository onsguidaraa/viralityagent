# -*- coding: utf-8 -*-
"""
api.py -- ViralAgent FastAPI Server
--------------------------------------
Exposes the viral-video prediction pipeline as an HTTP API so your
Node.js dashboard can POST a video file and get back an AI Coach report.

Endpoints:
  POST /analyze        -- Upload a video + title + platform -> full report
  GET  /health         -- Liveness check
  GET  /model-info     -- Model metadata (type, feature count...)

Run:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Install deps (if not already):
  pip install fastapi uvicorn[standard] python-multipart
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
# The model files sit next to this api.py file (d:\BviralAI\ViralAgent\)
BASE_DIR  = Path(__file__).parent
AGENT_DIR = BASE_DIR / "agentsteps"

# Override VIRAL_AGENT_BASE so config.py resolves to local Windows paths
os.environ.setdefault("VIRAL_AGENT_BASE", str(BASE_DIR))
os.environ.setdefault("GROQ_API_KEY",     "gsk_5J8EWnclovf4WCfmA743WGdyb3FY1cgXoRDUTCb1A86aWB1BY2h")

# Add agentsteps to path so we can import config + inference helpers
sys.path.insert(0, str(AGENT_DIR))
import config  # noqa: E402

# Override MODEL_DIR to point at the folder that holds best_model.pkl
config.MODEL_DIR   = str(BASE_DIR)
config.PCA_DIR     = str(BASE_DIR / "features" / "pca_reduced")
config.FEATURE_DIR = str(BASE_DIR / "features")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("viral_api")

# ---------------------------------------------------------------------------
# Import inference helpers directly from 05_agent_inference.py
# (reuses all the logic without duplicating code)
# ---------------------------------------------------------------------------
from importlib.util import module_from_spec, spec_from_file_location


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    mod  = module_from_spec(spec)       # type: ignore[arg-type]
    spec.loader.exec_module(mod)         # type: ignore[union-attr]
    return mod


_inf = _load_module("inference", AGENT_DIR / "05_agent_inference.py")

extract_clip           = _inf.extract_clip
extract_audio_features = _inf.extract_audio_features
extract_text_embedding = _inf.extract_text_embedding
run_llm_analysis       = _inf.run_llm_analysis
build_feature_vector   = _inf.build_feature_vector
get_shap_feedback      = _inf.get_shap_feedback
build_report           = _inf.build_report

# ---------------------------------------------------------------------------
# Startup: load model once, keep in memory
# ---------------------------------------------------------------------------
_state: dict = {}


def _load_model_artifacts():
    model_pkl = BASE_DIR / "best_model.pkl"
    meta_json = BASE_DIR / "model_meta.json"

    if not model_pkl.exists():
        raise RuntimeError(f"Model file not found: {model_pkl}")
    if not meta_json.exists():
        raise RuntimeError(f"Model meta not found: {meta_json}")

    with open(model_pkl, "rb") as fh:
        model = pickle.load(fh)
    with open(meta_json, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    log.info(
        "Loaded model: %s  |  features: %d",
        meta["model_type"],
        len(meta["feature_cols"]),
    )
    return model, meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, release on shutdown."""
    log.info("ViralAgent API starting up...")
    try:
        model, meta = _load_model_artifacts()
        _state["model"] = model
        _state["meta"]  = meta
        log.info("Model ready. API is live.")
    except Exception as exc:
        log.error("Failed to load model on startup: %s", exc)
        _state["startup_error"] = str(exc)
    yield
    log.info("ViralAgent API shutting down.")
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ViralAgent API",
    description="Predict the viral potential of a short-form video using AI.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS -- allow the Node dashboard on any origin (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: map log-view prediction to a 0-100 viral score
# ---------------------------------------------------------------------------
def _log_views_to_score(log_views: float) -> int:
    """
    Intuitive 0-100 score calibrated as:
      log1p(0)       ->   0
      log1p(10k)     ~   40
      log1p(100k)    ~   65
      log1p(1M)      ~   85
      log1p(10M+)    ->  100
    """
    max_log = float(np.log1p(10_000_000))
    score   = int(round(min(log_views / max_log * 100, 100)))
    return max(0, score)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    """Liveness + readiness check."""
    if "startup_error" in _state:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": _state["startup_error"]},
        )
    if "model" not in _state:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "model_type": _state["meta"]["model_type"]}


@app.get("/model-info", tags=["System"])
async def model_info():
    """Return metadata about the loaded model."""
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    meta = _state["meta"]
    return {
        "model_type":    meta["model_type"],
        "feature_count": len(meta["feature_cols"]),
        "target_col":    meta["target_col"],
        "pca_dir":       meta.get("pca_dir", config.PCA_DIR),
    }


@app.post("/analyze", tags=["Inference"])
async def analyze_video(
    file:     UploadFile = File(...,        description="Video file (.mp4, .mov, etc.)"),
    title:    str        = Form("",         description="Video title or caption"),
    platform: str        = Form("unknown",  description="Platform: tiktok | youtube | instagram | unknown"),
):
    """
    Analyze a video for viral potential.

    **Request** (multipart/form-data):
    - `file`     — the video file
    - `title`    — video title / caption (optional but improves accuracy)
    - `platform` — tiktok | youtube | instagram | unknown

    **Response** fields:
    - `video`                    — original filename
    - `viral_score`              — 0-100 intuitive score
    - `predicted_views_estimate` — estimated view count
    - `viral_tier`               — Low / Moderate / High / Viral label
    - `shap_pushing_up`          — top factors boosting the score
    - `shap_dragging_down`       — top factors hurting the score
    - `llm_analysis`             — hook score, tone, strengths, suggestions
    - `hook_transcript`          — transcribed first few seconds of speech
    - `title`                    — echoed back
    - `platform`                 — echoed back
    - `processing_time_seconds`  — wall-clock time for this request
    """
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health.")

    model        = _state["model"]
    meta         = _state["meta"]
    feature_cols = meta["feature_cols"]
    model_type   = meta["model_type"]

    t_start = time.time()

    # Keep the original file extension so ffmpeg / cv2 recognize the format
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"

    with tempfile.TemporaryDirectory(prefix="viral_") as tmp_dir:
        video_path = os.path.join(tmp_dir, f"input{suffix}")

        log.info("Saving upload -> %s", video_path)
        content = await file.read()
        with open(video_path, "wb") as fh:
            fh.write(content)

        if not os.path.isfile(video_path) or os.path.getsize(video_path) < 1024:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file appears empty or too small.",
            )

        log.info(
            "Analyzing: %s  (title=%r, platform=%r, size=%d bytes)",
            file.filename, title, platform, len(content),
        )

        # -- Step 1: CLIP visual embeddings ----------------------------------
        try:
            log.info("[1/4] CLIP visual embeddings...")
            hook_emb, full_emb = extract_clip(video_path)
        except Exception as exc:
            log.exception("CLIP extraction failed")
            raise HTTPException(
                status_code=422,
                detail=f"Visual feature extraction failed: {exc}",
            )

        # -- Step 2: Whisper ASR + Librosa acoustic features -----------------
        try:
            log.info("[2/4] Whisper audio + acoustic features...")
            acoustic, hook_tr, full_tr = extract_audio_features(video_path, tmp_dir)
            duration_s = acoustic.pop("duration_s", 0.0)
        except Exception as exc:
            log.warning("Audio extraction failed (continuing with empty): %s", exc)
            acoustic, hook_tr, full_tr, duration_s = {}, "", "", 0.0

        # -- Step 3: Text embedding ------------------------------------------
        try:
            log.info("[3/4] Text embedding (title + transcript)...")
            text_emb = extract_text_embedding(title, hook_tr, full_tr)
        except Exception as exc:
            log.exception("Text embedding failed")
            raise HTTPException(
                status_code=422,
                detail=f"Text embedding failed: {exc}",
            )

        # -- Step 4: Groq LLM analysis ---------------------------------------
        try:
            log.info("[4/4] Groq LLM analysis...")
            llm = run_llm_analysis(title, platform, duration_s, hook_tr, full_tr)
        except Exception as exc:
            log.warning("LLM analysis failed (continuing without it): %s", exc)
            llm = {}

        # -- Predict ---------------------------------------------------------
        try:
            X = build_feature_vector(
                hook_emb, full_emb, text_emb,
                acoustic, llm, duration_s, feature_cols,
            ).reshape(1, -1)

            log_views_pred = float(model.predict(X)[0])
        except Exception as exc:
            log.exception("Prediction failed")
            raise HTTPException(
                status_code=500,
                detail=f"Model prediction failed: {exc}",
            )

        # -- SHAP explanations -----------------------------------------------
        try:
            shap_up, shap_down = get_shap_feedback(model, X, feature_cols, model_type)
        except Exception as exc:
            log.warning("SHAP failed (non-critical): %s", exc)
            shap_up, shap_down = [], []

        # -- Build report ----------------------------------------------------
        report = build_report(
            log_views_pred, llm, hook_tr, full_tr, video_path, shap_up, shap_down,
        )

        elapsed = round(time.time() - t_start, 2)
        log.info(
            "Done in %.1fs | tier=%s | score=%d",
            elapsed, report.get("viral_tier", "?"), _log_views_to_score(log_views_pred),
        )

        return {
            **report,
            "viral_score":             _log_views_to_score(log_views_pred),
            "processing_time_seconds": elapsed,
            "title":                   title,
            "platform":                platform,
        }
