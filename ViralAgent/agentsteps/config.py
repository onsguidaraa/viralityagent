"""
config.py — Central configuration for the Viral Agent pipeline.
Edit paths and hyperparameters here. All scripts import from this file.
"""

import os

BASE = os.environ.get("VIRAL_AGENT_BASE", "/content/drive/MyDrive/viral_agent")

DATASET_DIR  = f"{BASE}/datasets"
METADATA_DIR = f"{DATASET_DIR}/metadata"
VIDEO_DIR    = f"{DATASET_DIR}/videos"
AUDIO_DIR    = f"{DATASET_DIR}/audios"          # extracted .wav files

FEATURE_DIR      = f"{BASE}/features"
HOOK_EMB_DIR     = f"{FEATURE_DIR}/hook_embeddings"
FULL_EMB_DIR     = f"{FEATURE_DIR}/full_video_embeddings"
TEXT_EMB_DIR     = f"{FEATURE_DIR}/text_embeddings"
AUDIO_FEAT_DIR   = f"{FEATURE_DIR}/audio_features"
WHISPER_DIR      = f"{FEATURE_DIR}/whisper_transcripts"
LLM_DIR          = f"{FEATURE_DIR}/llm_features"
PCA_DIR          = f"{FEATURE_DIR}/pca_reduced"

MODEL_DIR        = f"{BASE}/models"
RESULTS_DIR      = f"{BASE}/results"

ALL_METADATA_CSV     = f"{METADATA_DIR}/all_metadata.csv"
EMBEDDING_INDEX_CSV  = f"{FEATURE_DIR}/embedding_index.csv"
TEXT_INDEX_CSV       = f"{FEATURE_DIR}/text_embedding_index.csv"
AUDIO_INDEX_CSV      = f"{FEATURE_DIR}/audio_feature_index.csv"
WHISPER_INDEX_CSV    = f"{FEATURE_DIR}/whisper_transcript_index.csv"
FINAL_DATASET_PATH   = f"{FEATURE_DIR}/final_model_dataset.parquet"

# Feature Extraction 
HOOK_SECONDS  = 3          # seconds to consider as "hook"
HOOK_MAX_FRAMES = 12       # frames sampled from hook
FULL_MAX_FRAMES = 24       # frames sampled from full video
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_DIM = 512

TEXT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_DIM = 384

WHISPER_MODEL_SIZE = "base"   # tiny | base | small | medium | large

# PCA 
PCA_HOOK_COMPONENTS  = 64
PCA_FULL_COMPONENTS  = 64
PCA_TEXT_COMPONENTS  = 48

# LLM (Groq) 
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_RATE_LIMIT_SLEEP = 0.5   # seconds between API calls

# Training 
TARGET_COL   = "log_views"
# Columns that MUST be excluded (post-publication data leakage)
LEAKY_COLS   = ["likes", "comments", "views", "log_views", "video_id", "platform", "shares"]
# NEW (Safe)
META_COLS    = ["video_id", "platform", "duration_seconds", "log_views"]
# META_COLS    = ["video_id", "platform", "views", "likes", "comments", "duration_seconds", "log_views"]

CV_FOLDS     = 5
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# XGBoost defaults
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# LightGBM defaults
LGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# MLP defaults
MLP_HIDDEN_DIMS  = [512, 256, 128]
MLP_DROPOUT      = 0.3
MLP_EPOCHS       = 100
MLP_BATCH_SIZE   = 32
MLP_LR           = 1e-3

# Helpers 
def make_dirs():
    """Create all required directories."""
    dirs = [
        METADATA_DIR, VIDEO_DIR, AUDIO_DIR,
        HOOK_EMB_DIR, FULL_EMB_DIR, TEXT_EMB_DIR,
        AUDIO_FEAT_DIR, WHISPER_DIR, LLM_DIR, PCA_DIR,
        MODEL_DIR, RESULTS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("All directories ready.")
