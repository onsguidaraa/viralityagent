"""
02_feature_extraction.py
─────────────────────────
Extracts ALL features from raw video files:

  A. Visual — CLIP embeddings  (hook + full video)
  B. Audio  — Whisper ASR      (hook + full transcript)
             — Librosa acoustic features (energy, tempo, spectral…)
  C. Text   — SentenceTransformer (title + hook_transcript + full_transcript)
  D. LLM    — Groq analysis    (title + transcript → structured JSON)

Each feature type is saved to its own directory and indexed in a CSV.
Failed videos are logged separately so you can retry them.

Usage:
    python 02_feature_extraction.py                   # run all steps
    python 02_feature_extraction.py --steps clip whisper text llm
    python 02_feature_extraction.py --steps clip       # only CLIP

Requirements:
    pip install torch transformers pillow opencv-python tqdm
    pip install openai-whisper librosa soundfile
    pip install sentence-transformers groq
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import whisper
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info("Using device: %s", DEVICE)


# ═════════════════════════════════════════════════════════════════════════════
# A. VISUAL — CLIP embeddings
# ═════════════════════════════════════════════════════════════════════════════

def load_clip():
    log.info("Loading CLIP model: %s", config.CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    model.eval()
    return model, processor


def extract_frames(video_path: str, max_frames: int, end_second: float | None = None) -> list[np.ndarray]:
    """Sample up to `max_frames` RGB frames from a video, optionally clipping at `end_second`."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_second is not None:
        last_frame = min(int(end_second * fps), total_frames - 1)
    else:
        last_frame = total_frames - 1

    if last_frame <= 0:
        cap.release()
        return []

    indices = np.linspace(0, last_frame, min(max_frames, last_frame + 1), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def encode_frames(frames: list[np.ndarray], clip_model, clip_processor) -> np.ndarray:
    """
    Returns a 1-D (512,) normalized embedding by:
      1. Encoding each frame with CLIP vision tower
      2. Applying the visual projection head
      3. Normalizing and mean-pooling
      4. Giving the first frame a higher weight (captures opening shot)
    """
    images = [Image.fromarray(f) for f in frames]
    inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        feats = clip_model.visual_projection(out.pooler_output)   # (N, 512)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    feats = feats.cpu().numpy()                                    # (N, 512)

    # Weight first frame 2× more — opening shot matters most for hooks
    weights = np.ones(len(frames))
    weights[0] = 2.0
    weights /= weights.sum()

    embedding = (feats * weights[:, None]).sum(axis=0)
    embedding /= np.linalg.norm(embedding) + 1e-8
    return embedding.astype(np.float32)


def run_clip_extraction(df: pd.DataFrame, clip_model, clip_processor) -> pd.DataFrame:
    """Extract CLIP embeddings for all videos. Returns an index DataFrame."""
    records, failed = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CLIP embeddings"):
        vid = str(row["video_id"])
        vpath = str(row.get("local_video_path", ""))

        hook_path = os.path.join(config.HOOK_EMB_DIR, f"{vid}_hook.npy")
        full_path = os.path.join(config.FULL_EMB_DIR, f"{vid}_full.npy")

        if os.path.isfile(hook_path) and os.path.isfile(full_path):
            records.append({"video_id": vid, "hook_path": hook_path, "full_path": full_path})
            continue

        if not os.path.isfile(vpath):
            failed.append({"video_id": vid, "reason": "video file not found"})
            continue

        try:
            hook_frames = extract_frames(vpath, config.HOOK_MAX_FRAMES, end_second=config.HOOK_SECONDS)
            full_frames = extract_frames(vpath, config.FULL_MAX_FRAMES)

            if not hook_frames or not full_frames:
                raise ValueError("No frames extracted")

            hook_emb = encode_frames(hook_frames, clip_model, clip_processor)
            full_emb = encode_frames(full_frames, clip_model, clip_processor)

            np.save(hook_path, hook_emb)
            np.save(full_path, full_emb)

            records.append({"video_id": vid, "hook_path": hook_path, "full_path": full_path})

        except Exception as e:
            failed.append({"video_id": vid, "reason": str(e)})
            log.debug("CLIP failed for %s: %s", vid, e)

    log.info("CLIP: processed=%d  failed=%d", len(records), len(failed))
    _save_failures(failed, "clip_failures.csv")

    index = pd.DataFrame(records)
    index.to_csv(config.EMBEDDING_INDEX_CSV, index=False)
    return index


# ═════════════════════════════════════════════════════════════════════════════
# B. AUDIO — Whisper ASR + Librosa acoustic features
# ═════════════════════════════════════════════════════════════════════════════

def load_whisper_model():
    log.info("Loading Whisper model: %s", config.WHISPER_MODEL_SIZE)
    return whisper.load_model(config.WHISPER_MODEL_SIZE)


def extract_audio_wav(video_path: str, out_path: str) -> bool:
    """Extract audio track from video to a 16-kHz mono WAV file."""
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000",
        "-vn", out_path,
        "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and os.path.isfile(out_path)


def run_whisper_transcription(df: pd.DataFrame, whisper_model) -> pd.DataFrame:
    """Transcribe audio with Whisper. Saves hook + full transcripts as JSON."""
    records, failed = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Whisper ASR"):
        vid = str(row["video_id"])
        vpath = str(row.get("local_video_path", ""))
        save_path = os.path.join(config.WHISPER_DIR, f"{vid}_transcript.json")

        if os.path.isfile(save_path):
            records.append({"video_id": vid, "transcript_path": save_path})
            continue

        if not os.path.isfile(vpath):
            failed.append({"video_id": vid, "reason": "video file not found"})
            continue

        try:
            # Extract WAV
            wav_path = os.path.join(config.AUDIO_DIR, f"{vid}.wav")
            if not os.path.isfile(wav_path):
                ok = extract_audio_wav(vpath, wav_path)
                if not ok:
                    raise IOError("ffmpeg audio extraction failed")

            # Full transcription
            result = whisper_model.transcribe(wav_path, fp16=(DEVICE == "cuda"))
            full_text = result["text"].strip()
            full_segments = result.get("segments", [])

            # Hook transcript = segments within first HOOK_SECONDS
            hook_segs = [s for s in full_segments if s["start"] < config.HOOK_SECONDS]
            hook_text = " ".join(s["text"] for s in hook_segs).strip()

            data = {
                "full_transcript": full_text,
                "hook_transcript": hook_text,
                "language": result.get("language", ""),
                "segments": full_segments,
            }

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            records.append({"video_id": vid, "transcript_path": save_path})

        except Exception as e:
            failed.append({"video_id": vid, "reason": str(e)})
            log.debug("Whisper failed for %s: %s", vid, e)

    log.info("Whisper: processed=%d  failed=%d", len(records), len(failed))
    _save_failures(failed, "whisper_failures.csv")

    index = pd.DataFrame(records)
    index.to_csv(config.WHISPER_INDEX_CSV, index=False)
    return index


def extract_acoustic_features(wav_path: str) -> dict:
    """
    Extract acoustic features with librosa.
    Returns a flat dict of scalars — all safe for a DataFrame column.
    """
    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    if len(y) < sr * 0.5:      # shorter than 0.5s → skip
        return {}

    duration = librosa.get_duration(y=y, sr=sr)

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # Spectral centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Spectral rolloff
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    # MFCC — first 13 means
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Silence ratio (frames below -40 dB)
    stft = np.abs(librosa.stft(y))
    db = librosa.amplitude_to_db(stft, ref=np.max)
    silence_ratio = (db < -40).mean()

    feats = {
        "duration_s":         float(duration),
        "rms_mean":           float(rms.mean()),
        "rms_std":            float(rms.std()),
        "rms_max":            float(rms.max()),
        "zcr_mean":           float(zcr.mean()),
        "spec_centroid_mean": float(spec_cent.mean()),
        "spec_rolloff_mean":  float(spec_roll.mean()),
        "tempo_bpm":          float(tempo),
        "silence_ratio":      float(silence_ratio),
    }
    for i, v in enumerate(mfcc_means):
        feats[f"mfcc_{i}"] = float(v)

    return feats


def run_acoustic_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """Extract librosa features for all videos. Returns an index DataFrame."""
    records, failed = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Acoustic features"):
        vid = str(row["video_id"])
        vpath = str(row.get("local_video_path", ""))
        save_path = os.path.join(config.AUDIO_FEAT_DIR, f"{vid}_audio.json")

        if os.path.isfile(save_path):
            records.append({"video_id": vid, "audio_feat_path": save_path})
            continue

        if not os.path.isfile(vpath):
            failed.append({"video_id": vid, "reason": "video file not found"})
            continue

        try:
            wav_path = os.path.join(config.AUDIO_DIR, f"{vid}.wav")
            if not os.path.isfile(wav_path):
                ok = extract_audio_wav(vpath, wav_path)
                if not ok:
                    raise IOError("ffmpeg extraction failed")

            feats = extract_acoustic_features(wav_path)
            if not feats:
                raise ValueError("Audio too short")

            with open(save_path, "w") as f:
                json.dump(feats, f, indent=2)

            records.append({"video_id": vid, "audio_feat_path": save_path})

        except Exception as e:
            failed.append({"video_id": vid, "reason": str(e)})
            log.debug("Acoustic failed for %s: %s", vid, e)

    log.info("Acoustic: processed=%d  failed=%d", len(records), len(failed))
    _save_failures(failed, "acoustic_failures.csv")

    index = pd.DataFrame(records)
    index.to_csv(config.AUDIO_INDEX_CSV, index=False)
    return index


# ═════════════════════════════════════════════════════════════════════════════
# C. TEXT — SentenceTransformer embeddings
# ═════════════════════════════════════════════════════════════════════════════

def load_text_model():
    log.info("Loading text model: %s", config.TEXT_MODEL_NAME)
    return SentenceTransformer(config.TEXT_MODEL_NAME)


def _build_combined_text(title: str, hook_transcript: str, full_transcript: str) -> str:
    return (
        f"Title: {title}\n\n"
        f"Hook (first {config.HOOK_SECONDS}s):\n{hook_transcript}\n\n"
        f"Full transcript:\n{full_transcript}"
    ).strip()


def run_text_extraction(df: pd.DataFrame, text_model, whisper_index: pd.DataFrame) -> pd.DataFrame:
    """Encode title + transcripts into a single text embedding per video."""
    # Join transcripts
    df_text = df.merge(whisper_index, on="video_id", how="left")

    records, failed = [], []

    for _, row in tqdm(df_text.iterrows(), total=len(df_text), desc="Text embeddings"):
        vid = str(row["video_id"])
        save_path = os.path.join(config.TEXT_EMB_DIR, f"{vid}_text.npy")

        if os.path.isfile(save_path):
            records.append({"video_id": vid, "text_path": save_path})
            continue

        try:
            title = str(row.get("title", "") or "")

            hook_tr = ""
            full_tr = ""
            t_path = row.get("transcript_path", "")
            if pd.notna(t_path) and os.path.isfile(str(t_path)):
                with open(t_path, "r", encoding="utf-8") as f:
                    td = json.load(f)
                hook_tr = td.get("hook_transcript", "")
                full_tr = td.get("full_transcript", "")

            combined = _build_combined_text(title, hook_tr, full_tr)
            emb = text_model.encode(combined, normalize_embeddings=True)
            np.save(save_path, emb.astype(np.float32))

            records.append({"video_id": vid, "text_path": save_path})

        except Exception as e:
            failed.append({"video_id": vid, "reason": str(e)})
            log.debug("Text emb failed for %s: %s", vid, e)

    log.info("Text: processed=%d  failed=%d", len(records), len(failed))
    _save_failures(failed, "text_failures.csv")

    index = pd.DataFrame(records)
    index.to_csv(config.TEXT_INDEX_CSV, index=False)
    return index


# ═════════════════════════════════════════════════════════════════════════════
# D. LLM — Groq structured analysis (NOW WITH TRANSCRIPT)
# ═════════════════════════════════════════════════════════════════════════════

def _build_groq_prompt(title: str, platform: str, duration_s, hook_transcript: str, full_transcript: str) -> str:
    return f"""You are an expert viral short-form video analyst.

Video information:
- Platform: {platform}
- Title / caption: {title}
- Duration: {duration_s} seconds
- Hook transcript (first {config.HOOK_SECONDS}s): {hook_transcript or '(no speech detected)'}
- Full audio transcript: {full_transcript or '(no speech detected)'}

Based on ALL the above information, return ONLY valid JSON with this exact structure:

{{
  "hook_type": "curiosity | shock | tutorial | emotional | transformation | comedy | other",
  "hook_score": 0,
  "clarity_score": 0,
  "emotion": "string describing dominant emotion",
  "tone": "educational | entertaining | inspirational | controversial | relatable | other",
  "content_category": "string",
  "engagement_triggers": ["trigger1", "trigger2"],
  "strengths": ["strength1", "strength2"],
  "weaknesses": ["weakness1", "weakness2"],
  "quality_score": 0,
  "improvement_suggestion": "string"
}}

All scores are integers from 0 to 10. Return ONLY the JSON object, no markdown."""


import os
import json
import time
import pandas as pd
from tqdm import tqdm

def run_llm_analysis(df: pd.DataFrame, whisper_index: pd.DataFrame) -> pd.DataFrame:
    """
    Run Groq LLM on title + transcript for each video.
    Uses a hardcoded API key.
    """
    try:
        from groq import Groq
    except ImportError:
        log.error("groq not installed. Run: pip install groq")
        return pd.DataFrame()

    # API key hardcoded directly into the function
    api_key = "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"

    client = Groq(api_key=api_key)
    df_llm = df.merge(whisper_index, on="video_id", how="left")

    records, failed = [], []
    
    # Ensure the directory exists before we try to save files into it
    os.makedirs(config.LLM_DIR, exist_ok=True)

    for _, row in tqdm(df_llm.iterrows(), total=len(df_llm), desc="LLM analysis"):
        vid = str(row["video_id"])
        save_path = os.path.join(config.LLM_DIR, f"{vid}_llm.json")

        # Check for cached result
        if os.path.isfile(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                records.append({"video_id": vid, **saved})
                continue
            except Exception as e:
                log.warning(f"Corrupted cache file for {vid}, re-running. Error: {e}")
                pass # re-run if file is corrupted

        try:
            title      = str(row.get("title", "") or "")
            platform   = str(row.get("platform", "unknown") or "unknown")
            duration_s = row.get("duration_seconds", None)

            hook_tr = full_tr = ""
            t_path = row.get("transcript_path", "")
            if pd.notna(t_path) and os.path.isfile(str(t_path)):
                with open(t_path, "r", encoding="utf-8") as f:
                    td = json.load(f)
                hook_tr = td.get("hook_transcript", "")
                full_tr = td.get("full_transcript", "")

            prompt = _build_groq_prompt(title, platform, duration_s, hook_tr, full_tr)

            response = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a strict JSON generator and expert viral video analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)
            result["video_id"] = vid

            # Save the new result
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            records.append(result)

        except Exception as e:
            failed.append({"video_id": vid, "reason": str(e)})
            log.debug("LLM failed for %s: %s", vid, e)

        time.sleep(config.GROQ_RATE_LIMIT_SLEEP)

    log.info("LLM: processed=%d  failed=%d", len(records), len(failed))
    
    if failed:
        _save_failures(failed, "llm_failures.csv")
        
    return pd.DataFrame(records)


# ═════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════

def _save_failures(failed: list[dict], filename: str) -> None:
    if failed:
        path = os.path.join(config.FEATURE_DIR, filename)
        pd.DataFrame(failed).to_csv(path, index=False)
        log.info("Failure log → %s", path)


# ═════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════════

AVAILABLE_STEPS = ["clip", "whisper", "acoustic", "text", "llm"]


def parse_args():
    p = argparse.ArgumentParser(description="Extract all features from video dataset")
    p.add_argument(
        "--steps",
        nargs="+",
        choices=AVAILABLE_STEPS,
        default=AVAILABLE_STEPS,
        help=f"Which steps to run. Choices: {AVAILABLE_STEPS}",
    )
    return p.parse_args()


def main():
    args = parse_args()
    steps = set(args.steps)

    config.make_dirs()

    if not os.path.isfile(config.ALL_METADATA_CSV):
        log.error("Metadata CSV not found: %s\nRun data_tiktok.py and/or copy_of_youtube_shorts_ (1).py first.", config.ALL_METADATA_CSV)
        sys.exit(1)

    df = pd.read_csv(config.ALL_METADATA_CSV)
    df["video_id"] = df["video_id"].astype(str)
    log.info("Loaded metadata: %d videos", len(df))

    # ── A. CLIP ───────────────────────────────────────────────────────────
    if "clip" in steps:
        clip_model, clip_processor = load_clip()
        run_clip_extraction(df, clip_model, clip_processor)
        del clip_model, clip_processor
        torch.cuda.empty_cache()

    # ── B. Whisper ────────────────────────────────────────────────────────
    if "whisper" in steps:
        wmodel = load_whisper_model()
        run_whisper_transcription(df, wmodel)
        del wmodel
        torch.cuda.empty_cache()

    # ── B2. Acoustic ──────────────────────────────────────────────────────
    if "acoustic" in steps:
        run_acoustic_extraction(df)

    # ── C. Text embeddings ────────────────────────────────────────────────
    if "text" in steps:
        whisper_index = pd.DataFrame()
        if os.path.isfile(config.WHISPER_INDEX_CSV):
            whisper_index = pd.read_csv(config.WHISPER_INDEX_CSV)
            whisper_index["video_id"] = whisper_index["video_id"].astype(str)

        text_model = load_text_model()
        run_text_extraction(df, text_model, whisper_index)
        del text_model

    # ── D. LLM ────────────────────────────────────────────────────────────
    if "llm" in steps:
        whisper_index = pd.DataFrame()
        if os.path.isfile(config.WHISPER_INDEX_CSV):
            whisper_index = pd.read_csv(config.WHISPER_INDEX_CSV)
            whisper_index["video_id"] = whisper_index["video_id"].astype(str)

        run_llm_analysis(df, whisper_index)

    log.info("✅ Feature extraction complete.")


if __name__ == "__main__":
    main()
