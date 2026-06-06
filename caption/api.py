"""
FastAPI — Speech-to-Text Caption Pipeline
==========================================
Wraps the faster-whisper transcription + ASS/SRT generation used in the Colab notebook.
Designed to be called from a Node.js dashboard via HTTP.

Install deps:
    pip install fastapi uvicorn faster-whisper python-multipart aiofiles torch

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import re
import shutil
import subprocess
import tempfile
import pickle
from pathlib import Path
from typing import Literal

import torch
import aiofiles
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Speech-to-Text Caption API",
    description="Transcribes video/audio and returns SRT, ASS subtitles and optionally a burned video.",
    version="1.0.0",
)

# Allow your Node dashboard origin (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp output directory (created on startup)
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Utility helpers (ported from notebook)
# ─────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    """Seconds → SRT timestamp  HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def get_safe_font_size(text: str, video_width: int = 1080,
                       max_size: int = 72, min_size: int = 36) -> int:
    safe_width = video_width * 0.82
    longest = max(text.split(), key=len) if text.strip() else text
    chars = len(longest)
    ideal = int(safe_width / (chars * 0.62))
    return max(min_size, min(max_size, ideal))


def extract_audio(video_path: str, audio_output_path: str):
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_output_path,
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe_audio(audio_path: str, model_size: str = "small"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    return list(segments), info


def generate_srt(segments, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{seg.text.strip()}\n\n")


def make_word_level_srt(segments, output_path: str, words_per_flash: int = 2):
    idx = 1
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            words = segment.words
            if not words:
                continue
            chunks = [words[i:i + words_per_flash] for i in range(0, len(words), words_per_flash)]
            for chunk in chunks:
                start = format_timestamp(chunk[0].start)
                end = format_timestamp(chunk[-1].end)
                text = " ".join(w.word.strip() for w in chunk)
                f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
                idx += 1


def srt_to_ass_dynamic(srt_path: str, ass_path: str,
                        style: str = "stroke",
                        video_width: int = 1080,
                        video_height: int = 1920):
    style_defs = {
        "stroke": "Arial Black,72,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,5,0,2,10,10,120,1",
        "yellow": "Arial Black,72,&H00FFFFFF,&H00FFE500,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,4,2,2,10,10,120,1",
        "pill":   "Arial Black,72,&H00FFFFFF,&H000000FF,&HAA000000,&H00000000,1,0,0,0,100,100,0,0,4,0,0,2,10,10,120,1",
    }
    chosen = style_defs.get(style, style_defs["stroke"])
    header = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {video_width}\nPlayResY: {video_height}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, "
        "Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{chosen}\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    def srt_to_ass_time(t):
        hh, mm, rest = t.split(":")
        ss, ms = rest.split(",")
        return f"{int(hh)}:{mm}:{ss}.{ms[:2]}"

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    events = []
    for block in re.split(r"\n\s*\n", content):
        lines = block.strip().splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_s, end_s = [x.strip() for x in lines[1].split("-->")]
        text = " ".join(lines[2:]).strip()
        if not text:
            continue
        fs = get_safe_font_size(text, video_width=video_width)
        ass_text = f"{{\\fs{fs}\\an2}}" + text.upper()
        events.append(f"Dialogue: 0,{srt_to_ass_time(start_s)},{srt_to_ass_time(end_s)},Default,,0,0,0,,{ass_text}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))


def burn_ass_subtitles(video_path: str, ass_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"ass='{ass_path}':fontsdir=/usr/share/fonts",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy", "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def save_pkl(data: dict, pkl_path: str):
    """Persist segments + metadata to a pickle file for later reuse."""
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)


def load_pkl(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Speech-to-Text Caption API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/transcribe", tags=["Pipeline"])
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video or audio file (mp4, mov, avi, mp3, wav)"),
    model_size: Literal["tiny", "base", "small", "medium", "large"] = Form("small"),
    words_per_flash: int = Form(2, ge=1, le=10),
    subtitle_style: Literal["stroke", "yellow", "pill"] = Form("pill"),
    video_width: int = Form(1080),
    video_height: int = Form(1920),
    burn_subtitles: bool = Form(True),
    save_pickle: bool = Form(True),
):
    """
    Full pipeline endpoint.
    Upload a video → get back JSON with download URLs for:
    - SRT file
    - Word-level SRT
    - ASS file
    - Burned video (optional)
    - PKL file (optional)

    Node.js example:
        const form = new FormData();
        form.append('file', fs.createReadStream('video.mp4'));
        form.append('model_size', 'small');
        form.append('burn_subtitles', 'true');
        const res = await axios.post('http://localhost:8000/transcribe', form, { headers: form.getHeaders() });
    """
    # ── Save uploaded file ──────────────────────
    suffix = Path(file.filename).suffix or ".mp4"
    tmp_dir = tempfile.mkdtemp(dir=OUTPUT_DIR)
    tmp_path = Path(tmp_dir)

    input_path = tmp_path / f"input{suffix}"
    async with aiofiles.open(input_path, "wb") as out:
        content = await file.read()
        await out.write(content)

    try:
        # ── Extract audio ────────────────────────
        audio_path = tmp_path / "audio.wav"
        extract_audio(str(input_path), str(audio_path))

        # ── Transcribe ───────────────────────────
        segments, info = transcribe_audio(str(audio_path), model_size=model_size)

        # ── Save PKL ────────────────────────────
        pkl_url = None
        if save_pickle:
            pkl_path = tmp_path / "segments.pkl"
            save_pkl({"segments": segments, "info": info.__dict__}, str(pkl_path))
            pkl_url = f"/download/{tmp_path.name}/segments.pkl"

        # ── Generate SRT files ───────────────────
        srt_path = tmp_path / "subtitles.srt"
        word_srt_path = tmp_path / "subtitles_words.srt"
        generate_srt(segments, str(srt_path))
        make_word_level_srt(segments, str(word_srt_path), words_per_flash=words_per_flash)

        # ── Generate ASS ─────────────────────────
        ass_path = tmp_path / "subtitles_words.ass"
        srt_to_ass_dynamic(
            str(word_srt_path), str(ass_path),
            style=subtitle_style,
            video_width=video_width,
            video_height=video_height,
        )

        # ── Burn subtitles ───────────────────────
        video_url = None
        if burn_subtitles:
            output_video = tmp_path / "output_captioned.mp4"
            burn_ass_subtitles(str(input_path), str(ass_path), str(output_video))
            video_url = f"/download/{tmp_path.name}/output_captioned.mp4"

        # ── Build full transcript text ────────────
        full_transcript = " ".join(seg.text.strip() for seg in segments)

        return JSONResponse({
            "status": "success",
            "language": info.language,
            "duration_seconds": round(info.duration, 2),
            "segment_count": len(segments),
            "transcript": full_transcript,
            "downloads": {
                "srt": f"/download/{tmp_path.name}/subtitles.srt",
                "word_srt": f"/download/{tmp_path.name}/subtitles_words.srt",
                "ass": f"/download/{tmp_path.name}/subtitles_words.ass",
                "video": video_url,
                "pkl": pkl_url,
            },
        })

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/json-only", tags=["Pipeline"])
async def transcribe_json_only(
    file: UploadFile = File(...),
    model_size: Literal["tiny", "base", "small", "medium", "large"] = Form("small"),
):
    """
    Lightweight endpoint — returns only the transcript JSON (no subtitle files, no video).
    Useful when your Node dashboard only needs the text.
    """
    suffix = Path(file.filename).suffix or ".mp4"
    tmp_dir = tempfile.mkdtemp(dir=OUTPUT_DIR)
    input_path = Path(tmp_dir) / f"input{suffix}"
    audio_path = Path(tmp_dir) / "audio.wav"

    async with aiofiles.open(input_path, "wb") as out:
        await out.write(await file.read())

    try:
        extract_audio(str(input_path), str(audio_path))
        segments, info = transcribe_audio(str(audio_path), model_size=model_size)
        return {
            "language": info.language,
            "duration_seconds": round(info.duration, 2),
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
                for seg in segments
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{folder}/{filename}", tags=["Files"])
def download_file(folder: str, filename: str):
    """Download a generated file by folder/filename returned from /transcribe."""
    file_path = OUTPUT_DIR / folder / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path), filename=filename)


@app.delete("/cleanup/{folder}", tags=["Files"])
def cleanup(folder: str):
    """Delete a temp output folder once the client has downloaded everything."""
    folder_path = OUTPUT_DIR / folder
    if folder_path.exists():
        shutil.rmtree(folder_path)
        return {"status": "deleted", "folder": folder}
    raise HTTPException(status_code=404, detail="Folder not found")
