"""
Bhasha Live — Real-Time Production Backend Server
Serves:
  1. Frontend Single Page Application at /
  2. POST /api/process — Processes real microphone audio uploads through the 7-stage ML pipeline
  3. POST /api/process-text — Processes typed code-mixed sentences directly
"""

import os
import sys
import uuid
import wave
import shutil
import tempfile
import numpy as np
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import MultilingualVoicePipeline
from pipeline.utils.audio_utils import generate_synthetic_audio

app = FastAPI(title="Bhasha Live API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize deterministic 7-stage ML pipeline
pipeline = MultilingualVoicePipeline(config_path="config/default_config.yaml")

os.makedirs("data/uploads", exist_ok=True)


class TextProcessRequest(BaseModel):
    text: str
    target_lang: str = "en"
    tone_type: Optional[str] = "neutral"


@app.post("/api/process")
async def process_audio(
    audio: UploadFile = File(...),
    target_lang: str = Form("en"),
    override_transcript: Optional[str] = Form(None)
):
    """
    Receives real audio recording from browser microphone or file upload,
    runs deterministic MultilingualVoicePipeline, and returns full structured JSON.
    """
    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(audio.filename)[1].lower() if audio.filename else ".wav"
    if not ext or ext not in [".wav", ".mp3", ".ogg", ".webm", ".m4a"]:
        ext = ".wav"
    
    input_path = os.path.join("data/uploads", f"user_{file_id}{ext}")
    wav_path = os.path.join("data/uploads", f"user_{file_id}.wav")

    try:
        # Save incoming audio stream
        with open(input_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        # If incoming file is already WAV, use directly
        if ext == ".wav":
            wav_path = input_path
        else:
            # Convert to standard 16kHz mono WAV using system afconvert or fallback
            if shutil.which("afconvert"):
                os.system(f"afconvert -f WAVE -d LEI16 -c 1 -r 16000 '{input_path}' '{wav_path}'")
            elif shutil.which("ffmpeg"):
                os.system(f"ffmpeg -y -i '{input_path}' -ar 16000 -ac 1 '{wav_path}'")
            else:
                wav_path = input_path

        # If override_transcript was sent as empty string, set to None
        if override_transcript and not override_transcript.strip():
            override_transcript = None

        # Execute 7-stage ML pipeline on the user's actual audio
        result = pipeline.process(
            audio_path=wav_path,
            target_lang=target_lang,
            override_transcript=override_transcript
        )

        return result

    except Exception as e:
        print(f"[API Error] Audio processing failed: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        for p in [input_path, wav_path]:
            if p and os.path.exists(p) and p != input_path:
                try:
                    os.remove(p)
                except Exception:
                    pass


@app.post("/api/process-text")
async def process_text(request: TextProcessRequest):
    """
    Processes directly typed text through the pipeline with matching audio synthesis.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    target_lang = request.target_lang or "en"
    temp_audio = f"data/uploads/temp_text_{uuid.uuid4().hex[:8]}.wav"

    try:
        # Generate acoustic speech sample for this text
        generate_synthetic_audio(temp_audio, duration_sec=3.0, tone_type=request.tone_type or "neutral")

        # Run pipeline with the user's custom typed text
        result = pipeline.process(
            audio_path=temp_audio,
            target_lang=target_lang,
            override_transcript=text
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception:
                pass


# Mount static frontend files
frontend_dir = os.path.abspath("frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
