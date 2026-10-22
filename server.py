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
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
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
    source_lang: Optional[str] = "hi-en"
    tone_type: Optional[str] = "neutral"


@app.post("/api/process")
async def process_audio(
    audio: UploadFile = File(...),
    target_lang: str = Form("en"),
    source_lang: Optional[str] = Form("hi-en"),
    override_transcript: Optional[str] = Form(None),
    live_transcript: Optional[str] = Form(None)
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

        clean_live = live_transcript.strip() if (live_transcript and live_transcript.strip()) else None

        # Execute 7-stage ML pipeline on the user's actual audio
        result = pipeline.process(
            audio_path=wav_path,
            target_lang=target_lang,
            source_lang=source_lang,
            override_transcript=override_transcript,
            client_live_transcript=clean_live
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


@app.get("/api/health")
async def health_check():
    """Health check endpoint for frontend and diagnostic verification."""
    return {"status": "ok", "app": "Bhasha Live", "version": "2.0.0"}


@app.post("/api/process-text")
async def process_text(request: TextProcessRequest):
    """
    Processes directly typed text through the pipeline with fast in-memory execution.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    target_lang = request.target_lang or "en"
    source_lang = request.source_lang or "hi-en"

    try:
        result = pipeline.process_text(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            tone_type=request.tone_type or "neutral"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Static and Frontend Routes
base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(base_dir, "frontend")

@app.get("/")
@app.get("/app")
@app.get("/demo")
@app.get("/index")
@app.get("/home")
@app.get("/dashboard")
async def serve_frontend():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="index.html not found")

# Mount static asset directories
if os.path.exists(os.path.join(frontend_dir, "css")):
    app.mount("/css", StaticFiles(directory=os.path.join(frontend_dir, "css")), name="css")
if os.path.exists(os.path.join(frontend_dir, "js")):
    app.mount("/js", StaticFiles(directory=os.path.join(frontend_dir, "js")), name="js")
if os.path.exists(os.path.join(frontend_dir, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dir, "assets")), name="assets")

# Mount entire frontend for any remaining static resources
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    import socket
    import argparse

    parser = argparse.ArgumentParser(description="Bhasha Live Multilingual Voice & Text Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port to bind server to")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--ssl", action="store_true", default=False, help="Enable HTTPS using cert.pem/key.pem (default: disabled for clean localhost HTTP)")
    args, _ = parser.parse_known_args()

    def find_free_port(start_port: int, max_attempts: int = 20) -> int:
        for port in range(start_port, start_port + max_attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        return start_port

    target_port = find_free_port(args.port)
    if target_port != args.port:
        print(f"[Notice] Port {args.port} is currently in use by another process. Auto-switching to port {target_port}.")

    cert_file = os.path.join(base_dir, "cert.pem")
    key_file = os.path.join(base_dir, "key.pem")
    use_ssl = bool(args.ssl and os.path.exists(cert_file) and os.path.exists(key_file))

    protocol = "https" if use_ssl else "http"
    print("\n==================================================")
    print(f"🚀 Bhasha Live Web Server Ready! ({'SSL / HTTPS Enabled' if use_ssl else 'HTTP'})")
    print(f"👉 Open in browser: {protocol}://localhost:{target_port}")
    print(f"👉 Alternative URL: {protocol}://127.0.0.1:{target_port}")
    print("==================================================\n")

    if use_ssl:
        uvicorn.run(app, host=args.host, port=target_port, ssl_keyfile=key_file, ssl_certfile=cert_file)
    else:
        uvicorn.run(app, host=args.host, port=target_port)

