"""
AutoClip Studio — FastAPI Server
Serves the web UI and handles the video processing API.
"""

import logging
import os
import sys
import uuid
import threading
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Pre-load core modules to speed up initialization
from core.pipeline import jobs, JobStatus, run_pipeline


# ── Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-22s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("autoclip.server")

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(title="AutoClip Studio", version="1.0.0")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = STATIC_DIR / "output"
TEMPLATES_DIR = BASE_DIR / "templates"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Request/Response Models ────────────────────────────────────────
class GenerateRequest(BaseModel):
    url: str
    model: str = "large-v3"
    duration: float = 60.0
    viral_threshold: float = 5.0
    resolution: str = "1080"
    max_clips: int = 10


class GenerateResponse(BaseModel):
    job_id: str
    message: str


class VideoInfoRequest(BaseModel):
    url: str


class VideoInfoResponse(BaseModel):
    title: str
    duration: int
    thumbnail: str
    channel: str
    estimated_time: str


# ── Routes ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/video-info", response_model=VideoInfoResponse)
async def video_info(req: VideoInfoRequest):
    """Fetch video metadata and estimate processing time."""
    url = req.url.strip()
    if not url or ("youtube.com" not in url and "youtu.be" not in url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

    from core.downloader import get_video_info
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(None, get_video_info, url)
        
        dur_secs = info.get("duration", 0)
        # Estimate: Download (~15s) + Transcribe (duration / 10) + Render (approx 90s)
        est_secs = 15 + (dur_secs * 0.1) + 90
        
        if est_secs < 60:
            est_str = f"~{int(est_secs)} seconds"
        else:
            mins = int(est_secs // 60)
            est_str = f"~{mins} min {int(est_secs % 60)} sec"
            
        info["estimated_time"] = est_str
        return VideoInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Start a video processing job in the background."""
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")

    if "youtube.com" not in url and "youtu.be" not in url:
        raise HTTPException(status_code=400, detail="Please provide a valid YouTube URL.")


    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = JobStatus(job_id=job_id)

    # Run pipeline in a background thread (not asyncio — the pipeline is CPU-bound)
    thread = threading.Thread(
        target=run_pipeline,
        kwargs={
            "job_id": job_id,
            "url": url,
            "output_dir": str(OUTPUT_DIR),
            "model_size": req.model,
            "clip_duration": req.duration,
            "viral_threshold": req.viral_threshold,
            "resolution": req.resolution,
            "max_clips": req.max_clips,
        },
        daemon=True,
    )
    thread.start()

    logger.info(f"Job {job_id} started for URL: {url}")
    return GenerateResponse(job_id=job_id, message="Processing started!")


@app.get("/api/jobs")
async def get_jobs():
    """Fetch all historical jobs."""
    from dataclasses import asdict

    sorted_jobs = sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)
    return JSONResponse({"jobs": [asdict(j) for j in sorted_jobs]})


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll for job status."""
    from dataclasses import asdict


    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return JSONResponse(asdict(job))


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
