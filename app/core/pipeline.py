"""
AutoClip Studio — Pipeline Orchestrator
Runs the full pipeline: download -> transcribe -> clip -> render.
Updates a shared job status dict so the API can report progress.
"""

import logging
import os
import shutil
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger("autoclip.pipeline")

# FFmpeg path handling
def get_ffmpeg_bin():
    # 1. Check if it's already in the PATH
    if shutil.which("ffmpeg"):
        return None # No need to override
    
    # 2. Try the winget install location on this machine
    winget_path = r"C:\Users\Space\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
    if os.path.exists(winget_path):
        return winget_path
    
    return None

FFMPEG_BIN = get_ffmpeg_bin()


@dataclass
class JobStatus:
    job_id: str
    status: str = "queued"           # queued | downloading | transcribing | clipping | rendering | complete | error
    progress_text: str = "Waiting to start..."
    progress_percent: float = 0.0
    estimated_time_remaining: float = 0.0
    current_step_name: str = ""
    current_action: str = ""
    logs: list[str] = field(default_factory=list)
    processing_speed: str = ""
    video_info: dict = field(default_factory=dict)
    video_urls: list[str] = field(default_factory=list)
    clip_details: list[dict] = field(default_factory=list)
    error: str = ""
    created_at: float = field(default_factory=time.time)

    def log(self, msg: str):
        self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        # Keep logs manageable
        if len(self.logs) > 100:
            self.logs.pop(0)

JOBS_FILE = Path("data/jobs.json")

def load_jobs() -> dict[str, JobStatus]:
    if not JOBS_FILE.exists():
        return {}
    try:
        with open(JOBS_FILE, "r") as f:
            data = json.load(f)
            return {k: JobStatus(**v) for k, v in data.items()}
    except Exception as e:
        logger.error(f"Failed to load jobs: {e}")
        return {}

def save_jobs(jobs_dict: dict[str, JobStatus]):
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(JOBS_FILE, "w") as f:
            json.dump({k: asdict(v) for k, v in jobs_dict.items()}, f)
    except Exception as e:
        logger.error(f"Failed to save jobs: {e}")

# In-memory job store
jobs: dict[str, JobStatus] = load_jobs()


def run_pipeline(
    job_id: str,
    url: str,
    output_dir: str,
    model_size: str = "large-v3",
    clip_duration: float = 60.0,
    viral_threshold: float = 5.0,
    resolution: str = "1080",
    max_clips: int = 10,
) -> None:
    """
    Execute the full pipeline. Updates jobs[job_id] as it progresses.
    Meant to be called from a background thread.
    """
    # Ensure FFmpeg is on PATH if we found a specific binary folder
    if FFMPEG_BIN:
        os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

    job = jobs[job_id]
    temp_dir = os.path.join("temp", f"job_{job_id}")

    try:
        from core.downloader import get_video_info
        try:
            job.video_info = get_video_info(url)
            job.log(f"Video identified: {job.video_info.get('title')}")
            save_jobs(jobs)
        except Exception as e:
            job.log(f"Could not fetch video info early: {e}")

        # ── STEP 1: Download ──────────────────────────────────────
        job.status = "downloading"
        job.current_step_name = "Download"
        job.progress_percent = 0.0
        job.progress_text = "Downloading video from YouTube..."
        job.log("Starting download sequence via yt-dlp...")
        logger.info(f"[{job_id}] Downloading...")

        from core.downloader import download_video, check_ffmpeg

        if not check_ffmpeg():
            raise RuntimeError("FFmpeg is not installed or not on PATH.")

        def dl_hook(percent, speed, eta):
            job.progress_percent = percent
            job.processing_speed = speed
            job.estimated_time_remaining = eta
            job.current_action = f"Downloading... {speed}"
            
        video_path, audio_path = download_video(url, temp_dir, progress_hook=dl_hook)
        job.log("Download and audio extraction completed successfully.")
        job.progress_percent = 100.0
        save_jobs(jobs)

        # ── STEP 2: Transcribe ────────────────────────────────────
        job.status = "transcribing"
        job.current_step_name = "Transcribe"
        job.progress_percent = 0.0
        job.estimated_time_remaining = (job.video_info.get("duration", 0) / 10) + 10 # Rough guess
        job.current_action = "Running faster-whisper..."
        job.progress_text = "Transcribing audio with AI (this may take a few minutes)..."
        job.log(f"Initializing Whisper model ({model_size}). First run may take a few mins to download...")
        save_jobs(jobs)
        logger.info(f"[{job_id}] Transcribing...")

        from core.transcriber import transcribe

        transcribe_start_time = time.time()
        def transcribe_hook(percent):
            job.progress_percent = percent
            job.current_action = f"Transcribing audio... {percent:.1f}%"
            if percent > 0.1:
                elapsed = time.time() - transcribe_start_time
                estimated_total = elapsed / (percent / 100.0)
                job.estimated_time_remaining = max(0, estimated_total - elapsed)

        result = transcribe(audio_path, model_size=model_size, language="en", progress_hook=transcribe_hook)
        if not result.words:
            raise RuntimeError("Transcription produced no words.")

        job.log(f"Transcription finished: {result.word_count} words.")
        job.progress_percent = 100.0
        job.progress_text = f"Transcribed {result.word_count} words in {result.duration:.0f}s of audio."
        save_jobs(jobs)

        # ── STEP 3: Find Clips ────────────────────────────────────
        job.status = "clipping"
        job.current_step_name = "Analyze & Clip"
        job.progress_percent = 0.0
        job.current_action = "Analyzing energy and text..."
        job.progress_text = "Analyzing audio energy + transcript for engaging moments..."
        job.log("Running 5-signal engagement scorer...")
        save_jobs(jobs)
        logger.info(f"[{job_id}] Finding clips...")

        from core.clipper import find_clips

        # Use the requested duration as max_duration, min_duration hardcoded to 15.0 per requirements
        clips = find_clips(result.words, audio_path=audio_path, max_duration=clip_duration, min_duration=15.0, viral_threshold=viral_threshold, max_clips=max_clips)
        if not clips:
            raise RuntimeError("No suitable clips found in the video.")

        job.progress_text = f"Found {len(clips)} clip(s). Starting render..."
        job.log(f"Clustering complete. Identified {len(clips)} high-viral-score clips.")
        job.progress_percent = 100.0
        save_jobs(jobs)

        # ── STEP 4: Render ────────────────────────────────────────
        job.status = "rendering"
        job.current_step_name = "Render"
        job.progress_percent = 0.0
        job.current_action = "Compositing clips..."
        logger.info(f"[{job_id}] Rendering {len(clips)} clip(s)...")

        from core.editor import render_clip

        os.makedirs(output_dir, exist_ok=True)
        total_clips = len(clips)

        for i, clip_window in enumerate(clips, 1):
            base_percent = ((i - 1) / total_clips) * 100
            job.progress_percent = base_percent
            job.progress_text = f"Rendering clip {i}/{total_clips} (viral {clip_window.viral_score}/10)..."
            job.current_action = f"Rendering clip {i}/{total_clips}"
            job.log(f"Rendering Clip {i} [viral={clip_window.viral_score}, hook='{clip_window.hook_phrase[:20]}...']")
            save_jobs(jobs)

            filename = f"{job_id}_clip{i:02d}_v{clip_window.viral_score:.0f}.mp4"
            out_path = os.path.join(output_dir, filename)

            render_start_time = time.time()
            def render_hook(p):
                job.progress_percent = base_percent + (p / total_clips)
                if p > 0.1:
                    elapsed = time.time() - render_start_time
                    estimated_clip_total = elapsed / (p / 100.0)
                    clip_eta = max(0, estimated_clip_total - elapsed)
                    clips_remaining = total_clips - i
                    job.estimated_time_remaining = clip_eta + (estimated_clip_total * clips_remaining)
                
            render_clip(
                video_path=video_path,
                clip_window=clip_window,
                words=result.words,
                output_path=out_path,
                resolution=resolution,
                progress_hook=render_hook
            )

            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            job.video_urls.append(f"/static/output/{filename}")
            job.clip_details.append({
                "filename": filename,
                "url": f"/static/output/{filename}",
                "viral_score": clip_window.viral_score,
                "hook": clip_window.hook_phrase,
                "start": clip_window.start,
                "end": clip_window.end,
                "size_mb": round(size_mb, 1),
                "is_highlight_reel": clip_window.is_merged,
                "segment_count": len(clip_window.segments) if clip_window.segments else 1,
            })

        # ── Done ──────────────────────────────────────────────────
        job.status = "complete"
        job.progress_percent = 100.0
        job.current_step_name = "Done"
        job.current_action = "Finished"
        job.estimated_time_remaining = 0
        job.progress_text = f"Done! Generated {len(clips)} clip(s)."
        job.log("Pipeline executed successfully.")
        logger.info(f"[{job_id}] Pipeline complete.")
        save_jobs(jobs)

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.progress_text = f"Error: {e}"
        job.log(f"CRITICAL ERROR: {e}")
        logger.error(f"[{job_id}] Pipeline failed: {e}", exc_info=True)
        save_jobs(jobs)

    finally:
        # Clean up temp files - explicitly ensure original video is NOT saved locally
        try:
            v_path = os.path.join(temp_dir, "video.mp4")
            a_path = os.path.join(temp_dir, "audio.wav")
            for p in [v_path, a_path]:
                if os.path.exists(p):
                    for _ in range(3): # Retry on Windows file lock
                        try:
                            os.remove(p)
                            break
                        except Exception:
                            time.sleep(1)
        except Exception as e:
            logger.warning(f"[{job_id}] Warning: Could not explicitly delete media files: {e}")

        if os.path.exists(temp_dir):
            try:
                for _ in range(3):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    if not os.path.exists(temp_dir):
                        break
                    time.sleep(1)
            except Exception:
                pass
