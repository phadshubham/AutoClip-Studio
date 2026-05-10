"""
AutoClip Studio — Downloader
Downloads YouTube videos using yt-dlp and extracts audio.
"""

import os
import sys
import time
import shutil
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("autoclip.downloader")


class DownloadError(Exception):
    pass


def get_video_info(url: str) -> dict:
    """Fetch video metadata using yt-dlp without downloading."""
    import yt_dlp
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", "Unknown Title"),
                "duration": info.get("duration", 0),
                "thumbnail": info.get("thumbnail", ""),
                "channel": info.get("uploader", "Unknown Channel")
            }
        except Exception as e:
            logger.error(f"Failed to fetch video info: {e}")
            raise DownloadError(f"Could not fetch video details: {e}")

def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def download_video(url: str, output_dir: str, max_retries: int = 3, progress_hook=None) -> tuple[str, str]:
    """Download video + extract audio. Returns (video_path, audio_path)."""
    import yt_dlp
    import re
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = str(output_dir / "video.mp4")
    audio_path = str(output_dir / "audio.wav")

    def _hook(d):
        if progress_hook and d['status'] == 'downloading':
            try:
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                p_str = ansi_escape.sub('', d.get('_percent_str', '')).replace('%', '').strip()
                percent = float(p_str) if p_str else 0.0
                speed = ansi_escape.sub('', d.get('_speed_str', '')).strip()
                eta = d.get('eta', 0)
                progress_hook(percent, speed, eta)
            except Exception:
                pass

    ydl_opts_video = {
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'merge_output_format': 'mp4',
        'outtmpl': str(output_dir / "video.%(ext)s"),
        'noplaylist': True,
        'retries': 5,
        'fragment_retries': 5,
        'concurrent_fragment_downloads': 4,
        'progress_hooks': [_hook],
        'quiet': True,
        'no_warnings': True,
    }
    
    logger.info("Downloading video...")
    for attempt in range(1, max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
                ydl.download([url])
            break
        except Exception as e:
            if attempt == max_retries: raise DownloadError(f"Video download failed: {e}")
            time.sleep(2 ** attempt)

    # yt-dlp might create .mp4 if merge succeeds, rename if necessary
    for f in output_dir.glob("video.*"):
        if f.suffix != ".mp4":
            shutil.move(str(f), video_path)

    if not os.path.exists(video_path):
        raise DownloadError(f"Video file was not created at {video_path}.")

    ydl_opts_audio = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'postprocessor_args': {'ffmpeg': ['-ar', '16000', '-ac', '1']},
        'outtmpl': str(output_dir / "audio.%(ext)s"),
        'noplaylist': True,
        'retries': 5,
        'fragment_retries': 5,
        'progress_hooks': [_hook],
        'quiet': True,
        'no_warnings': True,
    }
    
    logger.info("Extracting audio...")
    for attempt in range(1, max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
                ydl.download([url])
            break
        except Exception as e:
            if attempt == max_retries: raise DownloadError(f"Audio extraction failed: {e}")
            time.sleep(2 ** attempt)

    if not os.path.exists(audio_path):
        raise DownloadError(f"Audio file was not created at {audio_path}.")

    v_mb = os.path.getsize(video_path) / (1024 * 1024)
    a_mb = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info(f"Downloaded video: {v_mb:.1f} MB, audio: {a_mb:.1f} MB")
    return video_path, audio_path
