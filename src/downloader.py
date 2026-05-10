"""
AutoClip — Downloader Module
Downloads YouTube videos using yt-dlp and extracts audio for transcription.
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
    """Raised when a video download fails."""
    pass


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available on the system PATH."""
    return shutil.which("ffmpeg") is not None


def download_video(
    url: str,
    output_dir: str,
    max_retries: int = 3,
    max_height: int = 1080,
) -> tuple[str, str]:
    """
    Download a YouTube video and extract its audio track.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save downloaded files.
        max_retries: Number of retry attempts on failure.
        max_height: Maximum video height (default 1080p).

    Returns:
        Tuple of (video_path, audio_path).

    Raises:
        DownloadError: If the download fails after all retries.
    """
    if not check_ffmpeg():
        raise DownloadError(
            "FFmpeg is not installed or not on PATH.\n"
            "Install it from https://ffmpeg.org/download.html\n"
            "On Windows: winget install FFmpeg\n"
            "On macOS:   brew install ffmpeg\n"
            "On Linux:   sudo apt install ffmpeg"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = str(output_dir / "video.mp4")
    audio_path = str(output_dir / "audio.wav")

    # ── Download Video ─────────────────────────────────────────────
    video_opts = _build_video_opts(video_path, max_height)
    _download_with_retries(url, video_opts, "video", max_retries)

    if not os.path.exists(video_path):
        raise DownloadError(
            f"Video file was not created at {video_path}. "
            "The download may have failed silently."
        )

    # ── Extract Audio ──────────────────────────────────────────────
    audio_opts = _build_audio_opts(audio_path)
    _download_with_retries(url, audio_opts, "audio", max_retries)

    if not os.path.exists(audio_path):
        raise DownloadError(
            f"Audio file was not created at {audio_path}. "
            "Audio extraction may have failed."
        )

    video_size = os.path.getsize(video_path) / (1024 * 1024)
    audio_size = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info(f"Downloaded video: {video_size:.1f} MB")
    logger.info(f"Extracted audio:  {audio_size:.1f} MB")

    return video_path, audio_path


def _build_video_opts(output_path: str, max_height: int) -> list[str]:
    """Build yt-dlp command-line arguments for video download."""
    return [
        sys.executable, "-m", "yt_dlp",
        "--format", f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
        "--merge-output-format", "mp4",
        "--output", output_path,
        "--no-playlist",
        "--no-overwrites",
        # Retry & resilience
        "--retries", "5",
        "--fragment-retries", "5",
        # Bypass age-restriction where possible
        "--age-limit", "99",
        # Avoid throttling
        "--concurrent-fragments", "4",
    ]


def _build_audio_opts(output_path: str) -> list[str]:
    """Build yt-dlp command-line arguments for audio extraction."""
    return [
        sys.executable, "-m", "yt_dlp",
        "--format", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",  # 16kHz mono for Whisper
        "--output", output_path,
        "--no-playlist",
        "--no-overwrites",
        "--retries", "5",
        "--fragment-retries", "5",
        "--age-limit", "99",
    ]


def _download_with_retries(
    url: str,
    base_cmd: list[str],
    label: str,
    max_retries: int,
) -> None:
    """
    Execute a yt-dlp download command with exponential backoff retries.

    Args:
        url: YouTube video URL.
        base_cmd: Base command-line arguments.
        label: Human-readable label for logging (e.g., "video", "audio").
        max_retries: Maximum number of retry attempts.

    Raises:
        DownloadError: If all retry attempts fail.
    """
    cmd = base_cmd + [url]
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading {label} (attempt {attempt}/{max_retries})...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10-minute timeout
            )

            if result.returncode == 0:
                logger.info(f"Successfully downloaded {label}.")
                return

            stderr = result.stderr.strip()
            logger.warning(f"Attempt {attempt} failed: {stderr[:500]}")
            last_error = stderr

            # Check for unrecoverable errors
            if _is_unrecoverable(stderr):
                raise DownloadError(_format_unrecoverable_error(stderr, url))

        except subprocess.TimeoutExpired:
            last_error = "Download timed out after 10 minutes."
            logger.warning(f"Attempt {attempt}: {last_error}")
        except DownloadError:
            raise  # Re-raise unrecoverable errors immediately

        # Exponential backoff: 2s, 4s, 8s...
        if attempt < max_retries:
            wait_time = 2 ** attempt
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise DownloadError(
        f"Failed to download {label} after {max_retries} attempts.\n"
        f"Last error: {last_error}"
    )


def _is_unrecoverable(stderr: str) -> bool:
    """Check if the error indicates an unrecoverable failure."""
    unrecoverable_patterns = [
        "Video unavailable",
        "Private video",
        "This video has been removed",
        "copyright",
        "Sign in to confirm your age",
        "Join this channel to get access",
        "This video is not available",
        "HTTP Error 403",
        "HTTP Error 404",
    ]
    stderr_lower = stderr.lower()
    return any(pattern.lower() in stderr_lower for pattern in unrecoverable_patterns)


def _format_unrecoverable_error(stderr: str, url: str) -> str:
    """Format a human-readable error message for unrecoverable failures."""
    stderr_lower = stderr.lower()

    if "sign in to confirm your age" in stderr_lower:
        return (
            f"Age-restricted video: {url}\n"
            "Workaround: Export browser cookies and pass them via yt-dlp's --cookies flag.\n"
            "See: https://github.com/yt-dlp/yt-dlp#cookies"
        )
    if "private video" in stderr_lower:
        return f"This video is private and cannot be downloaded: {url}"
    if "video unavailable" in stderr_lower or "not available" in stderr_lower:
        return (
            f"Video is unavailable: {url}\n"
            "It may be geo-blocked, deleted, or require authentication."
        )
    if "copyright" in stderr_lower:
        return f"Video blocked due to copyright claims: {url}"
    if "join this channel" in stderr_lower:
        return f"This is a members-only video: {url}"

    return f"Unrecoverable download error for {url}:\n{stderr[:1000]}"
