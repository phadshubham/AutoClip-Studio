"""
AutoClip -- Main Orchestrator
CLI entry point that chains: Download -> Transcribe -> Clip -> Render.

Usage:
    python main.py --url "https://youtube.com/watch?v=..."
    python main.py --url "https://youtube.com/watch?v=..." --model small --resolution 720
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# ── Logging Setup ──────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(name)-22s | %(levelname)-5s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── CLI Argument Parser ───────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoclip",
        description=(
            "AutoClip -- YouTube-to-Shorts Pipeline\n"
            "Downloads a YouTube video, transcribes it locally, "
            "finds high-engagement hooks, rates them on a viral scale (0-10), "
            "and renders 9:16 vertical shorts with dynamic subtitles."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url", required=True,
        help="YouTube video URL to process.",
    )
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base).",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Clip duration in seconds (default: 60).",
    )
    parser.add_argument(
        "--min-viral", type=float, default=7.0,
        help="Minimum viral score (0-10) to keep a clip (default: 7.0).",
    )
    parser.add_argument(
        "--resolution", default="1080",
        choices=["1080", "720"],
        help="Output resolution: 1080 (1080x1920) or 720 (720x1280). Default: 1080.",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output directory (default: ./output).",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep intermediate files (video, audio) after rendering.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    return parser


# ── Pipeline ───────────────────────────────────────────────────────
def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("autoclip.main")

    # Banner
    print()
    print("=" * 52)
    print("          AutoClip Pipeline")
    print("   YouTube -> Shorts with Dynamic Subtitles")
    print("=" * 52)
    print()

    start_time = time.time()
    temp_dir = os.path.join("temp", f"autoclip_{int(time.time())}")
    output_dir = args.output_dir
    
    # Ensure FFmpeg is found if installed via winget
    ffmpeg_path = r"C:\Users\Space\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

    try:
        # ── Step 1: Download ───────────────────────────────────────
        logger.info("=" * 50)
        logger.info("STEP 1/4: Downloading video...")
        logger.info("=" * 50)

        from src.downloader import download_video, check_ffmpeg

        if not check_ffmpeg():
            logger.error(
                "FFmpeg is not installed or not on PATH.\n"
                "Install: https://ffmpeg.org/download.html"
            )
            return 1

        video_path, audio_path = download_video(
            url=args.url,
            output_dir=temp_dir,
        )
        logger.info(f"Video: {video_path}")
        logger.info(f"Audio: {audio_path}")

        # ── Step 2: Transcribe ─────────────────────────────────────
        logger.info("=" * 50)
        logger.info("STEP 2/4: Transcribing audio (English)...")
        logger.info("=" * 50)

        from src.transcriber import transcribe

        result = transcribe(
            audio_path=audio_path,
            model_size=args.model,
            language="en",  # English only
        )

        if not result.words:
            logger.error("Transcription produced no words. Cannot proceed.")
            return 1

        logger.info(f"Language: {result.language} ({result.language_probability:.0%})")
        logger.info(f"Words:    {result.word_count}")
        logger.info(f"Duration: {result.duration:.1f}s")

        # ── Step 3: Find & Rate Clips ──────────────────────────────
        logger.info("=" * 50)
        logger.info("STEP 3/4: Analyzing hooks & rating viral potential...")
        logger.info("=" * 50)

        from src.clipper import find_clips

        clips = find_clips(
            words=result.words,
            clip_duration=args.duration,
            viral_threshold=args.min_viral,
        )

        if not clips:
            logger.error("No suitable clips found.")
            return 1

        # Display viral scorecard
        print()
        print("-" * 52)
        print(f"  {'#':<4} {'Viral':>6} {'Time Window':<20} {'Hook Phrase'}")
        print("-" * 52)
        for i, c in enumerate(clips, 1):
            score_bar = "#" * int(c.viral_score)
            print(
                f"  {i:<4} {c.viral_score:>5.1f}/10"
                f"  {c.start:.0f}s-{c.end:.0f}s"
                f"{'':>8}{c.hook_phrase}"
            )
        print("-" * 52)
        print(f"  {len(clips)} clip(s) scored >= {args.min_viral}/10")
        print()

        # ── Step 4: Render ─────────────────────────────────────────
        logger.info("=" * 50)
        logger.info(f"STEP 4/4: Rendering {len(clips)} short(s) at {args.resolution}p...")
        logger.info("=" * 50)

        from src.editor import render_clip

        os.makedirs(output_dir, exist_ok=True)
        rendered_files: list[tuple[str, float]] = []  # (path, viral_score)

        for i, clip_window in enumerate(clips, 1):
            output_name = f"short_{i:03d}_viral{clip_window.viral_score:.0f}.mp4"
            output_path = os.path.join(output_dir, output_name)

            logger.info(
                f"Rendering clip {i}/{len(clips)}: "
                f"{clip_window.start:.1f}s -> {clip_window.end:.1f}s "
                f"(viral={clip_window.viral_score:.1f}/10, "
                f"hook='{clip_window.hook_phrase}')"
            )

            render_clip(
                video_path=video_path,
                clip_window=clip_window,
                words=result.words,
                output_path=output_path,
                resolution=args.resolution,
            )

            rendered_files.append((output_path, clip_window.viral_score))

        # ── Cleanup ────────────────────────────────────────────────
        if not args.keep_temp:
            logger.info("Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            logger.info(f"Temporary files kept at: {temp_dir}")

        # ── Summary ────────────────────────────────────────────────
        elapsed = time.time() - start_time
        print()
        print("=" * 52)
        print("           Pipeline Complete!")
        print("=" * 52)
        print()
        print(f"  Time:       {elapsed:.1f}s")
        print(f"  Resolution: {args.resolution}p")
        print(f"  Output dir: {os.path.abspath(output_dir)}")
        print()
        for fpath, vscore in rendered_files:
            size = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  [viral {vscore:.1f}/10]  {os.path.basename(fpath)}  ({size:.1f} MB)")
        print()

        return 0

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
