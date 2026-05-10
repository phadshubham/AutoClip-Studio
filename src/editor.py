"""
AutoClip — Editor / Renderer Module
Crops video to 9:16, burns dynamic word-level subtitles, renders final short.
Supports 1080p and 720p output resolutions.
"""

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, VideoClip, CompositeVideoClip

from src.transcriber import WordTimestamp
from src.clipper import ClipWindow

logger = logging.getLogger("autoclip.editor")

# ── Constants ──────────────────────────────────────────────────────
TARGET_ASPECT = 9 / 16

# Resolution presets: (width, height, font_size, outline_width, shadow_offset)
RESOLUTION_PRESETS = {
    "1080": (1080, 1920, 64, 3, 3),
    "720":  (720,  1280, 42, 2, 2),
}

# Subtitle styling
ACTIVE_COLOR = (255, 215, 0)      # Gold for active word
INACTIVE_COLOR = (255, 255, 255)   # White for other words
SHADOW_COLOR = (0, 0, 0)
OUTLINE_COLOR = (0, 0, 0)
WORDS_PER_GROUP = 4
SUBTITLE_Y_PERCENT = 0.72

# Font URL (Montserrat ExtraBold — OFL license)
FONT_URL = (
    "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/"
    "Montserrat-ExtraBold.ttf"
)


def render_clip(
    video_path: str,
    clip_window: ClipWindow,
    words: list[WordTimestamp],
    output_path: str,
    font_path: str | None = None,
    resolution: str = "1080",
) -> str:
    """
    Render a 9:16 vertical short with dynamic word-level subtitles.

    Args:
        video_path: Path to the source video file.
        clip_window: The clip time window to extract.
        words: Word-level timestamps for the full video.
        output_path: Path to save the rendered short.
        font_path: Path to TTF font file. Auto-downloads if None.
        resolution: Output resolution ('1080' or '720').

    Returns:
        Path to the rendered output file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resolve resolution ─────────────────────────────────────────
    if resolution not in RESOLUTION_PRESETS:
        logger.warning(f"Unknown resolution '{resolution}', using 1080.")
        resolution = "1080"
    out_w, out_h, font_size, outline_w, shadow_off = RESOLUTION_PRESETS[resolution]

    # ── Resolve font ───────────────────────────────────────────────
    font_path = _resolve_font(font_path)
    font = _load_font(font_path, font_size)

    # ── Filter words for this clip window ──────────────────────────
    clip_words = [
        WordTimestamp(
            word=w.word,
            start=w.start - clip_window.start,
            end=w.end - clip_window.start,
        )
        for w in words
        if w.start >= clip_window.start and w.end <= clip_window.end
    ]
    logger.info(f"Clip contains {len(clip_words)} words for subtitles.")

    # ── Group words into display chunks ────────────────────────────
    word_groups = _group_words(clip_words, WORDS_PER_GROUP)

    # ── Load and crop video ────────────────────────────────────────
    logger.info(f"Loading video: {video_path}")
    video = VideoFileClip(video_path)
    clip = video.subclipped(clip_window.start, clip_window.end)

    cropped = _crop_to_vertical(clip, out_w, out_h)
    logger.info(f"Cropped to {out_w}x{out_h}")

    # ── Create subtitle overlay ────────────────────────────────────
    logger.info("Rendering subtitle overlays...")
    subtitle_clip = _create_subtitle_clip(
        word_groups, cropped.duration, font,
        out_w, out_h, outline_w, shadow_off,
    )

    # ── Composite ──────────────────────────────────────────────────
    final = CompositeVideoClip([cropped, subtitle_clip])

    # ── Write output ───────────────────────────────────────────────
    bitrate = "8000k" if resolution == "1080" else "4000k"
    logger.info(f"Encoding final video: {output_path}")
    final.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=30,
        preset="medium",
        bitrate=bitrate,
        logger=None,
    )

    video.close()
    final.close()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Rendered: {output_path} ({size_mb:.1f} MB)")
    return str(output_path)


# ── Internal helpers ───────────────────────────────────────────────


def _crop_to_vertical(clip: VideoFileClip, out_w: int, out_h: int):
    """Center-crop landscape to 9:16, then resize to target resolution."""
    w, h = clip.size
    target_w = int(h * TARGET_ASPECT)
    if target_w > w:
        target_w = w
    x_offset = (w - target_w) // 2
    cropped = clip.cropped(x1=x_offset, y1=0, x2=x_offset + target_w, y2=h)
    return cropped.resized((out_w, out_h))


@dataclass
class _WordGroup:
    """A group of words to display simultaneously."""
    words: list[WordTimestamp]
    start: float
    end: float

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


def _group_words(words: list[WordTimestamp], group_size: int) -> list[_WordGroup]:
    """Split words into display groups of N words each."""
    groups: list[_WordGroup] = []
    for i in range(0, len(words), group_size):
        chunk = words[i:i + group_size]
        if chunk:
            groups.append(_WordGroup(chunk, chunk[0].start, chunk[-1].end))
    return groups


def _create_subtitle_clip(
    word_groups: list[_WordGroup],
    duration: float,
    font: ImageFont.FreeTypeFont,
    out_w: int, out_h: int,
    outline_w: int, shadow_off: int,
) -> VideoClip:
    """Create transparent subtitle overlay with word-level highlighting."""

    last_t = -1.0
    last_rgba = None

    def get_rgba(t):
        nonlocal last_t, last_rgba
        if t == last_t and last_rgba is not None:
            return last_rgba

        img = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        active_group = None
        for group in word_groups:
            if group.start <= t <= group.end + 0.1:
                active_group = group
                break
        
        if active_group is None:
            last_t = t
            last_rgba = np.array(img)
            return last_rgba

        # Find active word
        active_word_idx = -1
        for idx, word in enumerate(active_group.words):
            if word.start <= t <= word.end + 0.05:
                active_word_idx = idx
                break
        if active_word_idx == -1:
            for idx, word in enumerate(active_group.words):
                if t < word.start:
                    active_word_idx = max(0, idx - 1)
                    break
            else:
                active_word_idx = len(active_group.words) - 1

        y_pos = int(out_h * SUBTITLE_Y_PERCENT)
        words_text = [w.word for w in active_group.words]
        space_w = _text_width(draw, " ", font)
        word_widths = [_text_width(draw, w, font) for w in words_text]
        total_width = sum(word_widths) + space_w * (len(words_text) - 1)
        x_start = (out_w - total_width) // 2

        x = x_start
        for idx, word_text in enumerate(words_text):
            color = ACTIVE_COLOR if idx == active_word_idx else INACTIVE_COLOR
            _draw_text_with_effects(draw, x, y_pos, word_text, font, color, outline_w, shadow_off)
            x += word_widths[idx] + space_w

        last_t = t
        last_rgba = np.array(img)
        return last_rgba

    def make_rgb(t):
        return get_rgba(t)[:, :, :3]

    def make_alpha(t):
        return get_rgba(t)[:, :, 3] / 255.0

    # Create the RGB clip and attach the Alpha mask
    rgb_clip = VideoClip(make_rgb, duration=duration)
    mask_clip = VideoClip(make_alpha, is_mask=True, duration=duration)
    return rgb_clip.with_mask(mask_clip)


def _draw_text_with_effects(draw, x, y, text, font, color, outline_w, shadow_off):
    """Draw text with drop shadow and outline for readability."""
    draw.text((x + shadow_off, y + shadow_off), text, font=font, fill=(*SHADOW_COLOR, 180))
    for dx in range(-outline_w, outline_w + 1):
        for dy in range(-outline_w, outline_w + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=(*OUTLINE_COLOR, 255))
    draw.text((x, y), text, font=font, fill=(*color, 255))


def _text_width(draw, text, font):
    """Get the pixel width of rendered text."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _resolve_font(font_path: str | None) -> str:
    """Resolve font path, downloading Montserrat if needed."""
    if font_path and os.path.exists(font_path):
        return font_path
    default_path = os.path.join("assets", "fonts", "Montserrat-ExtraBold.ttf")
    if os.path.exists(default_path):
        return default_path
    logger.info("Downloading Montserrat ExtraBold font...")
    os.makedirs(os.path.dirname(default_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(FONT_URL, default_path)
        logger.info(f"Font saved to {default_path}")
        return default_path
    except Exception as e:
        logger.warning(f"Font download failed: {e}. Using system default.")
        return None


def _load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont:
    """Load a TTF font, falling back to Pillow default."""
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception as e:
            logger.warning(f"Failed to load font {font_path}: {e}")
    logger.warning("Using Pillow default font (limited styling).")
    return ImageFont.load_default(size)
