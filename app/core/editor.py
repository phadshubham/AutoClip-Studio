"""
AutoClip Studio — Editor  (v2 — Smart Framing with Blurred Background)

Renders 9:16 vertical shorts with:
  - Full video scaled to fit inside the frame (no content loss)
  - Cinematic heavily-blurred + darkened background filling remaining space
  - Dynamic word-level karaoke subtitles
"""

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import VideoFileClip, VideoClip, CompositeVideoClip, concatenate_videoclips
from proglog import ProgressBarLogger


class RenderLogger(ProgressBarLogger):
    def __init__(self, hook):
        super().__init__()
        self.hook = hook
    def bars_callback(self, bar, attr, value, old_value=None):
        if attr == 'index' and 'total' in self.bars[bar] and self.bars[bar]['total'] > 0:
            percent = (value / self.bars[bar]['total']) * 100
            if self.hook:
                self.hook(percent)


from core.transcriber import WordTimestamp
from core.clipper import ClipWindow

logger = logging.getLogger("autoclip.editor")

TARGET_ASPECT = 9 / 16  # width / height

RESOLUTION_PRESETS = {
    "1080": (1080, 1920, 64, 3, 3),
    "720":  (720,  1280, 42, 2, 2),
}

ACTIVE_COLOR = (255, 215, 0)
INACTIVE_COLOR = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0)
OUTLINE_COLOR = (0, 0, 0)
WORDS_PER_GROUP = 4
SUBTITLE_Y_PERCENT = 0.78  # slightly lower since video is centered now

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
    progress_hook=None,
) -> str:
    """Render a 9:16 vertical short with smart framing and dynamic subtitles.
    Supports merged highlight reels (multiple non-contiguous segments)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if resolution not in RESOLUTION_PRESETS:
        resolution = "1080"
    out_w, out_h, font_size, outline_w, shadow_off = RESOLUTION_PRESETS[resolution]

    font_path = _resolve_font(font_path)
    font = _load_font(font_path, font_size)

    logger.info(f"Loading video: {video_path}")
    video = VideoFileClip(video_path)
    bitrate = "8000k" if resolution == "1080" else "4000k"

    if clip_window.is_merged:
        # ── Merged highlight reel: render each segment, then concatenate ──
        logger.info(f"Rendering highlight reel with {len(clip_window.segments)} segments...")
        segment_clips = []

        for seg_idx, (seg_start, seg_end) in enumerate(clip_window.segments, 1):
            logger.info(f"  Segment {seg_idx}: {seg_start:.1f}s - {seg_end:.1f}s")

            # Extract and frame this segment
            seg = video.subclipped(seg_start, seg_end)
            framed = _fit_to_vertical(seg, out_w, out_h)

            # Get words for this segment, shift to segment-local time
            seg_words = [
                WordTimestamp(w.word, w.start - seg_start, w.end - seg_start)
                for w in words if w.start >= seg_start and w.end <= seg_end
            ]
            word_groups = _group_words(seg_words, WORDS_PER_GROUP)

            # Add subtitles
            sub_clip = _create_subtitle_clip(
                word_groups, framed.duration, font, out_w, out_h, outline_w, shadow_off,
            )
            composited = CompositeVideoClip([framed, sub_clip])
            segment_clips.append(composited)

        # Concatenate all segments
        final = concatenate_videoclips(segment_clips, method="compose")
    else:
        # ── Single clip: standard render ──
        clip_words = [
            WordTimestamp(w.word, w.start - clip_window.start, w.end - clip_window.start)
            for w in words if w.start >= clip_window.start and w.end <= clip_window.end
        ]
        logger.info(f"Clip has {len(clip_words)} words for subtitles.")
        word_groups = _group_words(clip_words, WORDS_PER_GROUP)

        clip = video.subclipped(clip_window.start, clip_window.end)
        framed = _fit_to_vertical(clip, out_w, out_h)
        logger.info(f"Framed to {out_w}x{out_h} (scale-to-fit + blurred bg)")

        subtitle_clip = _create_subtitle_clip(
            word_groups, framed.duration, font, out_w, out_h, outline_w, shadow_off,
        )
        final = CompositeVideoClip([framed, subtitle_clip])

    prog_logger = RenderLogger(progress_hook) if progress_hook else None

    logger.info(f"Encoding: {output_path}")
    final.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=30,
        preset="medium",
        bitrate=bitrate,
        logger=prog_logger,
    )
    video.close()
    final.close()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Rendered: {output_path} ({size_mb:.1f} MB)")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════════════
# VIDEO FRAMING — Scale-to-Fit with Blurred Background
# ═══════════════════════════════════════════════════════════════════════

def _fit_to_vertical(clip, out_w, out_h):
    """
    Scale the full source video to fit inside the 9:16 canvas.
    Fill the remaining space with a heavily blurred + darkened version.

    Layout:
    ┌─────────────────┐
    │  blurred bg     │
    │ ┌─────────────┐ │
    │ │  full video  │ │
    │ │  (scaled)    │ │
    │ └─────────────┘ │
    │  blurred bg     │
    └─────────────────┘
    """
    src_w, src_h = clip.size
    src_aspect = src_w / src_h
    target_aspect = out_w / out_h  # 9/16 = 0.5625

    if abs(src_aspect - target_aspect) < 0.05:
        # Already close to 9:16, just resize directly
        return clip.resized((out_w, out_h))

    # Calculate the size of the main (foreground) video
    # Scale to fit width, then check height
    fg_w = out_w
    fg_h = int(out_w / src_aspect)

    if fg_h > out_h:
        # Video is taller than frame — fit by height instead
        fg_h = out_h
        fg_w = int(out_h * src_aspect)

    # Pre-compute offsets
    x_offset = (out_w - fg_w) // 2
    y_offset = (out_h - fg_h) // 2

    # Pre-compute blur parameters
    blur_radius = max(out_w // 12, 30)  # heavy blur

    def compose_frame(frame):
        """Compose each frame: blurred bg + sharp foreground."""
        # frame is a numpy array (H, W, 3)

        # Create background: resize original to fill canvas, then blur + darken
        bg_img = Image.fromarray(frame)
        bg_img = bg_img.resize((out_w, out_h), Image.Resampling.LANCZOS)
        bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Darken the background (multiply by 0.3)
        bg_arr = np.array(bg_img, dtype=np.float32)
        bg_arr *= 0.3
        bg_arr = np.clip(bg_arr, 0, 255).astype(np.uint8)

        # Create foreground: resize to fit
        fg_img = Image.fromarray(frame)
        fg_img = fg_img.resize((fg_w, fg_h), Image.Resampling.LANCZOS)
        fg_arr = np.array(fg_img)

        # Composite: place foreground on background
        canvas = bg_arr.copy()
        fh, fw = fg_arr.shape[:2]
        canvas[y_offset:y_offset + fh, x_offset:x_offset + fw] = fg_arr

        return canvas

    # Use image_transform for per-frame compositing (frame -> frame)
    framed = clip.image_transform(compose_frame)

    return framed


# ═══════════════════════════════════════════════════════════════════════
# SUBTITLES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class _WordGroup:
    words: list[WordTimestamp]
    start: float
    end: float

    @property
    def text(self):
        return " ".join(w.word for w in self.words)


def _group_words(words, n):
    groups = []
    for i in range(0, len(words), n):
        chunk = words[i:i + n]
        if chunk:
            groups.append(_WordGroup(chunk, chunk[0].start, chunk[-1].end))
    return groups


def _create_subtitle_clip(word_groups, duration, font, out_w, out_h, outline_w, shadow_off):
    """
    Create subtitle overlay as RGB VideoClip + separate alpha mask VideoClip.
    This avoids the moviepy RGBA/ImageClip bug that corrupts the output file.
    """

    # Memoize: avoid rendering the same Pillow image twice per frame
    # (once for RGB, once for alpha)
    cache = {"t": -1.0, "rgba": None}

    def _get_rgba(t):
        if t == cache["t"] and cache["rgba"] is not None:
            return cache["rgba"]

        img = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        active_group = None
        for grp in word_groups:
            if grp.start <= t <= grp.end + 0.1:
                active_group = grp
                break
        if active_group is None:
            cache["t"] = t
            cache["rgba"] = np.array(img)
            return cache["rgba"]

        # Find which word is currently spoken
        active_idx = -1
        for idx, w in enumerate(active_group.words):
            if w.start <= t <= w.end + 0.05:
                active_idx = idx
                break
        if active_idx == -1:
            for idx, w in enumerate(active_group.words):
                if t < w.start:
                    active_idx = max(0, idx - 1)
                    break
            else:
                active_idx = len(active_group.words) - 1

        y_pos = int(out_h * SUBTITLE_Y_PERCENT)
        w_texts = [w.word for w in active_group.words]
        sp = _text_width(draw, " ", font)
        widths = [_text_width(draw, wt, font) for wt in w_texts]
        total = sum(widths) + sp * (len(w_texts) - 1)
        x = (out_w - total) // 2

        for idx, wt in enumerate(w_texts):
            color = ACTIVE_COLOR if idx == active_idx else INACTIVE_COLOR
            _draw_text(draw, x, y_pos, wt, font, color, outline_w, shadow_off)
            x += widths[idx] + sp

        cache["t"] = t
        cache["rgba"] = np.array(img)
        return cache["rgba"]

    def make_rgb(t):
        return _get_rgba(t)[:, :, :3]

    def make_mask(t):
        return _get_rgba(t)[:, :, 3] / 255.0

    rgb_clip = VideoClip(make_rgb, duration=duration)
    mask_clip = VideoClip(make_mask, is_mask=True, duration=duration)
    return rgb_clip.with_mask(mask_clip)


def _draw_text(draw, x, y, text, font, color, outline_w, shadow_off):
    draw.text((x + shadow_off, y + shadow_off), text, font=font, fill=(*SHADOW_COLOR, 180))
    for dx in range(-outline_w, outline_w + 1):
        for dy in range(-outline_w, outline_w + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=(*OUTLINE_COLOR, 255))
    draw.text((x, y), text, font=font, fill=(*color, 255))


def _text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _resolve_font(font_path):
    if font_path and os.path.exists(font_path):
        return font_path
    default = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "Montserrat-ExtraBold.ttf")
    if os.path.exists(default):
        return default
    logger.info("Downloading Montserrat ExtraBold font...")
    os.makedirs(os.path.dirname(default), exist_ok=True)
    try:
        urllib.request.urlretrieve(FONT_URL, default)
        logger.info(f"Font saved to {default}")
        return default
    except Exception as e:
        logger.warning(f"Font download failed: {e}")
        return None


def _load_font(font_path, size):
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception as e:
            logger.warning(f"Failed to load font {font_path}: {e}")
    
    logger.warning("Using Pillow default font (size argument not supported for default font).")
    # load_default() does not take a size argument in any Pillow version.
    return ImageFont.load_default()
