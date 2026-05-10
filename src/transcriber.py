"""
AutoClip — Transcription Module
Transcribes audio using faster-whisper with word-level timestamps.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("autoclip.transcriber")


@dataclass
class WordTimestamp:
    """A single word with its timing information."""
    word: str
    start: float  # Start time in seconds
    end: float    # End time in seconds

    def __repr__(self) -> str:
        return f"WordTimestamp('{self.word}', {self.start:.2f}→{self.end:.2f})"


@dataclass
class TranscriptionResult:
    """Full transcription output with metadata."""
    words: list[WordTimestamp] = field(default_factory=list)
    full_text: str = ""
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0

    @property
    def word_count(self) -> int:
        return len(self.words)


def transcribe(
    audio_path: str,
    model_size: str = "base",
    language: str | None = None,
    device: str = "auto",
    compute_type: str = "auto",
) -> TranscriptionResult:
    """
    Transcribe an audio file using faster-whisper with word-level timestamps.

    Args:
        audio_path: Path to the .wav audio file.
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
        language: Language code (e.g., "en"). None for auto-detection.
        device: Device to use ("auto", "cpu", "cuda").
        compute_type: Compute type ("auto", "int8", "float16", "float32").

    Returns:
        TranscriptionResult with word-level timestamps.

    Raises:
        FileNotFoundError: If the audio file doesn't exist.
        RuntimeError: If transcription fails.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # ── Resolve device and compute type ────────────────────────────
    actual_device, actual_compute = _resolve_device(device, compute_type)
    logger.info(
        f"Loading faster-whisper model '{model_size}' "
        f"on {actual_device} with {actual_compute} precision"
    )

    # ── Load model ─────────────────────────────────────────────────
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper is not installed.\n"
            "Install it with: pip install faster-whisper"
        )

    model = WhisperModel(
        model_size,
        device=actual_device,
        compute_type=actual_compute,
    )

    # ── Transcribe ─────────────────────────────────────────────────
    logger.info(f"Transcribing: {audio_path.name}")
    try:
        segments, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language=language,
            beam_size=5,
            vad_filter=True,           # Skip silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e

    # ── Extract word-level timestamps ──────────────────────────────
    words: list[WordTimestamp] = []
    full_text_parts: list[str] = []

    for segment in segments:
        full_text_parts.append(segment.text.strip())

        if segment.words:
            for word_info in segment.words:
                cleaned_word = word_info.word.strip()
                if cleaned_word:
                    words.append(WordTimestamp(
                        word=cleaned_word,
                        start=round(word_info.start, 3),
                        end=round(word_info.end, 3),
                    ))

    full_text = " ".join(full_text_parts)

    result = TranscriptionResult(
        words=words,
        full_text=full_text,
        language=info.language,
        language_probability=round(info.language_probability, 3),
        duration=info.duration,
    )

    logger.info(
        f"Transcription complete: {result.word_count} words, "
        f"language={result.language} ({result.language_probability:.0%}), "
        f"duration={result.duration:.1f}s"
    )

    if result.word_count == 0:
        logger.warning(
            "No words detected in audio. The file may be silent, "
            "too noisy, or in an unsupported format."
        )

    return result


def _resolve_device(
    device: str,
    compute_type: str,
) -> tuple[str, str]:
    """
    Resolve the device and compute type based on system capabilities.

    Returns:
        Tuple of (device, compute_type).
    """
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                actual_device = "cuda"
                actual_compute = "float16" if compute_type == "auto" else compute_type
                logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                actual_device = "cpu"
                actual_compute = "int8" if compute_type == "auto" else compute_type
        except ImportError:
            actual_device = "cpu"
            actual_compute = "int8" if compute_type == "auto" else compute_type
            logger.info("PyTorch not found — using CPU with int8 quantization.")
    else:
        actual_device = device
        actual_compute = compute_type if compute_type != "auto" else (
            "float16" if device == "cuda" else "int8"
        )

    return actual_device, actual_compute
