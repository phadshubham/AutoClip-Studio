"""
AutoClip Studio — Transcriber
Transcribes audio using faster-whisper with word-level timestamps.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger("autoclip.transcriber")


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
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
    model_size: str = "large-v3",
    language: str = "en",
    progress_hook=None,
) -> TranscriptionResult:
    """Transcribe audio with word-level timestamps."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Resolve device
    device, compute = "cpu", "int8"
    if torch.cuda.is_available():
        device, compute = "cuda", "float16"
        logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")

    logger.info(f"Loading model '{model_size}' on {device}/{compute}")

    model = WhisperModel(model_size, device=device, compute_type=compute)

    logger.info(f"Transcribing: {audio_path.name}")
    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    words: list[WordTimestamp] = []
    text_parts: list[str] = []
    for segment in segments:
        text_parts.append(segment.text.strip())
        if segment.words:
            for w in segment.words:
                cleaned = w.word.strip()
                if cleaned:
                    words.append(WordTimestamp(cleaned, round(w.start, 3), round(w.end, 3)))
        
        if progress_hook and info.duration > 0:
            percent = (segment.end / info.duration) * 100
            progress_hook(min(100.0, percent))

    result = TranscriptionResult(
        words=words,
        full_text=" ".join(text_parts),
        language=info.language,
        language_probability=round(info.language_probability, 3),
        duration=info.duration,
    )
    logger.info(f"Transcribed {result.word_count} words, {result.duration:.1f}s")
    return result
