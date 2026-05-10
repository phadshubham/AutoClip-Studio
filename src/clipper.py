"""
AutoClip — Clipping Logic
Scans transcripts for high-engagement hooks, rates each clip on a
viral score (0-10), and returns all clips scoring 7+.
"""

import logging
import math
from dataclasses import dataclass
from src.transcriber import WordTimestamp

logger = logging.getLogger("autoclip.clipper")


# ── Viral Score Thresholds ─────────────────────────────────────────
# The raw hook score is normalized to a 0-10 "viral" scale.
# These constants tune the mapping.
VIRAL_THRESHOLD = 7.0       # Minimum viral score to keep a clip
MAX_RAW_SCORE = 40.0        # Raw score that maps to a perfect 10
MIN_RAW_FOR_NONZERO = 6.0   # Below this raw score → viral 0


@dataclass
class ClipWindow:
    """A time window representing a candidate clip."""
    start: float
    end: float
    score: float          # Raw hook score (sum of weights)
    viral_score: float    # Normalized 0-10 viral rating
    hook_phrase: str
    hook_time: float

    @property
    def duration(self) -> float:
        return self.end - self.start


# Hook phrases: (phrase, weight)
HOOK_PHRASES: list[tuple[str, int]] = [
    # Secrets & Reveals (10)
    ("the secret is", 10), ("here's the truth", 10), ("nobody tells you", 10),
    ("the real reason", 10), ("what they don't tell you", 10), ("the truth is", 10),
    # Warnings (9)
    ("never do this", 9), ("stop doing", 9), ("biggest mistake", 9),
    ("don't ever", 9), ("you're doing it wrong", 9), ("avoid this", 9), ("red flag", 9),
    # Emotional (8)
    ("changed my life", 8), ("blew my mind", 8), ("can't believe", 8),
    ("this is insane", 8), ("mind blowing", 8), ("game changer", 8),
    ("i was shocked", 8), ("you won't believe", 8),
    # Lists & Steps (7)
    ("number one", 7), ("first thing", 7), ("step one", 7),
    ("most important", 7), ("the key is", 7), ("rule number one", 7),
    # Authority (7)
    ("studies show", 7), ("research proves", 7), ("experts say", 7),
    ("science says", 7), ("proven to", 7),
    # Questions (6)
    ("do you know", 6), ("have you ever", 6), ("what if i told you", 6),
    ("did you know", 6), ("here's the thing", 6), ("ever wondered", 6),
    ("think about it", 6),
    # Urgency (6)
    ("right now", 6), ("listen carefully", 6), ("pay attention", 6),
    ("this is important", 6), ("before it's too late", 6),
]


@dataclass
class _HookHit:
    phrase: str
    weight: int
    word_index: int
    timestamp: float


def find_clips(
    words: list[WordTimestamp],
    clip_duration: float = 60.0,
    viral_threshold: float = VIRAL_THRESHOLD,
) -> list[ClipWindow]:
    """
    Find all high-engagement clip windows that score 7+ on the viral scale.

    Every candidate clip is rated on a viral score from 0 to 10 based on
    the density and strength of hook phrases it contains. Only clips
    at or above `viral_threshold` (default 7.0) are returned.

    If no clips meet the threshold, falls back to the first 60 seconds
    with a viral_score of 0.

    Args:
        words: List of WordTimestamp objects from the transcriber.
        clip_duration: Target clip duration in seconds (default 60).
        viral_threshold: Minimum viral score to include (default 7.0).

    Returns:
        List of ClipWindow objects sorted by viral_score (descending).
    """
    if not words:
        logger.warning("Empty transcript -- no clips to find.")
        return []

    video_duration = words[-1].end
    logger.info(f"Scanning {len(words)} words ({video_duration:.1f}s) for hooks...")

    hook_hits = _find_hooks(words)

    if not hook_hits:
        logger.info(f"No hooks found. Falling back to first {clip_duration:.0f}s.")
        return [ClipWindow(
            start=0.0, end=min(clip_duration, video_duration),
            score=0.0, viral_score=0.0,
            hook_phrase="(fallback: video start)", hook_time=0.0,
        )]

    logger.info(f"Found {len(hook_hits)} hook occurrences.")

    # Score all candidate windows
    candidates = _score_windows(hook_hits, clip_duration, video_duration)

    # Select non-overlapping clips that meet the viral threshold
    selected = _select_viral_clips(candidates, viral_threshold)

    if not selected:
        logger.info(
            f"No clips scored >= {viral_threshold:.1f}. "
            f"Falling back to first {clip_duration:.0f}s."
        )
        return [ClipWindow(
            start=0.0, end=min(clip_duration, video_duration),
            score=0.0, viral_score=0.0,
            hook_phrase="(fallback: video start)", hook_time=0.0,
        )]

    for i, clip in enumerate(selected, 1):
        logger.info(
            f"Clip {i}: {clip.start:.1f}s-{clip.end:.1f}s | "
            f"viral={clip.viral_score:.1f}/10 | "
            f"hook='{clip.hook_phrase}'"
        )
    return selected


def _raw_to_viral(raw_score: float) -> float:
    """
    Convert a raw hook-weight sum to a 0-10 viral score.

    Uses a logarithmic curve so that:
    - raw < MIN_RAW_FOR_NONZERO  -> 0
    - raw = ~12                  -> ~7.0
    - raw = ~20                  -> ~8.5
    - raw = MAX_RAW_SCORE (40)   -> 10.0
    """
    if raw_score < MIN_RAW_FOR_NONZERO:
        return 0.0

    # Normalize to 0-1 range using log scale for diminishing returns
    normalized = (raw_score - MIN_RAW_FOR_NONZERO) / (MAX_RAW_SCORE - MIN_RAW_FOR_NONZERO)
    normalized = min(normalized, 1.0)

    # Apply log curve: faster climb at low scores, plateaus near top
    # log1p(x * e) / log1p(e) maps [0,1] -> [0,1] with a log shape
    e = math.e
    viral = math.log1p(normalized * e) / math.log1p(e)

    # Scale to 5-10 range (anything that passes MIN_RAW is at least a 5)
    score = 5.0 + viral * 5.0
    return round(min(score, 10.0), 1)


def _find_hooks(words: list[WordTimestamp]) -> list[_HookHit]:
    """Scan word-level transcript for hook phrases."""
    hits: list[_HookHit] = []
    word_texts = [w.word.lower().strip(".,!?;:\"'()[]{}") for w in words]

    for phrase, weight in HOOK_PHRASES:
        phrase_words = phrase.lower().split()
        plen = len(phrase_words)
        for i in range(len(word_texts) - plen + 1):
            if word_texts[i:i + plen] == phrase_words:
                hits.append(_HookHit(phrase, weight, i, words[i].start))

    hits.sort(key=lambda h: h.timestamp)
    return hits


def _score_windows(
    hook_hits: list[_HookHit],
    clip_duration: float,
    video_duration: float,
) -> list[ClipWindow]:
    """Create and score candidate windows centered on each hook."""
    candidates: list[ClipWindow] = []
    seen: set[float] = set()

    for hit in hook_hits:
        half = clip_duration / 2.0
        start = max(0.0, hit.timestamp - half)
        end = start + clip_duration
        if end > video_duration:
            end = video_duration
            start = max(0.0, end - clip_duration)
        start, end = round(start, 2), round(end, 2)

        if start in seen:
            continue
        seen.add(start)

        raw_score = sum(h.weight for h in hook_hits if start <= h.timestamp <= end)
        viral = _raw_to_viral(raw_score)
        best = max((h for h in hook_hits if start <= h.timestamp <= end), key=lambda h: h.weight)

        candidates.append(ClipWindow(
            start=start, end=end,
            score=raw_score, viral_score=viral,
            hook_phrase=best.phrase, hook_time=best.timestamp,
        ))

    candidates.sort(key=lambda c: (-c.viral_score, c.start))
    return candidates


def _select_viral_clips(
    candidates: list[ClipWindow],
    viral_threshold: float,
) -> list[ClipWindow]:
    """
    Select all non-overlapping clips that meet the viral threshold.
    No max_clips cap -- every qualifying clip is returned.
    """
    selected: list[ClipWindow] = []
    for c in candidates:
        if c.viral_score < viral_threshold:
            continue
        if not any(c.start < s.end and s.start < c.end for s in selected):
            selected.append(c)
    return selected
