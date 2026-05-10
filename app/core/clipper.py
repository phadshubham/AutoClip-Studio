"""
AutoClip Studio — Clipper v3
Audio-driven + transcript intelligence for entertainment/comedy content.
Supports merging non-contiguous moments into highlight reels.
"""

import logging
import math
import wave
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from core.transcriber import WordTimestamp


logger = logging.getLogger("autoclip.clipper")

VIRAL_THRESHOLD = 5.0

SCORE_WEIGHTS = {
    "audio":        0.35,
    "entertainment": 0.25,
    "dynamics":     0.15,
    "hooks":        0.15,
    "coherence":    0.10,
}


@dataclass
class ClipWindow:
    start: float
    end: float
    score: float
    viral_score: float
    hook_phrase: str
    hook_time: float
    segments: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.segments:
            return sum(e - s for s, e in self.segments)
        return self.end - self.start

    @property
    def is_merged(self) -> bool:
        return len(self.segments) > 1


@dataclass
class _AudioProfile:
    energies: np.ndarray
    time_step: float
    peaks: list
    mean_energy: float
    std_energy: float


@dataclass
class _Sentence:
    words: list
    start: float
    end: float
    text: str


# ── Phrase / word banks ──────────────────────────────────────────────

HOOK_PHRASES: list[tuple[str, int]] = [
    ("the secret is", 10), ("here's the truth", 10), ("nobody tells you", 10),
    ("the real reason", 10), ("the truth is", 10), ("never do this", 9),
    ("stop doing", 9), ("biggest mistake", 9), ("you're doing it wrong", 9),
    ("changed my life", 8), ("blew my mind", 8), ("can't believe", 8),
    ("game changer", 8), ("you won't believe", 8), ("mind blowing", 8),
    ("most important", 7), ("the key is", 7), ("studies show", 7),
    ("here's the thing", 6), ("did you know", 6), ("think about it", 6),
    ("here's why", 8), ("the problem is", 7), ("the answer is", 7),
    ("what most people", 8), ("people don't realize", 8), ("the one thing", 8),
    ("let me tell you", 7), ("here's what happened", 7), ("check this out", 7),
]

ENTERTAINMENT_PHRASES: list[tuple[str, int]] = [
    ("oh my god", 10), ("oh my days", 10), ("no way", 9), ("are you serious", 9),
    ("what the hell", 9), ("what the", 8), ("i can't believe", 9),
    ("you're joking", 8), ("no chance", 7), ("allow it", 8), ("allow it fam", 9),
    ("i'm dead", 8), ("i'm done", 7), ("get out", 7), ("shut up", 7),
    ("come on", 6), ("let's go", 7), ("that's peak", 8), ("that's mad", 7),
    ("that's sick", 7), ("you lot", 6), ("oh no", 7), ("wait what", 8),
    ("are you kidding", 8), ("how is that", 7), ("that's jokes", 8),
    ("moving mad", 8), ("violation", 8), ("that's a violation", 9),
    ("you're moving mad", 9), ("nah nah nah", 8), ("rah", 7),
    ("he's finished", 8), ("it's over", 7), ("we're done", 7),
]

REACTION_WORDS: set[str] = {
    "oh", "wow", "what", "no", "yes", "whoa", "damn", "dude",
    "bruv", "mate", "fam", "innit", "mad", "sick", "peak",
    "man", "oi", "nah", "bro", "lad", "lads", "rah", "yo",
    "insane", "crazy", "ridiculous", "mental", "wild", "jokes",
    "hilarious", "funny", "dead", "violated", "finished",
    "literally", "actually", "honestly", "seriously",
    "obviously", "absolutely", "definitely", "completely",
}

POWER_WORDS: set[str] = {
    "absolutely", "incredible", "amazing", "terrible", "insane",
    "literally", "exactly", "honestly", "seriously", "perfect",
    "worst", "best", "never", "always", "impossible", "unbelievable",
    "crazy", "massive", "huge", "critical", "stunning", "shocking",
    "fantastic", "horrible", "outstanding", "powerful", "epic",
    "legendary", "genius", "stupid", "obsessed", "hilarious",
}


# ═════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════

def find_clips(
    words: list[WordTimestamp],
    audio_path: str | None = None,
    max_duration: float = 60.0,
    min_duration: float = 15.0,
    viral_threshold: float = VIRAL_THRESHOLD,
    max_clips: int = 10,
) -> list[ClipWindow]:
    """Find engaging clips using audio + transcript analysis."""
    if not words:
        return []

    video_duration = words[-1].end
    logger.info(f"Scanning {len(words)} words ({video_duration:.1f}s)...")

    # Audio analysis
    audio_profile = None
    if audio_path:
        audio_profile = _analyze_audio(audio_path)
        if audio_profile:
            logger.info(f"Audio: {len(audio_profile.peaks)} energy peaks detected.")

    # Sentence detection
    sentences = _detect_sentences(words)
    logger.info(f"Detected {len(sentences)} sentences.")

    if not sentences:
        return [_fallback_clip(video_duration, max_duration)]

    # Generate candidates
    candidates = _generate_candidates(sentences, words, min_duration, max_duration, audio_profile)
    logger.info(f"Generated {len(candidates)} candidate windows.")

    if not candidates:
        return [_fallback_clip(video_duration, max_duration)]

    # Score each candidate
    all_texts = [w.word.lower().strip(".,!?;:\"'()[]{}") for w in words]
    scored = []
    for cand in candidates:
        scores = _score_candidate(cand, words, all_texts, audio_profile)
        composite = sum(SCORE_WEIGHTS[k] * scores[k] for k in SCORE_WEIGHTS)
        viral = _composite_to_viral(composite)
        hook_phrase, hook_time = _best_hook_in_window(cand, words)

        scored.append(ClipWindow(
            start=cand["start"], end=cand["end"],
            score=round(composite, 4), viral_score=viral,
            hook_phrase=hook_phrase, hook_time=hook_time,
        ))

    scored.sort(key=lambda c: (-c.viral_score, c.start))
    selected = _select_non_overlapping(scored)
    selected = [c for c in selected if c.viral_score >= viral_threshold][:max_clips]

    if not selected:
        scored.sort(key=lambda c: -c.viral_score)
        selected = _select_non_overlapping(scored)[:max_clips]

    if not selected:
        return [_fallback_clip(video_duration, max_duration)]



    for i, c in enumerate(selected, 1):
        tag = " [HIGHLIGHT REEL]" if c.is_merged else ""
        logger.info(f"Clip {i}: {c.start:.1f}s-{c.end:.1f}s ({c.duration:.0f}s) viral={c.viral_score}{tag}")

    return selected


# ═════════════════════════════════════════════════════════════════════
# AUDIO ANALYSIS
# ═════════════════════════════════════════════════════════════════════

def _analyze_audio(audio_path: str) -> _AudioProfile | None:
    """Analyze WAV file for energy spikes."""
    try:
        audio, sr = _read_wav(audio_path)
        energies, time_step = _compute_energy(audio, sr)
        mean_e = float(np.mean(energies))
        std_e = float(np.std(energies))
        peaks = _find_energy_peaks(energies, time_step, mean_e, std_e)
        return _AudioProfile(energies, time_step, peaks, mean_e, std_e)
    except Exception as e:
        logger.warning(f"Audio analysis failed: {e}")
        return None


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as f:
        sr = f.getframerate()
        n = f.getnframes()
        sw = f.getsampwidth()
        raw = f.readframes(n)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio /= float(np.iinfo(dtype).max)
    nc = len(raw) // (n * sw) if n > 0 else 1
    if nc > 1:
        audio = audio.reshape(-1, nc).mean(axis=1)
    return audio, sr


def _compute_energy(audio, sr, window_sec=0.5, hop_sec=0.25):
    win = int(sr * window_sec)
    hop = int(sr * hop_sec)
    energies = []
    for i in range(0, len(audio) - win, hop):
        rms = float(np.sqrt(np.mean(audio[i:i + win] ** 2)))
        energies.append(rms)
    return np.array(energies), hop_sec


def _find_energy_peaks(energies, time_step, mean_e, std_e, threshold_k=1.2):
    """Find sustained energy peaks above mean + k*std."""
    threshold = mean_e + threshold_k * std_e
    peaks = []
    in_peak = False
    start_i = 0

    for i, e in enumerate(energies):
        if e > threshold and not in_peak:
            in_peak = True
            start_i = i
        elif e <= threshold and in_peak:
            in_peak = False
            peak_e = float(np.max(energies[start_i:i]))
            s = start_i * time_step
            end = i * time_step
            if end - s >= 1.0:  # at least 1s peak
                peaks.append((s, end, peak_e))

    if in_peak:
        peak_e = float(np.max(energies[start_i:]))
        s = start_i * time_step
        end = len(energies) * time_step
        if end - s >= 1.0:
            peaks.append((s, end, peak_e))

    # Merge peaks within 3s of each other
    merged = []
    for p in peaks:
        if merged and p[0] - merged[-1][1] < 3.0:
            merged[-1] = (merged[-1][0], p[1], max(merged[-1][2], p[2]))
        else:
            merged.append(list(p))

    return [tuple(m) for m in merged]


# ═════════════════════════════════════════════════════════════════════
# SENTENCE DETECTION
# ═════════════════════════════════════════════════════════════════════

def _detect_sentences(words: list[WordTimestamp]) -> list[_Sentence]:
    if not words:
        return []
    sentences, current = [], []
    for i, w in enumerate(words):
        current.append(w)
        is_end = False
        stripped = w.word.strip()
        if stripped and stripped[-1] in ".!?":
            is_end = True
        elif i < len(words) - 1 and words[i + 1].start - w.end > 0.7:
            is_end = True
        if is_end and len(current) >= 2:
            sentences.append(_Sentence(list(current), current[0].start, current[-1].end,
                                       " ".join(ww.word for ww in current)))
            current = []
    if current and len(current) >= 2:
        sentences.append(_Sentence(list(current), current[0].start, current[-1].end,
                                   " ".join(ww.word for ww in current)))
    return sentences


# ═════════════════════════════════════════════════════════════════════
# CANDIDATE GENERATION
# ═════════════════════════════════════════════════════════════════════

def _generate_candidates(sentences, words, min_dur, max_dur, audio_profile):
    candidates = []
    n = len(sentences)
    vid_end = words[-1].end + 0.5 if words else 0

    # Method 1: sentence-based sliding window
    for i in range(n):
        for j in range(i + 1, n + 1):
            group = sentences[i:j]
            raw_s, raw_e = group[0].start, group[-1].end
            dur = raw_e - raw_s
            if dur < min_dur:
                continue
            if dur > max_dur:
                break
            start = max(0.0, raw_s - 1.0)
            end = min(vid_end, raw_e + 0.5)
            clip_words = [w for w in words if w.start >= raw_s and w.end <= raw_e]
            candidates.append({"start": round(start, 2), "end": round(end, 2),
                               "clip_words": clip_words})

    # Method 2: audio peak-centered candidates
    if audio_profile:
        for peak_s, peak_e, _ in audio_profile.peaks:
            # Find sentences overlapping with peak ±5s
            context_s, context_e = peak_s - 5, peak_e + 5
            relevant = [s for s in sentences if s.end > context_s and s.start < context_e]
            if not relevant:
                continue
            for size in [3, 5, 8]:  # different group sizes
                center_idx = len(relevant) // 2
                lo = max(0, center_idx - size // 2)
                hi = min(len(relevant), lo + size)
                group = relevant[lo:hi]
                if not group:
                    continue
                raw_s, raw_e = group[0].start, group[-1].end
                dur = raw_e - raw_s
                if dur < min_dur or dur > max_dur:
                    continue
                start = max(0.0, raw_s - 1.0)
                end = min(vid_end, raw_e + 0.5)
                clip_words = [w for w in words if w.start >= raw_s and w.end <= raw_e]
                candidates.append({"start": round(start, 2), "end": round(end, 2),
                                   "clip_words": clip_words})

    # Deduplicate near-identical candidates
    if len(candidates) > 500:
        candidates = _prune_candidates(candidates)

    return candidates


def _prune_candidates(candidates, max_keep=400):
    by_bucket = {}
    for c in candidates:
        b = int(c["start"] // 5)
        by_bucket.setdefault(b, []).append(c)
    pruned = []
    for bucket in by_bucket.values():
        bucket.sort(key=lambda c: c["end"] - c["start"])
        if len(bucket) <= 3:
            pruned.extend(bucket)
        else:
            pruned.extend([bucket[0], bucket[len(bucket)//2], bucket[-1]])
    return pruned[:max_keep]


# ═════════════════════════════════════════════════════════════════════
# MULTI-SIGNAL SCORING
# ═════════════════════════════════════════════════════════════════════

def _score_candidate(cand, all_words, all_texts, audio_profile):
    cw = cand["clip_words"]
    dur = cand["end"] - cand["start"]
    if not cw or dur <= 0:
        return {k: 0.0 for k in SCORE_WEIGHTS}
    ct = [w.word.lower().strip(".,!?;:\"'()[]{}") for w in cw]
    return {
        "audio":        _score_audio(cand, audio_profile),
        "entertainment": _score_entertainment(ct, cw),
        "dynamics":     _score_dynamics(cw, dur),
        "hooks":        _score_hooks(ct),
        "coherence":    _score_coherence(ct, all_texts),
    }


def _score_audio(cand, profile):
    """Score based on audio energy overlap with this window."""
    if not profile:
        return 0.3  # neutral when no audio
    s, e = cand["start"], cand["end"]
    dur = e - s

    # Calculate average energy in this window
    si = max(0, int(s / profile.time_step))
    ei = min(len(profile.energies), int(e / profile.time_step))
    if si >= ei:
        return 0.0
    window_energy = float(np.mean(profile.energies[si:ei]))
    relative = (window_energy - profile.mean_energy) / (profile.std_energy + 1e-8)

    # Count how many peaks overlap
    peak_overlap = sum(1 for ps, pe, _ in profile.peaks if ps < e and pe > s)

    energy_score = min(max(relative / 2.0, 0.0), 1.0)
    peak_score = min(peak_overlap / 3.0, 1.0)
    return energy_score * 0.5 + peak_score * 0.5


def _score_entertainment(clip_texts, clip_words):
    """Score entertainment value — reactions, banter, Sidemen-style moments."""
    if not clip_texts:
        return 0.0

    # Entertainment phrases
    phrase_score = 0
    for phrase, weight in ENTERTAINMENT_PHRASES:
        pw = phrase.lower().split()
        for i in range(len(clip_texts) - len(pw) + 1):
            if clip_texts[i:i + len(pw)] == pw:
                phrase_score += weight
                break

    # Reaction words density
    reaction_count = sum(1 for w in clip_texts if w in REACTION_WORDS)
    reaction_density = reaction_count / len(clip_texts)

    # Exclamation density
    excl_count = sum(1 for w in clip_words if w.word.strip().endswith("!"))
    question_count = sum(1 for w in clip_words if w.word.strip().endswith("?"))

    # Power words
    power_count = sum(1 for w in clip_texts if w in POWER_WORDS)

    p = min(phrase_score / 15.0, 1.0)
    r = min(reaction_density * 5, 1.0)
    e = min((excl_count + question_count) / 4.0, 1.0)
    pw = min(power_count / 4.0, 1.0)

    return p * 0.4 + r * 0.25 + e * 0.2 + pw * 0.15


def _score_dynamics(clip_words, duration):
    """Score speech dynamics — pacing changes indicate excitement."""
    if len(clip_words) < 5 or duration < 3:
        return 0.0
    # Local speech rate in 3s windows
    rates = []
    t = clip_words[0].start
    end_t = clip_words[-1].end
    while t < end_t:
        count = sum(1 for w in clip_words if t <= w.start < t + 3)
        rates.append(count / 3.0)
        t += 1.5
    if not rates:
        return 0.0
    avg = sum(rates) / len(rates)
    var = sum((r - avg) ** 2 for r in rates) / len(rates)
    rate_s = min(avg / 4.0, 1.0)
    var_s = min(math.sqrt(var) / 1.5, 1.0)
    return rate_s * 0.5 + var_s * 0.5


def _score_hooks(clip_texts):
    total = 0
    found = set()
    for phrase, weight in HOOK_PHRASES:
        pw = phrase.lower().split()
        for i in range(len(clip_texts) - len(pw) + 1):
            if clip_texts[i:i + len(pw)] == pw and phrase not in found:
                total += weight
                found.add(phrase)
    return min(total / 20.0, 1.0)


def _score_coherence(clip_texts, all_texts):
    stop = {"the","a","an","is","are","was","were","be","been","have","has","had",
            "do","does","did","will","would","could","should","to","of","in","for",
            "on","with","at","by","from","as","into","through","but","and","or","if",
            "that","this","it","its","i","you","he","she","we","they","me","him",
            "her","us","them","my","your","his","our","their","so","just","like",
            "going","know","think","want","get","got","go","said","say","about","up"}
    cc = [w for w in clip_texts if w and len(w) > 2 and w not in stop]
    ac = [w for w in all_texts if w and len(w) > 2 and w not in stop]
    if not cc or not ac:
        return 0.5
    cf, af = Counter(cc), Counter(ac)
    ct, at = len(cc), len(ac)
    specs = []
    for word, count in cf.items():
        tf_c = count / ct
        tf_a = af.get(word, 0) / at
        if tf_a > 0:
            specs.append(min(tf_c / tf_a, 10.0))
    if not specs:
        return 0.5
    avg_spec = sum(specs) / len(specs)
    unique_ratio = len(cf) / ct
    rep_score = 1.0 - unique_ratio
    return min(max(min(avg_spec / 3.0, 1.0) * 0.6 + min(rep_score * 2, 1.0) * 0.4, 0.0), 1.0)


# ═════════════════════════════════════════════════════════════════════
# HIGHLIGHT REEL (merging non-contiguous moments)
# ═════════════════════════════════════════════════════════════════════

def _create_highlight_reel(scored_clips, words, max_total=60.0):
    """Merge top short moments into one highlight reel clip."""
    # Find short, high-scoring moments (5-20s each)
    moments = []
    for c in scored_clips:
        dur = c.end - c.start
        if 5 <= dur <= 25 and c.viral_score >= 4.0:
            moments.append(c)
    if len(moments) < 3:
        return None

    # Sort by viral score, take top moments that fit
    moments.sort(key=lambda c: -c.viral_score)
    selected = []
    total_dur = 0
    for m in moments:
        dur = m.end - m.start
        if total_dur + dur <= max_total:
            # Check no overlap with already selected
            if not any(m.start < s.end and s.start < m.end for s in selected):
                selected.append(m)
                total_dur += dur
        if total_dur >= max_total or len(selected) >= 6:
            break

    if len(selected) < 3:
        return None

    # Sort by time for chronological order
    selected.sort(key=lambda c: c.start)
    segments = [(c.start, c.end) for c in selected]
    avg_viral = sum(c.viral_score for c in selected) / len(selected)

    hooks = [c.hook_phrase for c in selected if c.hook_phrase != "(engaging segment)"]
    hook_text = " | ".join(hooks[:3]) if hooks else "Best Moments Compilation"

    return ClipWindow(
        start=segments[0][0], end=segments[-1][1],
        score=avg_viral / 10.0, viral_score=round(avg_viral, 1),
        hook_phrase=f"🔥 HIGHLIGHT REEL: {hook_text}",
        hook_time=segments[0][0],
        segments=segments,
    )


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

def _composite_to_viral(composite):
    curved = math.pow(max(composite, 0), 0.75)
    return round(min(curved * 10.0, 10.0), 1)


def _best_hook_in_window(cand, words):
    cw = cand["clip_words"]
    if not cw:
        return ("(engaging segment)", cand["start"])
    texts = [w.word.lower().strip(".,!?;:\"'()[]{}") for w in cw]
    best_p, best_w, best_t = "", 0, cand["start"]

    for phrases in [ENTERTAINMENT_PHRASES, HOOK_PHRASES]:
        for phrase, weight in phrases:
            pw = phrase.lower().split()
            for i in range(len(texts) - len(pw) + 1):
                if texts[i:i + len(pw)] == pw and weight > best_w:
                    best_p, best_w, best_t = phrase, weight, cw[i].start

    if best_p:
        return (best_p, best_t)
    preview = " ".join(w.word for w in cw[:6])
    if len(cw) > 6:
        preview += "..."
    return (preview, cw[0].start)


def _select_non_overlapping(candidates):
    selected = []
    for c in candidates:
        if not any(c.start < s.end and s.start < c.end for s in selected):
            selected.append(c)
    return selected


def _fallback_clip(video_duration, max_duration):
    return ClipWindow(0.0, min(max_duration, video_duration), 0.0, 0.0,
                      "(fallback: video start)", 0.0)
