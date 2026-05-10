"""Quick smoke test for the clipper module with viral scoring."""
from src.transcriber import WordTimestamp
from src.clipper import find_clips

# Mock transcript with multiple hooks at different positions
words = [
    # Hook cluster 1: "the secret is" + "changed my life" (should score high)
    WordTimestamp("the", 10.0, 10.3),
    WordTimestamp("secret", 10.3, 10.8),
    WordTimestamp("is", 10.8, 11.0),
    WordTimestamp("that", 11.0, 11.3),
    WordTimestamp("this", 11.3, 11.6),
    WordTimestamp("changed", 15.0, 15.3),
    WordTimestamp("my", 15.3, 15.5),
    WordTimestamp("life", 15.5, 15.8),
    WordTimestamp("and", 20.0, 20.3),
    WordTimestamp("blew", 20.3, 20.6),
    WordTimestamp("my", 20.6, 20.8),
    WordTimestamp("mind", 20.8, 21.1),
    # Hook cluster 2: far away, single weak hook
    WordTimestamp("do", 200.0, 200.3),
    WordTimestamp("you", 200.3, 200.5),
    WordTimestamp("know", 200.5, 200.8),
    WordTimestamp("what", 200.8, 201.0),
    # End
    WordTimestamp("ending", 300.0, 300.5),
]

# Test: find all clips with viral >= 7
clips = find_clips(words, clip_duration=60.0, viral_threshold=7.0)
print(f"\nClips with viral >= 7.0:")
for c in clips:
    print(f"  {c.start:.0f}s-{c.end:.0f}s | viral={c.viral_score:.1f}/10 | raw={c.score} | hook='{c.hook_phrase}'")

assert len(clips) >= 1, "Should find at least 1 high-viral clip"
assert all(c.viral_score >= 7.0 for c in clips), "All clips should have viral >= 7"

# Test: the strong cluster should beat the weak one
if len(clips) >= 1:
    best = clips[0]
    assert best.viral_score >= 7.0, f"Best clip should be >= 7, got {best.viral_score}"
    print(f"\nBest clip: viral={best.viral_score:.1f}/10, hook='{best.hook_phrase}'")

# Test: very low threshold gets more clips
all_clips = find_clips(words, clip_duration=60.0, viral_threshold=0.0)
print(f"\nAll clips (threshold=0): {len(all_clips)} found")
for c in all_clips:
    print(f"  {c.start:.0f}s-{c.end:.0f}s | viral={c.viral_score:.1f}/10 | hook='{c.hook_phrase}'")

# Test: empty transcript
empty = find_clips([], clip_duration=60.0)
assert len(empty) == 0

# Test: no hooks -> fallback
no_hooks = [WordTimestamp("hello", 0.0, 0.5), WordTimestamp("world", 0.5, 1.0)]
fallback = find_clips(no_hooks, clip_duration=60.0)
assert len(fallback) == 1
assert fallback[0].viral_score == 0.0
assert fallback[0].hook_phrase == "(fallback: video start)"
print(f"\nFallback test: viral={fallback[0].viral_score}, phrase='{fallback[0].hook_phrase}'")

print("\nAll clipper tests PASSED!")
