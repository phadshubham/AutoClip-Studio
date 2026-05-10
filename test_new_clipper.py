"""Test the v3 clipper with entertainment/Sidemen-style content."""
import sys
sys.path.insert(0, "app")
from core.transcriber import WordTimestamp
from core.clipper import find_clips

words = []
t = 0.0
sentences = [
    # Classic Sidemen-style banter and reactions
    "Right so today we're doing the craziest challenge ever.",
    "Oh my god are you serious right now?",
    "Bruv that is absolutely insane.",
    "Nah nah nah there's no way you just did that.",
    "That's a violation fam honestly.",
    "I can't believe what I'm looking at right now.",
    "Wait what how is that even possible?",
    "You're moving mad bro literally moving mad.",
    "That's peak that is genuinely peak.",
    "Allow it fam just allow it.",
    "He's actually finished it's over for him.",
    "Let's go that was incredible!",
    "Oh no this is going to be bad.",
    "Are you kidding me right now?",
    "Bro that's the funniest thing I've ever seen.",
    "No chance no chance at all mate.",
    "Honestly that's jokes man that's actually jokes.",
    "I'm dead I'm literally dead right now.",
    "You lot are ridiculous absolutely ridiculous.",
    "Rah that's actually mad you know.",
    "So basically what happened was something crazy.",
    "Think about it for a second though.",
    "The key is to just keep going and never give up.",
    "Studies show that people who start early win more.",
]

for sent in sentences:
    for word in sent.split():
        words.append(WordTimestamp(word, round(t, 3), round(t + 0.3, 3)))
        t += 0.4
    t += 0.8

print(f"Total: {len(words)} words, {t:.1f}s\n")

clips = find_clips(words, max_duration=60, min_duration=15, viral_threshold=3.0, max_clips=5)
print(f"\nFound {len(clips)} clips:")
for i, c in enumerate(clips, 1):
    merged = " [HIGHLIGHT REEL]" if c.is_merged else ""
    print(f"  Clip {i}: {c.start:.1f}s-{c.end:.1f}s ({c.duration:.0f}s) viral={c.viral_score} hook=\"{c.hook_phrase}\"{merged}")

print("\n✅ Test PASSED!" if clips else "\n❌ Test FAILED")
