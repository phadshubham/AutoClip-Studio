from moviepy import ColorClip, VideoClip, CompositeVideoClip
import numpy as np

def make_rgb_frame(t):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = 255 # Red
    return frame

def make_alpha_frame(t):
    if t < 1:
        return np.ones((100, 100), dtype=float) # Solid
    else:
        return np.zeros((100, 100), dtype=float) # Transparent

bg = ColorClip(size=(200, 200), color=(0, 255, 0), duration=2)
overlay = VideoClip(make_rgb_frame, duration=2).with_mask(VideoClip(make_alpha_frame, is_mask=True, duration=2))

try:
    final = CompositeVideoClip([bg, overlay.with_position("center")])
    print("Overlay has mask?", overlay.mask is not None)
    final.write_videofile("test_overlay2.mp4", fps=24, codec="libx264")
    print("Success")
except Exception as e:
    print(f"Error: {e}")
