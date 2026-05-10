# AutoClip Studio

AutoClip Studio is a fully automated, 100% local AI pipeline that transforms full-length YouTube videos into viral 9:16 vertical shorts. With just a YouTube URL, it downloads the video, transcribes the audio using state-of-the-art AI, analyzes it for high-engagement "viral" moments, and renders cinematic clips complete with dynamic, karaoke-style subtitles.

## 🚀 Features

- **One-Click Pipeline:** Paste a YouTube URL and let the app handle everything (Download → Transcribe → Analyze → Render).
- **100% Local Processing:** No API keys, no subscription fees, no rate limits. All processing, including AI transcription, runs directly on your machine.
- **State-of-the-Art AI:** Uses `faster-whisper` (large-v3 model) for incredibly accurate, word-level timestamps.
- **Privacy First:** Only the finalized short clips are saved locally; original full-length YouTube videos are securely deleted immediately after processing.
- **Viral Engagement Scoring:** Analyzes audio dynamics, reaction words, and hook phrases to automatically find the most engaging parts of a video.
- **Cinematic Rendering:** 
  - Smart framing scales the video perfectly to 9:16.
  - Automatically adds a heavily blurred, darkened background to fill empty space.
  - Generates dynamic, word-level active/inactive karaoke subtitles.
- **God Mode Dashboard:** A beautiful, dark-themed web interface that displays real-time progress, live ETA, active terminal logs, and clip results.
- **Highlight Reels:** Automatically merges non-contiguous high-energy segments into a single compilation.

## 🛠️ Technology Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** HTML, Tailwind CSS, Vanilla JS
- **AI / ML:** faster-whisper, PyTorch
- **Media Processing:** moviepy, yt-dlp, FFmpeg, Pillow, NumPy

## 📦 Prerequisites

- Python 3.10+
- FFmpeg (Must be installed and added to your system PATH)

## ⚡ Quick Start

1. Clone this repository.
2. Install FFmpeg:
   - On Windows: `winget install Gyan.FFmpeg`
3. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   - Double-click `Start_AutoClip.bat` (Windows)
   - OR run `python start_server.py` in your terminal.

The web interface will automatically open in your default browser at `http://127.0.0.1:8000`.
