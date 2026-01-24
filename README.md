# Deepfake Forensics

Deepfake Forensics is a full-stack lab for detecting manipulated media. A FastAPI backend runs multiple detectors for video, audio, and metadata, while a React + Vite frontend provides an interactive analysis dashboard with heatmaps, risk scoring, and explainability hints.

## Project Layout

```
backend/
  api/                FastAPI app (`main.py`) with analyze & streaming routes
  requirements.txt    Core backend/ML dependencies
ml/                   Reusable detection modules (video, audio, metadata, explainability)
ml_models/            Stored model weights (e.g., wav2vec2_audio.pth)
frontend/             React + Vite + Tailwind UI
```

## Features

- Upload video or audio and get authenticity scores with risk levels.
- Video pipeline: face detection, frame sampling, transformer classifier, Grad-CAM heatmap video stream.
- Audio pipeline: Hugging Face audio classifiers with segment-level aggregation and confidence calibration.
- Metadata checks: ffprobe-based provenance hints and tampering indicators.
- REST API: `/analyze/video`, `/analyze/audio`, `/stream/video/{id}`, `/health`.
- Frontend: modern dashboard with threat meter, per-modality breakdown, and streamed heatmap playback.

## Prerequisites

- Python 3.10+ (CPU works; GPU optional if CUDA available)
- Node.js 18+ and npm (or pnpm/yarn)
- FFmpeg available on PATH (required for audio extraction and metadata)
- (Optional) Git LFS for large model files

## Backend Setup (FastAPI)

```bash
cd backend/api
python -m venv .venv
# Activate the venv (PowerShell)
.venv\Scripts\Activate.ps1
# or bash: source .venv/bin/activate
pip install --upgrade pip
pip install -r ../requirements.txt
# If you need the full pinned set, use: pip install -r requirements.txt

# Run the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Notes:
- First run will download Hugging Face models for audio/video detectors.
- Heatmap videos are written to `backend/api/heatmap_videos/` and served via `/stream/video/{id}`.
- Upload and temp files go to `uploads/` and `temp/` (auto-created).

## Frontend Setup (Vite + React)

```bash
cd frontend
npm install
npm run dev
```

The UI expects the backend at `http://localhost:8000` (configured in `src/App.jsx`). Vite defaults to port 5173.

## Using the API Directly

Analyze a video:

```bash
curl -X POST "http://localhost:8000/analyze/video" \
  -F "file=@sample.mp4" 
```

Analyze audio:

```bash
curl -X POST "http://localhost:8000/analyze/audio" \
  -F "file=@sample.wav"
```

Stream a generated heatmap video:

```bash
curl -OJ "http://localhost:8000/stream/video/<video_id>"
```

Health check:

```bash
curl http://localhost:8000/health
```

## Key Modules

- Video detection: `ml/video_detector.py` – face detection, transformer inference, Grad-CAM overlay, and heatmap mp4 export.
- Audio detection: `ml/audio_detector.py` – Hugging Face audio classifiers with label inversion and segment robustness.
- Metadata analysis: `ml/metadata_analyzer.py` – ffprobe-driven tamper scoring and provenance hints.
- Explainability: `ml/explainability.py` – summarizes findings and recommendations.

## Model Weights

- `ml_models/wav2vec2_audio.pth` is included for audio; other Hugging Face weights download at runtime.
- If running offline, pre-download models and point environment cache (e.g., `HF_HOME`).

## Development Tips

- Keep FFmpeg updated; missing ffprobe/ffmpeg will break audio extraction and metadata checks.
- For GPU inference, install CUDA builds of `torch`, `torchvision`, and `torchaudio` matching your CUDA version.
- Tailwind styles live in `frontend/src` (`App.css`, `index.css`); adjust colors/gradients there.
- Set `MODEL_CHOICE` and `AUDIO_MODEL_CHOICE` in `backend/api/main.py` to switch detector variants.

## Troubleshooting

- "Analysis failed" from frontend: confirm backend is running and CORS allows `http://localhost:5173`.
- Missing models: clear `~/.cache/huggingface` and re-run, or set `HF_HOME` to a writable path.
- FFmpeg errors: ensure `ffmpeg` and `ffprobe` are on PATH (`ffmpeg -version`).
