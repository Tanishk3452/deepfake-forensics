import os
os.environ["TORCHVISION_DISABLE_TORCHCODEC"] = "1"
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from pathlib import Path
from fastapi import BackgroundTasks
import uvicorn
import uuid
import shutil
from pathlib import Path
import json
from datetime import datetime
import time
import traceback
from fastapi.staticfiles import StaticFiles





# Import ML modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../', 'ml'))
from video_detector import VideoDeepfakeDetector
from audio_detector import AudioDeepfakeDetector
from metadata_analyzer import MetadataAnalyzer
from explainability import ExplainabilityGenerator



app = FastAPI(title="Deepfake Detector API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
UPLOAD_DIR = Path("uploads")
TEMP_DIR = Path("temp")
UPLOAD_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

HEATMAP_VIDEOS = {}



MODEL_CHOICE = "vit"  # ← Change this based on test results

# ===== Initialize detectors =====
video_detector = VideoDeepfakeDetector(model_choice=MODEL_CHOICE)
audio_detector = AudioDeepfakeDetector(
    model_path="../ml_models/wav2vec2_audio.pth"
)
metadata_analyzer = MetadataAnalyzer()
explainability_gen = ExplainabilityGenerator()

# Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.ogg'}



def cleanup_temp_files(file_path: Path):
    import time
    time.sleep(1)  # Windows-safe delay

    try:
        if file_path.exists():
            os.remove(file_path)

        temp_folder = TEMP_DIR / file_path.stem
        if temp_folder.exists():
            shutil.rmtree(temp_folder)

    except Exception as e:
        print(f"Cleanup warning (ignored): {e}")


@app.get("/")
async def root():
    return {
        "message": "Deepfake Detector API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "video_detector": video_detector.is_loaded(),
            "audio_detector": audio_detector.is_loaded()
        }
    }



@app.post("/analyze/video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    file_ext = Path(file.filename).suffix.lower()
    analysis_id = str(uuid.uuid4())
    temp_file = TEMP_DIR / f"{analysis_id}{file_ext}"

    try:
        # Save upload
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # -------- ANALYSIS --------
        video_result = video_detector.analyze(str(temp_file))
        audio_result = audio_detector.analyze_from_video(str(temp_file))
        metadata_result = metadata_analyzer.analyze(str(temp_file))

        final_score = calculate_authenticity_score(
            video_result, audio_result, metadata_result
        )

        # ✅ FIX: convert heatmap path → streaming URL
        heatmap_video_url = None
        heatmap_path = video_result.get("heatmap_video_path")

        if heatmap_path and Path(heatmap_path).exists():
            video_id = str(uuid.uuid4())
            HEATMAP_VIDEOS[video_id] = heatmap_path
            heatmap_video_url = f"/stream/video/{video_id}"




        # -------- RESPONSE (UNCHANGED STRUCTURE) --------
        response = {
            "analysis_id": analysis_id,
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "authenticity_score": final_score,

            "video_analysis": {
                "prediction": video_result["prediction"],
                "confidence": video_result["confidence"],
                "frames_analyzed": video_result["frames_analyzed"],
                "heatmap_video_url": heatmap_video_url   # ✅ ONLY CHANGE
            },

            "audio_analysis": {
                "prediction": audio_result["prediction"],
                "confidence": audio_result["confidence"],
                "synthetic_indicators": audio_result.get("indicators", [])
            },

            "metadata_analysis": {
                "tampering_detected": metadata_result["tampering_detected"],
                "tampering_score": metadata_result["tampering_score"],
                "indicators": metadata_result.get("indicators", [])
            },

            "ethical_disclaimer":
                "This system provides probabilistic forensic analysis, not legal proof."
        }

        

        background_tasks.add_task(cleanup_temp_files, temp_file)
        return JSONResponse(content=response)

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Video analysis failed")

# -------------------- VIDEO STREAM --------------------

@app.get("/stream/video/{video_id}")
def stream_video(video_id: str, request: Request):

    if video_id not in HEATMAP_VIDEOS:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(HEATMAP_VIDEOS[video_id])

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    def iterfile(start: int, end: int):
        with open(video_path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    if range_header:
        match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if not match:
            raise HTTPException(status_code=416, detail="Invalid range")

        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else file_size - 1

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Content-Type": "video/mp4",
        }

        return StreamingResponse(
            iterfile(start, end),
            status_code=206,
            headers=headers,
            media_type="video/mp4",
        )

    return StreamingResponse(
        open(video_path, "rb"),
        media_type="video/mp4",
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
        },
    )


@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze audio for synthetic voice detection
    """
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )
    
    analysis_id = str(uuid.uuid4())
    temp_file = TEMP_DIR / f"{analysis_id}{file_ext}"
    
    try:
        # Save uploaded file
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Audio analysis
        audio_result = audio_detector.analyze(str(temp_file))
        
        # Metadata analysis
        metadata_result = metadata_analyzer.analyze(str(temp_file))
        
        # Calculate score
        final_score = (audio_result["confidence"] * 0.7 + 
                      (100 - metadata_result["tampering_score"]) * 0.3)
        
        response = {
            "analysis_id": analysis_id,
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "authenticity_score": round(final_score, 2),
            "risk_level": get_risk_level(final_score),
            "audio_analysis": {
                "prediction": audio_result["prediction"],
                "confidence": audio_result["confidence"],
                "synthetic_indicators": audio_result.get("indicators", []),
                "spectral_analysis": audio_result.get("spectral_features", {})
            },
            "metadata_analysis": {
                "tampering_detected": metadata_result["tampering_detected"],
                "indicators": metadata_result["indicators"]
            },
            "ethical_disclaimer": "This analysis is for informational purposes only."
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        time.sleep(0.3)
        cleanup_temp_files(temp_file)


def calculate_authenticity_score(video_result, audio_result, metadata_result):
    vw, aw, mw = 0.6, 0.25, 0.15

    avg_fake = video_result.get("avg_fake_probability", 50)

    # Penalize uncertainty
    certainty = abs(avg_fake - 50) * 2   # 0–100
    certainty = max(10, certainty)

    if video_result["prediction"] == "real":
        video_score = 50 + certainty / 2
    else:
        video_score = 50 - certainty / 2

    audio_score = (
        audio_result["confidence"]
        if audio_result["prediction"] == "real"
        else 100 - audio_result["confidence"]
    )

    metadata_score = 100 - metadata_result["tampering_score"]

    final = (
        video_score * vw +
        audio_score * aw +
        metadata_score * mw
    )

    return round(max(0, min(100, final)), 2)



def get_risk_level(score):
    """Determine risk level based on authenticity score"""
    if score >= 80:
        return "LOW"
    elif score >= 60:
        return "MEDIUM"
    elif score >= 40:
        return "HIGH"
    else:
        return "CRITICAL"

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)