import os
os.environ["TORCHVISION_DISABLE_TORCHCODEC"] = "1"
os.environ["TORCHCODEC_DISABLE"] = "1"
import torch
import cv2
import uuid
import numpy as np
import mediapipe as mp
from PIL import Image
from pathlib import Path
import subprocess


class VideoDeepfakeDetector:
    def __init__(self, model_choice="vit"):
        """
        Args:
            model_choice: Which model to use
                - "vit": Vision Transformer (best overall)
                - "siglip": SigLIP model (good balance)
                - "ensemble": Use multiple models and vote
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_choice = model_choice
        
        print(f"🔄 Loading deepfake detection model: {model_choice}")
        
        try:
            from transformers import (
                AutoModelForImageClassification, 
                AutoImageProcessor,
                ViTForImageClassification,
                ViTImageProcessor
            )
            
            if model_choice == "vit":
                # Vision Transformer - Very good performance (92% accuracy)
                model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
                self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
                self.processor = ViTImageProcessor.from_pretrained(model_name)
                self.models = [(self.model, self.processor, model_name)]
                
            elif model_choice == "siglip":
                # SigLIP - Good accuracy (94% accuracy)
                model_name = "prithivMLmods/deepfake-detector-model-v1"
                self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.models = [(self.model, self.processor, model_name)]
                
            elif model_choice == "ensemble":
                # Use both models and vote
                print("   Loading multiple models for ensemble...")
                
                vit_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
                vit_model = ViTForImageClassification.from_pretrained(vit_name).to(self.device)
                vit_proc = ViTImageProcessor.from_pretrained(vit_name)
                
                sig_name = "prithivMLmods/deepfake-detector-model-v1"
                sig_model = AutoModelForImageClassification.from_pretrained(sig_name).to(self.device)
                sig_proc = AutoImageProcessor.from_pretrained(sig_name)
                
                self.models = [
                    (vit_model, vit_proc, vit_name),
                    (sig_model, sig_proc, sig_name)
                ]
            else:
                raise ValueError(f"Unknown model choice: {model_choice}")
            
            # Set all models to eval mode
            for model, _, name in self.models:
                model.eval()
                print(f"   ✅ Loaded: {name.split('/')[-1]}")
                print(f"      Labels: {model.config.id2label}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        # Face detection
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.6
        )
        
        # Results directory
        BASE_DIR = Path(__file__).resolve().parent.parent
        self.results_dir = BASE_DIR / "backend" / "api" / "heatmap_videos"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def is_loaded(self):
        return True

    def _predict_face(self, face_rgb):
        """
        Predict if a face is fake or real
        Returns: probability that face is FAKE (0.0 = real, 1.0 = fake)
        """
        img = Image.fromarray(face_rgb)
        
        predictions = []
        
        for model, processor, model_name in self.models:
            # Preprocess
            inputs = processor(images=img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                
                # Get label mapping
                id2label = model.config.id2label
                
                # Find which index is "fake" or "deepfake"
                fake_idx = None
                for idx, label in id2label.items():
                    label_lower = label.lower()
                    if 'fake' in label_lower or 'deepfake' in label_lower or 'synthetic' in label_lower:
                        fake_idx = int(idx)
                        break
                
                # If no "fake" label found, assume index 1 is fake
                if fake_idx is None:
                    fake_idx = 1
                
                fake_prob = probs[0, fake_idx].item()
                predictions.append(fake_prob)
        
        # Return average if ensemble, otherwise just the single prediction
        return np.mean(predictions)

    def _overlay(self, frame_rgb, fake_prob, bbox):
        """Create visualization overlay on frame"""
        overlay = frame_rgb.copy()
        x1, y1, x2, y2 = bbox
        
        # Color based on prediction (green=real, red=fake)
        if fake_prob > 0.7:
            color = (255, 0, 0)  # Strong fake - red
        elif fake_prob > 0.5:
            color = (255, 165, 0)  # Weak fake - orange
        elif fake_prob > 0.3:
            color = (255, 255, 0)  # Uncertain - yellow
        else:
            color = (0, 255, 0)  # Real - green
        
        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        label = f"Fake: {fake_prob*100:.1f}%"
        cv2.putText(overlay, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return overlay

    def analyze(self, video_path, max_frames=120):
        """
        Analyze video for deepfake detection
        """
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            return self._empty("Invalid video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_id = uuid.uuid4().hex
        raw_path = self.results_dir / f"{video_id}_raw.mp4"
        final_path = self.results_dir / f"{video_id}.mp4"

        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        # Sample frames
        sample_indices = np.linspace(0, total - 1, min(max_frames, total)).astype(int)

        predictions = []
        frame_idx = 0
        last_overlay = None

        print(f"\n[VIDEO ANALYSIS] Processing: {Path(video_path).name}")
        print(f"  Total frames: {total}, Sampling: {len(sample_indices)} frames")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Analyze sampled frames
            if frame_idx in sample_indices:
                faces = self.mp_face.process(rgb)

                if faces.detections:
                    # Find largest face
                    d = max(
                        faces.detections,
                        key=lambda x: x.location_data.relative_bounding_box.width *
                                    x.location_data.relative_bounding_box.height
                    )

                    box = d.location_data.relative_bounding_box
                    x1 = max(0, int(box.xmin * w))
                    y1 = max(0, int(box.ymin * h))
                    x2 = min(w, int((box.xmin + box.width) * w))
                    y2 = min(h, int((box.ymin + box.height) * h))

                    face = frame[y1:y2, x1:x2]

                    if face.size > 0:
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        fake_prob = self._predict_face(face_rgb)
                        predictions.append(fake_prob)
                        
                        last_bbox = (x1, y1, x2, y2)
                        last_overlay = self._overlay(rgb, fake_prob, last_bbox)

            # Write frame
            if last_overlay is not None:
                writer.write(cv2.cvtColor(last_overlay, cv2.COLOR_RGB2BGR))
            else:
                writer.write(frame)

            frame_idx += 1

        cap.release()
        writer.release()

        # Convert to web format
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(raw_path),
                "-c:v", "libx264", "-profile:v", "high",
                "-level", "4.0", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(final_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            raw_path.unlink(missing_ok=True)
        except subprocess.CalledProcessError:
            print("⚠️  FFmpeg conversion failed, using raw video")
            final_path = raw_path

        if len(predictions) < 5:
            return self._empty("Insufficient face detections")

        # Statistical analysis
        predictions = np.array(predictions)
        
        # Remove outliers (values too extreme)
        q1 = np.percentile(predictions, 25)
        q3 = np.percentile(predictions, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Keep only non-outlier predictions
        filtered = predictions[(predictions >= lower_bound) & (predictions <= upper_bound)]
        
        if len(filtered) < 3:
            filtered = predictions  # Use all if too few remain
        
        # Smooth with moving average
        if len(filtered) >= 5:
            smoothed = np.convolve(filtered, np.ones(5)/5, mode='valid')
        else:
            smoothed = filtered
        
        mean_prob = np.mean(smoothed)
        median_prob = np.median(smoothed)
        
        # Use median for final decision (more robust to outliers)
        final_prob = median_prob
        
        # Calculate confidence
        std_dev = np.std(smoothed)
        consistency = 1.0 - min(std_dev * 2, 1.0)  # Higher consistency = more confident
        distance_from_middle = abs(final_prob - 0.5) * 2  # 0 to 1
        
        confidence = (distance_from_middle * 0.7 + consistency * 0.3) * 100
        confidence = min(confidence, 100.0)
        
        prediction = "fake" if final_prob > 0.5 else "real"
        
        # Debug info
        print(f"  Faces detected: {len(predictions)} frames")
        print(f"  Raw probabilities - Min: {predictions.min():.3f}, Max: {predictions.max():.3f}")
        print(f"  After filtering - Mean: {mean_prob:.3f}, Median: {median_prob:.3f}, Std: {std_dev:.3f}")
        print(f"  Final decision: {prediction.upper()} (prob={final_prob:.3f}, conf={confidence:.1f}%)\n")

        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "frames_analyzed": len(predictions),
            "avg_fake_probability": round(final_prob * 100, 2),
            "heatmap_video_path": str(final_path),
            "debug_info": {
                "raw_range": [round(predictions.min(), 4), round(predictions.max(), 4)],
                "mean_prob": round(mean_prob, 4),
                "median_prob": round(median_prob, 4),
                "std_dev": round(std_dev, 4)
            }
        }

    def _empty(self, reason):
        return {
            "prediction": "unknown",
            "confidence": 0,
            "frames_analyzed": 0,
            "avg_fake_probability": 50,
            "heatmap_video_path": None,
            "reason": reason
        }