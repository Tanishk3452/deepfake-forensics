import os
# 🔒 Disable TorchCodec completely (important for your setup)
os.environ["TORCHVISION_DISABLE_TORCHCODEC"] = "1"
os.environ["TORCHCODEC_DISABLE"] = "1"

import torch
import torchaudio
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class AudioDeepfakeDetector:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        # 🔹 Load processor
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )

        # 🔹 Load base Wav2Vec2
        self.base_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        # 🔹 Strong classifier head (IMPORTANT)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 1)
        )

        self.base_model.to(self.device)
        self.classifier.to(self.device)

        self._load_weights()
        self.base_model.eval()
        self.classifier.eval()

        print("✅ Audio deepfake model loaded")

    # --------------------------------------------------

    def _load_weights(self):
        """
        Load pretrained classifier weights
        """
        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=False
        )

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # Load only classifier weights if present
        classifier_state = {
            k.replace("classifier.", ""): v
            for k, v in checkpoint.items()
            if "classifier" in k
        }

        if classifier_state:
            self.classifier.load_state_dict(classifier_state, strict=False)
        else:
            # fallback – try loading whole checkpoint
            self.classifier.load_state_dict(checkpoint, strict=False)

    # --------------------------------------------------

    def is_loaded(self):
        return True

    # --------------------------------------------------

    def _load_audio(self, path: str) -> np.ndarray:
        waveform, sr = torchaudio.load(path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        # Resample to 16kHz (Wav2Vec2 requirement)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=16000
            )(waveform)

        return waveform.numpy()

    # --------------------------------------------------

    def _calibrate_confidence(self, prob: float, variance: float) -> float:
        """
        Calibrated, UI-friendly confidence
        """
        # distance from uncertainty
        distance = abs(prob - 0.5)

        # nonlinear stretch
        confidence = (distance ** 0.7) * 200

        # temporal variance boost (synthetic audio is less stable)
        confidence += min(20, variance * 10)

        # clamp for sanity
        confidence = max(15, min(100, confidence))

        return round(confidence, 2)

    # --------------------------------------------------

    def analyze(self, audio_path: str) -> dict:
        audio = self._load_audio(audio_path)

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self.base_model(**inputs).last_hidden_state

            # Mean pooling
            pooled = features.mean(dim=1)

            # Temporal variance (fake voices are unstable)
            variance = features.var(dim=1).mean().item()

            logits = self.classifier(pooled)
            prob = torch.sigmoid(logits).item()

        confidence = self._calibrate_confidence(prob, variance)

        return {
            "prediction": "fake" if prob > 0.5 else "real",
            "confidence": confidence,
            "fake_probability": round(prob * 100, 2),
            "variance_score": round(variance, 4)
        }

    # --------------------------------------------------

    def analyze_from_video(self, video_path: str) -> dict:
        """
        Extract audio using FFmpeg and analyze
        """
        import subprocess

        tmp_audio = Path(video_path).with_suffix(".wav")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            str(tmp_audio)
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        try:
            result = self.analyze(str(tmp_audio))
        finally:
            tmp_audio.unlink(missing_ok=True)

        return result
