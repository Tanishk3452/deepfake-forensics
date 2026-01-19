import cv2
import numpy as np
import torch
import base64
from io import BytesIO
from PIL import Image


class ExplainabilityGenerator:
    def __init__(self):
        self.gradcam_enabled = False  # detector does not expose layers

    # ---------------------------
    # Image helper
    # ---------------------------

    def image_to_base64(self, image):
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)

            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG", quality=85)

            return "data:image/jpeg;base64," + base64.b64encode(
                buffered.getvalue()
            ).decode()

        except Exception as e:
            print(f"Base64 conversion error: {e}")
            return None

    # ---------------------------
    # Main API
    # ---------------------------

    def generate_video_explanation(self, video_result, video_path):
        explanation = {
            "summary": self._generate_summary(video_result),
            "key_findings": self._extract_key_findings(video_result),
            "confidence_breakdown": self._generate_confidence_breakdown(video_result),
            "visual_evidence": [],
            "recommendations": self._generate_recommendations(video_result),
        }

        # Video detector does NOT provide suspicious frames
        # so visual evidence is intentionally skipped

        return explanation

    # ---------------------------
    # Explanation sections
    # ---------------------------

    def _generate_summary(self, video_result):
        prediction = video_result.get("prediction", "unknown")
        confidence = video_result.get("confidence", 0)

        if prediction == "fake":
            return (
                f"Analysis indicates this video likely contains deepfake manipulation "
                f"(confidence score: {confidence})."
            )
        elif prediction == "real":
            return (
                f"Analysis suggests this video is likely authentic "
                f"(confidence score: {confidence})."
            )
        else:
            return "Unable to determine authenticity with confidence."

    def _extract_key_findings(self, video_result):
        findings = []

        prediction = video_result.get("prediction", "unknown")
        findings.append(f"Overall prediction: {prediction.upper()}")

        frames = video_result.get("frames_analyzed", 0)
        findings.append(f"Analyzed {frames} video frames")

        deepfake_prob = video_result.get("avg_deepfake_probability", 0)
        findings.append(f"Average deepfake probability: {deepfake_prob:.2f}%")

        return findings

    def _generate_confidence_breakdown(self, video_result):
        return {
            "overall_confidence": video_result.get("confidence", 0),
            "frames_analyzed": video_result.get("frames_analyzed", 0),
            "faces_detected": "Not reported by detector",
            "suspicious_regions": 0,
            "deepfake_probability": video_result.get("avg_fake_probability", 0),
        }

    def _generate_recommendations(self, video_result):
        recommendations = []

        prediction = video_result.get("prediction", "unknown")
        confidence = video_result.get("confidence", 0)

        if prediction == "fake":
            recommendations.append("🔴 Do not share this content without verification")
            recommendations.append("🔍 Consider additional deepfake analysis tools")
            recommendations.append("📢 Report suspicious content if applicable")

            if confidence < 5:
                recommendations.append("⚠️ Low confidence score – manual review advised")

        elif prediction == "real":
            if confidence < 5:
                recommendations.append(
                    "⚠️ Likely authentic, but confidence score is low"
                )
                recommendations.append("🔍 Extra verification recommended")
            else:
                recommendations.append("✓ Content appears authentic")
                recommendations.append("📋 Verify source and context before sharing")

        else:
            recommendations.append("⚠️ Authenticity could not be determined")
            recommendations.append("🔍 Manual verification recommended")

        recommendations.append("ℹ️ This analysis is for informational purposes only")

        return recommendations
