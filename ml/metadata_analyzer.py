import subprocess
import json
from datetime import datetime
from pathlib import Path
import re
import math

class MetadataAnalyzer:
    def __init__(self):
        self.suspicious_tools = [
            'deepfacelab', 'faceswap', 'deepfake', 'fakeapp',
            'synthetic', 'generated', 'ai-generated'
        ]
        
    def extract_metadata(self, file_path):
        """Extract metadata using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metadata = json.loads(result.stdout)
            return metadata
            
        except subprocess.CalledProcessError as e:
            print(f"Metadata extraction failed: {e}")
            return None
        except FileNotFoundError:
            print("ffprobe not found. Please install ffmpeg.")
            return None
        except json.JSONDecodeError:
            print("Failed to parse metadata JSON")
            return None
    
    def check_encoding_history(self, metadata):
        """Detect signs of multiple re-encodings"""
        indicators = []
        score = 0
        
        if not metadata:
            return indicators, 0
        
        format_info = metadata.get('format', {})
        
        # Check for multiple codec changes
        nb_streams = format_info.get('nb_streams', 0)
        if nb_streams > 3:
            indicators.append("Multiple streams detected (possible re-encoding)")
            score += 10
        
        # Check format name for conversions
        format_name = format_info.get('format_name', '').lower()
        if ',' in format_name:
            indicators.append(f"Multiple format conversions: {format_name}")
            score += 15
        
        # Check for unusual bit rates
        bit_rate = format_info.get('bit_rate')
        if bit_rate:
            try:
                br = int(bit_rate)
                # Very low or very high bit rates can indicate manipulation
                if br < 100000 or br > 50000000:
                    indicators.append(f"Unusual bit rate: {br}")
                    score += 10
            except ValueError:
                pass
        
        return indicators, score
    
    def check_creation_data(self, metadata):
        """Check creation date and modification indicators"""
        indicators = []
        score = 0
        
        if not metadata:
            return indicators, 0
        
        format_info = metadata.get('format', {})
        tags = format_info.get('tags', {})
        
        # Check for missing creation date
        creation_time = tags.get('creation_time')
        if not creation_time:
            indicators.append("Missing creation timestamp")
            score += 15
        else:
            # Check if creation date is suspiciously recent
            try:
                created = datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                age_days = (datetime.now(created.tzinfo) - created).days
                
                if age_days < 1:
                    indicators.append("File created very recently")
                    score += 5
            except:
                pass
        
        # Check for encoding tool
        encoder = tags.get('encoder', '').lower()
        if encoder:
            # Check against suspicious tools
            for suspicious in self.suspicious_tools:
                if suspicious in encoder:
                    indicators.append(f"Suspicious encoder detected: {encoder}")
                    score += 30
                    break
        else:
            indicators.append("Missing encoder information")
            score += 10
        
        return indicators, score
    
    def check_stream_consistency(self, metadata):
        """Check for inconsistencies in stream data"""
        indicators = []
        score = 0
        
        if not metadata:
            return indicators, 0
        
        streams = metadata.get('streams', [])
        
        if len(streams) == 0:
            indicators.append("No streams detected")
            return indicators, 50
        
        video_streams = [s for s in streams if s.get('codec_type') == 'video']
        audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
        
        # Check video stream properties
        for vs in video_streams:
            # Check for unusual frame rates
            avg_frame_rate = vs.get('avg_frame_rate', '0/0')
            try:
                num, den = map(int, avg_frame_rate.split('/'))
                if den != 0:
                    fps = num / den
                    if fps < 15 or fps > 120:
                        indicators.append(f"Unusual frame rate: {fps:.2f} fps")
                        score += 10
            except:
                pass
            
            # Check resolution
            width = vs.get('width', 0)
            height = vs.get('height', 0)
            if width > 0 and height > 0:
                aspect = width / height
                # Check for unusual aspect ratios
                if aspect < 0.5 or aspect > 3.0:
                    indicators.append(f"Unusual aspect ratio: {aspect:.2f}")
                    score += 5
        
        # Check audio stream properties
        for aus in audio_streams:
            sample_rate = aus.get('sample_rate')
            if sample_rate:
                try:
                    sr = int(sample_rate)
                    # Common sample rates: 8000, 16000, 22050, 44100, 48000
                    common_rates = [8000, 16000, 22050, 44100, 48000, 96000]
                    if sr not in common_rates:
                        indicators.append(f"Unusual sample rate: {sr} Hz")
                        score += 5
                except ValueError:
                    pass
        
        return indicators, score
    
    def check_metadata_stripping(self, metadata):
        """Detect if metadata has been stripped or minimized"""
        indicators = []
        score = 0
        
        if not metadata:
            indicators.append("No metadata available")
            return indicators, 40
        
        format_info = metadata.get('format', {})
        tags = format_info.get('tags', {})
        
        # Check number of metadata tags
        if len(tags) < 3:
            indicators.append("Minimal metadata tags (possible stripping)")
            score += 20
        
        # Check for common missing fields
        common_fields = ['encoder', 'creation_time', 'duration']
        missing = [f for f in common_fields if f not in tags]
        
        if len(missing) > 1:
            indicators.append(f"Missing common metadata fields: {', '.join(missing)}")
            score += 15
        
        return indicators, score
    
    def generate_provenance_hints(self, metadata, tampering_score):
        """Generate hints about file provenance"""
        hints = []
        
        if not metadata:
            return ["Unable to determine provenance - no metadata available"]
        
        format_info = metadata.get('format', {})
        tags = format_info.get('tags', {})
        
        # Tool/encoder hints
        encoder = tags.get('encoder', 'Unknown')
        hints.append(f"Encoding tool: {encoder}")
        
        # Format hints
        format_name = format_info.get('format_long_name', 'Unknown')
        hints.append(f"Container format: {format_name}")
        
        # Creation time hints
        creation_time = tags.get('creation_time')
        if creation_time:
            hints.append(f"Created: {creation_time}")
        else:
            hints.append("Creation time: Not available")
        
        # Tampering assessment
        if tampering_score > 60:
            hints.append("⚠️ High likelihood of post-processing or manipulation")
        elif tampering_score > 40:
            hints.append("⚠️ Moderate signs of editing or re-encoding")
        elif tampering_score > 20:
            hints.append("⚠️ Minor editing indicators detected")
        else:
            hints.append("✓ Metadata appears relatively intact")
        
        return hints

    def calibrated_score(self, strong, medium, weak):
        weighted_sum = strong * 1.0 + medium * 0.6 + weak * 0.3
        return round(100 * (1 - math.exp(-weighted_sum / 60)), 2)


    def analyze(self, file_path):
        metadata = self.extract_metadata(file_path)

        if metadata is None:
            return {
                "tampering_detected": False,
                "tampering_score": 25,
                "indicators": ["Metadata unavailable (platform stripped or inaccessible)"],
                "provenance": ["Metadata missing – common on social platforms"],
                "metadata_available": False
            }

        encoding_ind, enc_score = self.check_encoding_history(metadata)
        creation_ind, cre_score = self.check_creation_data(metadata)
        stream_ind, str_score = self.check_stream_consistency(metadata)
        strip_ind, sti_score = self.check_metadata_stripping(metadata)

        indicators = encoding_ind + creation_ind + stream_ind + strip_ind

        # 🧠 Categorized scoring
        strong = 0
        medium = 0
        weak = 0

        for i in indicators:
            i_low = i.lower()
            if "suspicious encoder" in i_low or "deepfake" in i_low:
                strong += 35
            elif any(k in i_low for k in ["re-encoding", "multiple", "unusual frame", "bit rate"]):
                medium += 20
            else:
                weak += 10

        score = self.calibrated_score(strong, medium, weak)

        provenance = self.generate_provenance_hints(metadata, score)

        return {
            "tampering_detected": score > 45,
            "tampering_score": score,
            "indicators": indicators if indicators else ["No metadata anomalies detected"],
            "provenance": provenance,
            "metadata_available": True
        }
