import { useState } from "react";
import {
  Upload,
  AlertTriangle,
  XCircle,
  Loader2,
  FileVideo,
  FileAudio,
  ShieldAlert,
} from "lucide-react";

const API_BASE = "http://localhost:8000";

/* ------------------ UI Helpers ------------------ */

function Badge({ level }) {
  const map = {
    SAFE: "bg-green-500/10 text-green-400 border-green-500/40",
    SUSPICIOUS: "bg-yellow-500/10 text-yellow-400 border-yellow-500/40",
    CRITICAL: "bg-red-500/10 text-red-400 border-red-500/40",
  };

  return (
    <span
      className={`px-4 py-1 rounded-full border text-xs font-semibold tracking-wider ${map[level]}`}
    >
      {level}
    </span>
  );
}

function EvidenceCard({ title, icon, items }) {
  return (
    <div className="relative rounded-2xl p-[1px] bg-gradient-to-br from-blue-500/30 via-purple-500/30 to-pink-500/30">
      <div className="bg-[#050b18] rounded-2xl p-6">
        <h3 className="flex items-center gap-2 text-lg font-semibold mb-4">
          {icon} {title}
        </h3>
        <ul className="space-y-2 text-sm text-white/80">
          {items.map((item, i) => (
            <li key={i} className="flex gap-2">
              <span className="text-blue-400">•</span>
              {item}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function HeatmapGallery({ heatmaps }) {
  if (!heatmaps || heatmaps.length === 0) return null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-6">
      {heatmaps.map((img, i) => (
        <img
          key={i}
          src={`data:image/jpeg;base64,${img}`}
          className="rounded-xl border border-white/20"
        />
      ))}
    </div>
  );
}

function HeatmapVideo({ video }) {
  if (!video) return null;

  return (
    <div className="mt-10">
      <h3 className="text-lg font-semibold mb-4">
        Heatmap Forensic Video (Grad-CAM)
      </h3>

      <video
        key={video} // 👈 ADD THIS LINE
        controls
        playsInline
        preload="auto"
        crossOrigin="anonymous"
        className="w-full rounded-xl border border-white/20"
      >
        <source src={`http://localhost:8000${video}`} type="video/mp4" />
        Your browser does not support the video tag.
      </video>
    </div>
  );
}

/* ------------------ Main App ------------------ */

export default function App() {
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState("video");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setResults(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const endpoint =
        fileType === "video" ? "/analyze/video" : "/analyze/audio";

      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error();
      setResults(await res.json());
    } catch {
      setError("Forensic analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const score = results?.authenticity_score ?? 0;
  const verdict =
    score >= 70 ? "SAFE" : score >= 40 ? "SUSPICIOUS" : "CRITICAL";

  return (
    <div className="min-h-screen relative bg-gradient-to-br from-[#050b18] via-[#0a1226] to-[#020617] text-white overflow-hidden">
      {/* Cyber Grid */}
      <div className="absolute inset-0 bg-[radial-gradient(#ffffff15_1px,transparent_1px)] [background-size:20px_20px]" />

      {/* Header */}
      <header className="relative border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-6 flex justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-wide">
              Deepfake Forensics Lab
            </h1>
            <p className="text-sm text-white/60">
              AI-Powered Media Authenticity Intelligence
            </p>
          </div>
          <div className="flex items-center gap-2 bg-green-500/10 border border-green-500/30 px-3 py-1 rounded-full">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            System Online
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="relative max-w-7xl mx-auto px-6 py-10">
        {/* Upload */}
        <div className="relative rounded-3xl p-[1px] bg-gradient-to-r from-blue-500/40 via-purple-500/40 to-pink-500/40 mb-10">
          <div className="bg-[#050b18] rounded-3xl p-8">
            <h2 className="text-xl font-semibold mb-6">
              Initiate Forensic Scan
            </h2>

            <div className="flex gap-4 mb-6">
              {["video", "audio"].map((t) => (
                <button
                  key={t}
                  onClick={() => setFileType(t)}
                  className={`px-4 py-2 rounded-lg border ${
                    fileType === t
                      ? "bg-blue-600 border-blue-500"
                      : "border-white/20"
                  }`}
                >
                  {t === "video" ? (
                    <FileVideo size={18} />
                  ) : (
                    <FileAudio size={18} />
                  )}{" "}
                  {t.toUpperCase()}
                </button>
              ))}
            </div>

            <label className="block border-2 border-dashed border-white/20 rounded-xl p-10 text-center cursor-pointer hover:border-blue-500 transition">
              <Upload className="mx-auto mb-4 text-white/50" size={36} />
              <p className="text-white/70">Drop media or click to upload</p>
              <input
                type="file"
                className="hidden"
                accept={fileType === "video" ? "video/*" : "audio/*"}
                onChange={(e) => setFile(e.target.files[0])}
              />
            </label>

            {file && (
              <p className="mt-4 text-sm text-white/60">Loaded: {file.name}</p>
            )}

            {error && (
              <p className="mt-4 text-red-400 flex gap-2">
                <XCircle size={18} /> {error}
              </p>
            )}

            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className="mt-6 w-full py-3 bg-blue-600 hover:bg-blue-700 rounded-lg flex justify-center gap-2 disabled:opacity-50"
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin" /> Scanning…
                </>
              ) : (
                <>
                  <ShieldAlert /> Run Forensic Scan
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results */}
        {results && (
          <>
            {/* Verdict */}
            <div className="grid md:grid-cols-3 gap-6 mb-10">
              <div className="relative overflow-hidden rounded-2xl p-[1px] bg-gradient-to-br from-blue-500/40 to-purple-500/40">
                <div className="bg-[#050b18] rounded-2xl p-6 text-center">
                  <p className="text-sm text-white/60">Authenticity Score</p>
                  <p className="text-5xl font-bold mt-2">{score}%</p>
                  <div className="mt-3 flex justify-center">
                    <Badge level={verdict} />
                  </div>

                  {/* Threat Meter */}
                  <div className="mt-5">
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          verdict === "SAFE"
                            ? "bg-green-400"
                            : verdict === "SUSPICIOUS"
                            ? "bg-yellow-400"
                            : "bg-red-400"
                        }`}
                        style={{ width: `${score}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <div className="md:col-span-2 bg-white/5 border border-white/10 rounded-2xl p-6">
                <h3 className="text-lg font-semibold mb-4">
                  Threat Assessment Pipeline
                </h3>
                <ul className="space-y-3 text-sm text-white/70">
                  <li>• Media Ingestion & Normalization</li>
                  <li>• Neural Feature Extraction</li>
                  <li>• Cross-Modal Consistency Check</li>
                  <li>• Final Authenticity Verdict</li>
                </ul>
              </div>
            </div>

            {/* Evidence */}
            <div className="grid md:grid-cols-2 gap-6">
              {results.video_analysis && (
                <EvidenceCard
                  title="Video Forensics"
                  icon={<FileVideo />}
                  items={[
                    `Prediction: ${results.video_analysis.prediction}`,
                    `Confidence: ${results.video_analysis.confidence}%`,
                    `Frames Analyzed: ${results.video_analysis.frames_analyzed}`,
                  ]}
                />
              )}

              {results.audio_analysis && (
                <EvidenceCard
                  title="Audio Forensics"
                  icon={<FileAudio />}
                  items={[
                    `Prediction: ${results.audio_analysis.prediction}`,
                    `Confidence: ${results.audio_analysis.confidence}%`,
                  ]}
                />
              )}

              {results.metadata_analysis && (
                <EvidenceCard
                  title="Metadata Integrity"
                  icon={<AlertTriangle />}
                  items={[
                    results.metadata_analysis.tampering_detected
                      ? "Metadata Tampering Detected"
                      : "No Metadata Anomalies",
                    ...(results.metadata_analysis.indicators || []),
                  ]}
                />
              )}
            </div>

            {results.video_analysis?.heatmap_video_url && (
              <HeatmapVideo video={results.video_analysis.heatmap_video_url} />
            )}

            <div className="mt-10 bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4 text-sm text-yellow-300">
              ⚠ This system provides probabilistic forensic analysis, not legal
              proof.
            </div>
          </>
        )}
      </main>

      <footer className="border-t border-white/10 text-center text-xs text-white/50 py-6">
        Manthan 1.0 · MANIT Bhopal · Team NULL BUDDIES · NIT Raipur
      </footer>
    </div>
  );
}
