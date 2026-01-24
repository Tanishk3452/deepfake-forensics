import { useState } from "react";
import {
  Upload,
  AlertTriangle,
  XCircle,
  Loader2,
  FileVideo,
  FileAudio,
  ShieldAlert,
  Activity,
  Zap,
  Eye,
  Database,
  Lock,
  Shield,
  AlertCircle,
} from "lucide-react";

const API_BASE = "https://deepfake-forensics-3.onrender.com";

/* ------------------ UI Helpers ------------------ */

function Badge({ level }) {
  const map = {
    SAFE: "bg-emerald-500/20 text-emerald-300 border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.3)]",
    SUSPICIOUS:
      "bg-amber-500/20 text-amber-300 border-amber-500/50 shadow-[0_0_15px_rgba(245,158,11,0.3)]",
    CRITICAL:
      "bg-rose-500/20 text-rose-300 border-rose-500/50 shadow-[0_0_15px_rgba(244,63,94,0.3)]",
  };

  return (
    <span
      className={`px-5 py-2 rounded-full border-2 text-sm font-bold tracking-widest uppercase ${map[level]} animate-pulse`}
    >
      {level}
    </span>
  );
}

function MetricCard({ icon, label, value, color = "blue" }) {
  const gradients = {
    blue: "from-blue-500/20 via-cyan-500/20 to-blue-500/20",
    purple: "from-purple-500/20 via-pink-500/20 to-purple-500/20",
    green: "from-emerald-500/20 via-teal-500/20 to-emerald-500/20",
    red: "from-rose-500/20 via-red-500/20 to-rose-500/20",
  };

  return (
    <div className={`relative rounded-xl overflow-hidden group`}>
      <div
        className={`absolute inset-0 bg-gradient-to-br ${gradients[color]} opacity-50 group-hover:opacity-70 transition-opacity`}
      />
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" />
      <div className="relative p-6 border border-white/10">
        <div className="flex items-center gap-3 mb-3">
          <div
            className={`p-2 rounded-lg bg-gradient-to-br ${gradients[color]}`}
          >
            {icon}
          </div>
          <span className="text-xs text-white/60 uppercase tracking-wider">
            {label}
          </span>
        </div>
        <p className="text-3xl font-bold">{value}</p>
      </div>
    </div>
  );
}

function AnalysisSection({ title, icon, data, gradient }) {
  return (
    <div className="relative group">
      <div
        className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-20 transition-opacity rounded-2xl blur-xl`}
      />
      <div className="relative bg-black/40 backdrop-blur-md border border-white/10 rounded-2xl p-6 hover:border-white/20 transition-all">
        <div className="flex items-center gap-3 mb-6">
          <div className={`p-3 rounded-xl bg-gradient-to-br ${gradient}`}>
            {icon}
          </div>
          <h3 className="text-xl font-bold tracking-wide">{title}</h3>
        </div>

        <div className="space-y-4">
          {Object.entries(data).map(([key, value]) => (
            <div
              key={key}
              className="flex justify-between items-center py-2 border-b border-white/5"
            >
              <span className="text-sm text-white/70 capitalize">
                {key.replace(/_/g, " ")}
              </span>
              <span className="font-semibold text-white/90">{value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function HeatmapVideo({ video }) {
  if (!video) return null;

  return (
    <div className="relative group mt-10">
      <div className="absolute inset-0 bg-gradient-to-r from-purple-500/30 via-pink-500/30 to-red-500/30 rounded-2xl blur-2xl opacity-0 group-hover:opacity-100 transition-opacity" />
      <div className="relative bg-black/60 backdrop-blur-md border border-white/10 rounded-2xl p-6">
        <div className="flex items-center gap-3 mb-4">
          <Eye className="text-purple-400" size={24} />
          <h3 className="text-xl font-bold">
            Neural Activation Heatmap (Grad-CAM)
          </h3>
        </div>

        <video
          key={video}
          controls
          playsInline
          preload="auto"
          crossOrigin="anonymous"
          className="w-full rounded-xl border-2 border-purple-500/30 shadow-[0_0_30px_rgba(168,85,247,0.3)]"
        >
          <source src={`http://localhost:8000${video}`} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      </div>
    </div>
  );
}

function ThreatMeter({ score }) {
  const getColor = () => {
    if (score >= 70)
      return {
        bg: "bg-emerald-400",
        shadow: "shadow-emerald-400/50",
        glow: "rgba(16,185,129,0.5)",
      };
    if (score >= 40)
      return {
        bg: "bg-amber-400",
        shadow: "shadow-amber-400/50",
        glow: "rgba(245,158,11,0.5)",
      };
    return {
      bg: "bg-rose-400",
      shadow: "shadow-rose-400/50",
      glow: "rgba(244,63,94,0.5)",
    };
  };

  const color = getColor();

  return (
    <div className="relative">
      <div className="h-4 bg-white/5 rounded-full overflow-hidden border border-white/10">
        <div
          className={`h-full ${color.bg} ${color.shadow} shadow-lg transition-all duration-1000 ease-out`}
          style={{
            width: `${score}%`,
            boxShadow: `0 0 20px ${color.glow}`,
          }}
        />
      </div>
      <div className="flex justify-between mt-2 text-xs text-white/50">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
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
    <div className="min-h-screen relative bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(59,130,246,0.1),transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(139,92,246,0.1),transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_80%,rgba(236,72,153,0.1),transparent_50%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />
      </div>

      {/* Scanline Effect */}
      <div
        className="absolute inset-0 pointer-events-none bg-gradient-to-b from-transparent via-blue-500/5 to-transparent animate-[scan_8s_linear_infinite]"
        style={{ height: "100px" }}
      />

      {/* Header */}
      <header className="relative border-b border-white/10 backdrop-blur-xl bg-black/20">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex justify-between items-start">
            <div>
              <div className="flex items-center gap-4 mb-2">
                <Shield className="text-blue-400" size={40} />
                <h1 className="text-4xl font-black tracking-tight bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                  DEEPFAKE FORENSICS LAB
                </h1>
              </div>
              <p className="text-sm text-white/60 ml-14">
                Advanced Neural Intelligence • Media Authenticity Verification
                System
              </p>
            </div>
            <div className="flex flex-col items-end gap-2">
              <div className="flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/30 px-4 py-2 rounded-full">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                </span>
                <span className="text-xs font-semibold text-emerald-300">
                  SYSTEM ACTIVE
                </span>
              </div>
              <div className="flex items-center gap-2 text-xs text-white/40">
                <Activity size={14} />
                <span>Neural Network: v4.5.1</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="relative max-w-7xl mx-auto px-6 py-12">
        {/* Upload Section */}
        <div className="relative mb-12 group">
          <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-3xl blur-lg opacity-25 group-hover:opacity-50 transition-opacity" />
          <div className="relative bg-black/60 backdrop-blur-xl border border-white/10 rounded-3xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <Zap className="text-yellow-400" size={28} />
              <h2 className="text-2xl font-bold">Forensic Analysis Protocol</h2>
            </div>

            <div className="flex gap-4 mb-8">
              {["video", "audio"].map((t) => (
                <button
                  key={t}
                  onClick={() => setFileType(t)}
                  className={`flex items-center gap-3 px-6 py-3 rounded-xl border-2 transition-all ${
                    fileType === t
                      ? "bg-gradient-to-r from-blue-600 to-purple-600 border-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.5)]"
                      : "border-white/20 hover:border-white/40 bg-white/5"
                  }`}
                >
                  {t === "video" ? (
                    <FileVideo size={20} />
                  ) : (
                    <FileAudio size={20} />
                  )}
                  <span className="font-semibold">
                    {t.toUpperCase()} ANALYSIS
                  </span>
                </button>
              ))}
            </div>

            <label className="block border-2 border-dashed border-white/20 rounded-2xl p-12 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-500/5 transition-all group/upload">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 opacity-0 group-hover/upload:opacity-20 blur-2xl transition-opacity" />
                <Upload
                  className="relative mx-auto mb-4 text-white/50 group-hover/upload:text-blue-400 transition-colors"
                  size={48}
                />
              </div>
              <p className="text-lg text-white/70 mb-2">
                Drop media file or click to browse
              </p>
              <p className="text-sm text-white/50">
                Supported: MP4, AVI, MOV, WAV, MP3
              </p>
              <input
                type="file"
                className="hidden"
                accept={fileType === "video" ? "video/*" : "audio/*"}
                onChange={(e) => setFile(e.target.files[0])}
              />
            </label>

            {file && (
              <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-xl">
                <p className="text-sm text-blue-300 flex items-center gap-2">
                  <Database size={16} />
                  <span className="font-semibold">File Loaded:</span>{" "}
                  {file.name}
                </p>
              </div>
            )}

            {error && (
              <div className="mt-6 p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl">
                <p className="text-rose-300 flex items-center gap-2">
                  <XCircle size={18} />
                  <span className="font-semibold">{error}</span>
                </p>
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className="mt-8 w-full py-4 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 hover:from-blue-500 hover:via-purple-500 hover:to-pink-500 rounded-xl flex justify-center items-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed font-bold text-lg shadow-[0_0_30px_rgba(59,130,246,0.5)] transition-all"
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin" size={24} />
                  ANALYZING MEDIA...
                </>
              ) : (
                <>
                  <ShieldAlert size={24} />
                  INITIATE FORENSIC SCAN
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results */}
        {results && (
          <div className="space-y-8">
            {/* Verdict Banner */}
            <div className="relative group">
              <div
                className={`absolute -inset-1 rounded-3xl blur-xl transition-opacity ${
                  verdict === "SAFE"
                    ? "bg-emerald-500/50"
                    : verdict === "SUSPICIOUS"
                    ? "bg-amber-500/50"
                    : "bg-rose-500/50"
                }`}
              />
              <div className="relative bg-black/80 backdrop-blur-xl border-2 border-white/20 rounded-3xl p-8">
                <div className="text-center mb-6">
                  <p className="text-sm text-white/50 uppercase tracking-widest mb-3">
                    Final Verdict
                  </p>
                  <div className="flex justify-center mb-4">
                    <Badge level={verdict} />
                  </div>
                </div>

                <div className="grid md:grid-cols-3 gap-6">
                  <div className="md:col-span-1 text-center">
                    <p className="text-6xl font-black mb-2 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                      {score}%
                    </p>
                    <p className="text-sm text-white/60 uppercase tracking-wider">
                      Authenticity Score
                    </p>
                  </div>

                  <div className="md:col-span-2">
                    <p className="text-xs text-white/50 uppercase tracking-wider mb-3">
                      Threat Assessment
                    </p>
                    <ThreatMeter score={score} />
                    <div className="mt-6 space-y-2 text-sm text-white/70">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-blue-400" />
                        <span>Multi-modal Neural Analysis Complete</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-purple-400" />
                        <span>Cross-verification Protocol Executed</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-pink-400" />
                        <span>Forensic Confidence Level: High</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {results.video_analysis && (
                <>
                  <MetricCard
                    icon={<FileVideo size={20} />}
                    label="Video Prediction"
                    value={results.video_analysis.prediction}
                    color="blue"
                  />
                  <MetricCard
                    icon={<Activity size={20} />}
                    label="Confidence"
                    value={`${results.video_analysis.confidence}%`}
                    color="purple"
                  />
                </>
              )}
              {results.audio_analysis && (
                <>
                  <MetricCard
                    icon={<FileAudio size={20} />}
                    label="Audio Prediction"
                    value={results.audio_analysis.prediction}
                    color="green"
                  />
                  <MetricCard
                    icon={<Zap size={20} />}
                    label="Confidence"
                    value={`${results.audio_analysis.confidence}%`}
                    color="purple"
                  />
                </>
              )}
            </div>

            {/* Analysis Sections */}
            <div className="grid md:grid-cols-2 gap-6">
              {results.video_analysis && (
                <AnalysisSection
                  title="Video Forensics"
                  icon={<FileVideo size={24} />}
                  gradient="from-blue-500/30 to-cyan-500/30"
                  data={{
                    prediction: results.video_analysis.prediction,
                    confidence: `${results.video_analysis.confidence}%`,
                    frames_analyzed: results.video_analysis.frames_analyzed,
                  }}
                />
              )}

              {results.audio_analysis && (
                <AnalysisSection
                  title="Audio Forensics"
                  icon={<FileAudio size={24} />}
                  gradient="from-emerald-500/30 to-teal-500/30"
                  data={{
                    prediction: results.audio_analysis.prediction,
                    confidence: `${results.audio_analysis.confidence}%`,
                  }}
                />
              )}

              {results.metadata_analysis && (
                <AnalysisSection
                  title="Metadata Analysis"
                  icon={<Lock size={24} />}
                  gradient="from-amber-500/30 to-orange-500/30"
                  data={{
                    tampering_status: results.metadata_analysis
                      .tampering_detected
                      ? "Detected"
                      : "Clean",
                    indicators:
                      results.metadata_analysis.indicators?.join(", ") ||
                      "None",
                  }}
                />
              )}
            </div>

            {/* Heatmap Video */}
            {results.video_analysis?.heatmap_video_url && (
              <HeatmapVideo video={results.video_analysis.heatmap_video_url} />
            )}

            {/* Disclaimer */}
            <div className="relative group">
              <div className="absolute inset-0 bg-amber-500/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative bg-amber-500/10 border border-amber-500/30 rounded-2xl p-6 flex gap-4">
                <AlertCircle
                  className="text-amber-400 flex-shrink-0"
                  size={24}
                />
                <div>
                  <p className="font-semibold text-amber-300 mb-1">
                    Legal Disclaimer
                  </p>
                  <p className="text-sm text-amber-200/80">
                    This system provides probabilistic forensic analysis based
                    on neural network predictions. Results should be used as
                    investigative guidance, not as definitive legal evidence.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="relative border-t border-white/10 backdrop-blur-xl bg-black/20 mt-20">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex justify-between items-center text-sm text-white/50">
            <div>
              <p className="font-semibold">
                Manthan 1.0 • Deepfake Detection System
              </p>
              <p className="text-xs mt-1">
                MANIT Bhopal • Team NULL BUDDIES • NIT Raipur
              </p>
            </div>
            <div className="text-right">
              <p className="text-xs">Powered by Advanced Neural Networks</p>
              <p className="text-xs mt-1">© 2026 All Rights Reserved</p>
            </div>
          </div>
        </div>
      </footer>

      <style>{`
        @keyframes scan {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
      `}</style>
    </div>
  );
}
