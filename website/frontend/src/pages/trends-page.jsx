import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Activity,
  Gauge,
  Timer,
  Dumbbell,
  User,
  Calendar,
  GitCompare,
  ArrowRight,
  X,
  Download,
  BarChart3,
  Printer,
  AlertTriangle,
  Copy,
  Check,
  Sun,
  Moon,
} from "lucide-react";
import { useEffect, useState, useMemo, useCallback } from "react";
import { Link } from "react-router-dom";

import { fetchTrends, compareSwimmers } from "@/api";
import { SiteHeader } from "@/components/site-header";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const METRIC_CONFIG = {
  stroke_rate: { label: "Stroke Rate", unit: "spm", icon: Activity, color: "#60a5fa" },
  body_roll: { label: "Body Roll", unit: "°", icon: Gauge, color: "#f472b6" },
  symmetry_index: { label: "Symmetry", unit: "%", icon: Dumbbell, color: "#a78bfa" },
  cycle_duration_seconds: { label: "Cycle Duration", unit: "s", icon: Timer, color: "#34d399" },
  left_elbow_flexion: { label: "L Elbow Flexion", unit: "°", icon: Activity, color: "#fbbf24" },
  right_elbow_flexion: { label: "R Elbow Flexion", unit: "°", icon: Activity, color: "#fb923c" },
};

function Sparkline({ values, dates, color, metric, height = 40, width = 120, unit, goalValue }) {
  const [hoverIdx, setHoverIdx] = useState(null);

  const { points, dotCoords, pathLength, gradientId, safeColor, h, padding, goalY } = useMemo(() => {
    if (!values || values.length < 2) return { points: "", pathLength: 0, dotCoords: [] };
    const vals = values.filter(v => v != null);
    if (vals.length < 2) return { points: "", pathLength: 0, dotCoords: [] };

    const c = color || "#60a5fa";
    const gId = `spark-fill-${metric || "default"}-${c.replace("#", "")}`;
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const range = max - min || 1;
    const p = 2;
    const w = width - p * 2;
    const hgt = height - p * 2;

    const dots = vals.map((v, i) => {
      const x = p + (i / (vals.length - 1)) * w;
      const y = p + hgt - ((v - min) / range) * hgt;
      return { x, y, v, idx: i };
    });

    const pts = dots.map(d => `${d.x.toFixed(1)},${d.y.toFixed(1)}`).join(" ");

    let len = 0;
    for (let i = 1; i < dots.length; i++) {
      len += Math.sqrt((dots[i].x - dots[i-1].x) ** 2 + (dots[i].y - dots[i-1].y) ** 2);
    }

    let gY = null;
    if (goalValue != null && !isNaN(goalValue)) {
      gY = p + hgt - ((goalValue - min) / range) * hgt;
    }

    return { points: pts, dotCoords: dots, pathLength: len, gradientId: gId, safeColor: c, h: hgt, padding: p, goalY: gY };
  }, [values, color, metric, width, height, goalValue]);

  const handleKeyDown = useCallback((e) => {
    if (!dotCoords.length) return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      setHoverIdx(prev => {
        if (prev == null) return e.key === "ArrowLeft" ? dotCoords.length - 1 : 0;
        const next = e.key === "ArrowLeft" ? prev - 1 : prev + 1;
        if (next < 0) return dotCoords.length - 1;
        if (next >= dotCoords.length) return 0;
        return next;
      });
    } else if (e.key === "Enter") {
      e.preventDefault();
      setHoverIdx(prev => (prev != null ? null : 0));
    } else if (e.key === "Escape") {
      setHoverIdx(null);
    }
  }, [dotCoords.length]);

  if (!points || !pathLength) return null;

  return (
    <div
      className="relative shrink-0 outline-none focus-visible:ring-2 focus-visible:ring-white/40 focus-visible:rounded"
      style={{ width, height }}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseLeave={() => setHoverIdx(null)}
      aria-label={`Sparkline for ${metric || "metric"}`}
    >
      <svg width={width} height={height} style={{ overflow: "visible" }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={safeColor} stopOpacity="0.25" />
            <stop offset="100%" stopColor={safeColor} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {/* Goal line */}
        {goalY != null && (
          <>
            <line
              x1={0} y1={goalY} x2={width} y2={goalY}
              stroke={safeColor}
              strokeWidth="1"
              strokeDasharray="4 3"
              opacity="0.5"
            />
            <text
              x={width - 2}
              y={goalY - 3}
              textAnchor="end"
              fill={safeColor}
              fontSize="8"
              opacity="0.7"
            >
              {goalValue.toFixed(1)}
            </text>
          </>
        )}
        <polygon
          points={`${padding},${h + padding} ${points} ${width - padding},${h + padding}`}
          fill={`url(#${gradientId})`}
          className="spark-fill-appear"
        />
        <polyline
          points={points}
          fill="none"
          stroke={safeColor}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeDasharray={pathLength}
          strokeDashoffset={pathLength}
          className="spark-line-animate"
        />
        {/* Hover targets */}
        {dotCoords.map((d) => (
          <circle
            key={d.idx}
            cx={d.x}
            cy={d.y}
            r={6}
            fill="transparent"
            stroke="transparent"
            style={{ cursor: "pointer" }}
            onMouseEnter={() => setHoverIdx(d.idx)}
            onClick={() => setHoverIdx(d.idx === hoverIdx ? null : d.idx)}
          />
        ))}
        {/* Active dot indicator */}
        {hoverIdx != null && dotCoords[hoverIdx] && (
          <circle
            cx={dotCoords[hoverIdx].x}
            cy={dotCoords[hoverIdx].y}
            r={3}
            fill={safeColor}
            stroke="#fff"
            strokeWidth="1"
          />
        )}
      </svg>
      {/* Tooltip */}
      {hoverIdx != null && dotCoords[hoverIdx] && (
        <div
          className="absolute z-20 rounded-lg border border-white/10 bg-gray-900/95 backdrop-blur px-2 py-1 text-[10px] text-white shadow-lg pointer-events-none whitespace-nowrap"
          style={{
            left: Math.min(dotCoords[hoverIdx].x, width - 80),
            top: Math.max(0, dotCoords[hoverIdx].y - 32),
          }}
        >
          <p className="font-medium">{dotCoords[hoverIdx].v.toFixed(1)} {unit || ""}</p>
          {dates && dates[hoverIdx] && (
            <p className="text-white/50">{dates[hoverIdx]}</p>
          )}
        </div>
      )}
    </div>
  );
}

function TrendBar({ value, maxValue, color, label, sparkValues, sparkMetric, sparkDates, unit, goalValue }) {
  const pct = maxValue > 0 ? Math.min((value / maxValue) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-2">
      <span className="w-28 text-xs text-white/50 truncate">{label}</span>
      <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <Sparkline values={sparkValues} dates={sparkDates} color={color} metric={sparkMetric} unit={unit} goalValue={goalValue} />
      <span className="w-14 text-xs text-white/70 text-right">{value.toFixed(1)}</span>
    </div>
  );
}

function TrendDirection({ direction, change }) {
  if (direction === "improving") {
    return (
      <Badge className="bg-emerald-500/15 text-emerald-400 border-emerald-500/30 gap-1">
        <TrendingUp className="h-3 w-3" />
        Improving
        {change !== undefined && <span className="ml-1 opacity-70">+{change.toFixed(1)}</span>}
      </Badge>
    );
  }
  if (direction === "declining") {
    return (
      <Badge className="bg-red-500/15 text-red-400 border-red-500/30 gap-1">
        <TrendingDown className="h-3 w-3" />
        Declining
        {change !== undefined && <span className="ml-1 opacity-70">{change.toFixed(1)}</span>}
      </Badge>
    );
  }
  return (
    <Badge className="bg-white/10 text-white/60 border-white/20 gap-1">
      Stable
    </Badge>
  );
}

function ComparisonBar({ valueA, valueB, maxValue, color, label }) {
  const pctA = maxValue > 0 ? Math.min((valueA / maxValue) * 100, 100) : 0;
  const pctB = maxValue > 0 ? Math.min((valueB / maxValue) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-2">
      <span className="w-28 text-xs text-white/50 truncate">{label}</span>
      <div className="flex-1 space-y-1">
        <div className="flex items-center gap-2">
          <span className="w-8 text-[10px] text-white/40 text-right">A</span>
          <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${pctA}%`, backgroundColor: color }}
            />
          </div>
          <span className="w-12 text-[10px] text-white/70 text-right">{valueA?.toFixed(1) || "—"}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-8 text-[10px] text-white/40 text-right">B</span>
          <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${pctB}%`, backgroundColor: "#fb923c" }}
            />
          </div>
          <span className="w-12 text-[10px] text-white/70 text-right">{valueB?.toFixed(1) || "—"}</span>
        </div>
      </div>
    </div>
  );
}

export function TrendsPage() {
  const [trends, setTrends] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [primaryMetric, setPrimaryMetric] = useState("stroke_rate");
  const [swimmerId, setSwimmerId] = useState("");
  const [analysisMode, setAnalysisMode] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [aggregation, setAggregation] = useState("");
  const [outlierMethod, setOutlierMethod] = useState("2sigma");
  const [copied, setCopied] = useState(false);
  const [goalValue, setGoalValue] = useState("");

  // Dark/light mode
  const [theme, setTheme] = useState(() => {
    return typeof window !== "undefined" ? localStorage.getItem("swimvision-theme") || "dark" : "dark";
  });

  useEffect(() => {
    localStorage.setItem("swimvision-theme", theme);
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const [compareMode, setCompareMode] = useState(false);
  const [swimmerA, setSwimmerA] = useState("");
  const [swimmerB, setSwimmerB] = useState("");
  const [comparison, setComparison] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const data = await fetchTrends(primaryMetric, swimmerId, analysisMode, startDate, endDate, aggregation);
        if (cancelled) return;
        setTrends(data);
        setError("");
      } catch (err) {
        if (cancelled) return;
        setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [primaryMetric, swimmerId, analysisMode, startDate, endDate, aggregation]);

  async function handleCompare() {
    if (!swimmerA || !swimmerB) {
      setCompareError("Select two swimmers to compare.");
      return;
    }
    if (swimmerA === swimmerB) {
      setCompareError("Select two different swimmers to compare.");
      return;
    }
    setCompareLoading(true);
    setCompareError("");
    try {
      const data = await compareSwimmers(swimmerA, swimmerB, primaryMetric);
      setComparison(data);
    } catch (err) {
      setCompareError(err.message);
    } finally {
      setCompareLoading(false);
    }
  }

  function handleExportCSV() {
    if (!sessions.length) return;

    const metricKeys = Object.keys(metricTrends);
    const headers = ["Session", "Date", "Mode", "Severity", ...metricKeys];
    const rows = sessions.map((s) => {
      const row = [
        s.session_id,
        s.date || "",
        s.analysis_mode,
        s.overall_severity,
      ];
      metricKeys.forEach((k) => {
        row.push(s.metrics?.[k] != null ? s.metrics[k] : "");
      });
      return row;
    });

    const csvContent = [
      headers.join(","),
      ...rows.map((r) => r.map((v) => (typeof v === "string" && v.includes(",") ? `"${v}"` : v)).join(",")),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `swimvision-trends-${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function handlePrintPDF() {
    window.print();
  }

  const handleCopyReport = useCallback(async () => {
    if (!summary) return;
    const cfg = METRIC_CONFIG[primaryMetric] || { label: primaryMetric, unit: "" };
    const lines = [
      "SwimVision — Performance Trends Report",
      "========================================",
      "",
      `Primary Metric: ${cfg.label} (${cfg.unit})`,
      `Sessions: ${summary.num_sessions}`,
      `Date Range: ${summary.date_range?.first || "—"} — ${summary.date_range?.last || "—"}`,
      `Overall Severity: ${summary.overall_worst_severity}`,
      `Metrics with Trends: ${summary.metrics_with_trends}`,
      `Outliers (${outlierMethod === "2sigma" ? "2σ" : "IQR"}): ${outlierInfo.count}`,
      "",
      "Primary Trend:",
      `  Mean: ${summary.primary_trend?.mean?.toFixed(1) || "—"}`,
      `  Range: ${summary.primary_trend?.min?.toFixed(1) || "—"} – ${summary.primary_trend?.max?.toFixed(1) || "—"}`,
      `  Direction: ${summary.primary_trend?.direction || "—"}`,
      `  Slope: ${summary.primary_trend?.slope?.toFixed(3) || "—"}`,
      "",
      summary.summary_verdict || "",
      "",
      "Metric Breakdown:",
    ];
    Object.entries(metricTrends).forEach(([metric, trend]) => {
      const mCfg = METRIC_CONFIG[metric] || { label: metric, unit: "" };
      lines.push(`  ${mCfg.label}: mean=${trend.mean?.toFixed(1) || "—"} ${mCfg.unit}`);
    });
    lines.push("", `Generated: ${new Date().toISOString().slice(0, 10)}`);

    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = lines.join("\n");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      try { document.execCommand("copy"); } catch { /* silent */ }
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [summary, primaryMetric, metricTrends, outlierInfo, outlierMethod]);

  const summary = trends?.trend_summary;
  const sessions = trends?.sessions || [];
  const metricTrends = trends?.metric_trends || {};
  const availableSwimmers = trends?.available_swimmer_ids || [];
  const availableModes = trends?.available_analysis_modes || [];

  const sparklineData = useMemo(() => {
    const data = {};
    Object.keys(metricTrends).forEach((metric) => {
      data[metric] = sessions.map((s) => s.metrics?.[metric]).filter((v) => v != null);
    });
    return data;
  }, [sessions, metricTrends]);

  const sparklineDates = useMemo(() => {
    return sessions.map((s) => s.date || "");
  }, [sessions]);

  // Parse goal value
  const parsedGoal = useMemo(() => {
    const v = parseFloat(goalValue);
    return isNaN(v) ? null : v;
  }, [goalValue]);

  const outlierInfo = useMemo(() => {
    const vals = sessions.map((s) => s.metrics?.[primaryMetric]).filter((v) => v != null);
    if (vals.length < 3) {
      return { outlierIds: new Set(), count: 0, bounds: null, method: outlierMethod };
    }

    let lower, upper;
    const sorted = [...vals].sort((a, b) => a - b);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;

    if (outlierMethod === "iqr") {
      const q1Idx = Math.floor(sorted.length * 0.25);
      const q3Idx = Math.floor(sorted.length * 0.75);
      const q1 = sorted[q1Idx];
      const q3 = sorted[q3Idx];
      const iqr = q3 - q1;
      lower = q1 - 1.5 * iqr;
      upper = q3 + 1.5 * iqr;
    } else {
      const variance = vals.reduce((sum, v) => sum + (v - mean) ** 2, 0) / vals.length;
      const std = Math.sqrt(variance);
      lower = mean - 2 * std;
      upper = mean + 2 * std;
    }

    const outlierIds = new Set();
    sessions.forEach((s, idx) => {
      const v = s.metrics?.[primaryMetric];
      if (v != null && (v < lower || v > upper)) {
        outlierIds.add(s.session_id || idx);
      }
    });

    return {
      outlierIds,
      count: outlierIds.size,
      bounds: { lower, upper, mean },
      method: outlierMethod,
    };
  }, [sessions, primaryMetric, outlierMethod]);

  return (
    <div className="min-h-screen pb-24">
      <style>{`
        @keyframes sparkline-draw {
          to { stroke-dashoffset: 0; }
        }
        .spark-line-animate {
          animation: sparkline-draw 0.8s ease-out forwards;
        }
        @keyframes spark-fill-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .spark-fill-appear {
          animation: spark-fill-in 0.6s ease-out 0.3s forwards;
          opacity: 0;
        }
        /* Dark theme (default) */
        :root, [data-theme="dark"] {
          --bg-color: #0a0a0a;
          --card-bg: rgba(255,255,255,0.03);
          --text-primary: #ffffff;
          --text-secondary: rgba(255,255,255,0.7);
          --text-muted: rgba(255,255,255,0.5);
          --border-color: rgba(255,255,255,0.1);
          --border-light: rgba(255,255,255,0.05);
        }
        [data-theme="light"] {
          --bg-color: #f8f9fa;
          --card-bg: #ffffff;
          --text-primary: #1a1a2e;
          --text-secondary: #4a4a6a;
          --text-muted: #6b7280;
          --border-color: #e5e7eb;
          --border-light: #f0f0f0;
        }
        body {
          background-color: var(--bg-color);
          color: var(--text-primary);
          transition: background-color 0.3s, color 0.3s;
        }
        .theme-card {
          background-color: var(--card-bg);
          border-color: var(--border-color);
        }
        .theme-text-primary { color: var(--text-primary); }
        .theme-text-secondary { color: var(--text-secondary); }
        .theme-text-muted { color: var(--text-muted); }
        .theme-border { border-color: var(--border-color); }
        [data-theme="light"] .glass-line {
          background: rgba(255,255,255,0.8) !important;
          border-color: #e5e7eb !important;
        }
        [data-theme="light"] .backdrop-blur-xl {
          backdrop-filter: blur(12px);
        }
        [data-theme="light"] .hero-veil {
          opacity: 0.15 !important;
        }
        [data-theme="light"] .bg-white\\/10,
        [data-theme="light"] .bg-white\\/\\[0\\.03\\],
        [data-theme="light"] .bg-white\\/\\[0\\.04\\],
        [data-theme="light"] .bg-white\\/\\[0\\.05\\],
        [data-theme="light"] .bg-white\\/\\[0\\.06\\] {
          background: #f0f0f0 !important;
        }
        [data-theme="light"] .border-white\\/10,
        [data-theme="light"] .border-white\\/20,
        [data-theme="light"] .border-white\\/30 {
          border-color: #e5e7eb !important;
        }
        [data-theme="light"] .text-white { color: #1a1a2e !important; }
        [data-theme="light"] .text-white\\/40,
        [data-theme="light"] .text-white\\/50,
        [data-theme="light"] .text-white\\/55,
        [data-theme="light"] .text-white\\/58,
        [data-theme="light"] .text-white\\/60,
        [data-theme="light"] .text-white\\/72,
        [data-theme="light"] .text-white\\/70,
        [data-theme="light"] .text-white\\/80 { color: #4a4a6a !important; }
        [data-theme="light"] .bg-black\\/35 { background: rgba(255,255,255,0.8) !important; }
        [data-theme="light"] select option { background: #fff; color: #1a1a2e; }
        [data-theme="light"] input[type="date"] { color-scheme: light; }
        [data-theme="light"] input { color: #1a1a2e; }
        [data-theme="light"] ::placeholder { color: #9ca3af; }
        @media print {
          body * { visibility: hidden; }
          .print-area, .print-area * { visibility: visible; }
          .print-area { position: absolute; left: 0; top: 0; width: 100%; }
          .no-print { display: none !important; }
          .print-area .container { max-width: 100% !important; padding: 1cm !important; }
          .print-area table { font-size: 10px !important; }
          .print-area .text-white\\/40,
          .print-area .text-white\\/50,
          .print-area .text-white\\/55,
          .print-area .text-white\\/60,
          .print-area .text-white\\/70 { color: #555 !important; }
          .print-area .text-white { color: #000 !important; }
          .print-area .bg-white\\/10,
          .print-area .bg-white\\/[0\\.03\\],
          .print-area .bg-white\\/[0\\.04\\],
          .print-area .bg-white\\/[0\\.05\\] { background: #f5f5f5 !important; }
          .print-area .border-white\\/10,
          .print-area .border-white\\/20 { border-color: #ddd !important; }
          .print-area .bg-black\\/35 { background: transparent !important; }
          .print-area .backdrop-blur-xl { backdrop-filter: none !important; }
          .print-area svg { max-width: 100%; }
          .print-area .flex-wrap { flex-wrap: wrap !important; }
        }
      `}</style>

      <div className="no-print">
        <SiteHeader />
      </div>
      <main className="container pt-12 print-area">
        <Link to="/" className="mb-6 inline-flex items-center gap-2 text-sm text-white/55 transition hover:text-white no-print">
          <ArrowLeft className="h-4 w-4" />
          Back to home
        </Link>

        <div className="mb-8">
          <h1 className="text-4xl font-semibold tracking-tight text-white">Performance Trends</h1>
          <p className="mt-2 text-white/50">Longitudinal analysis across completed swim analysis sessions.</p>
        </div>

        {/* Filter Bar */}
        <Card className="mb-6 no-print">
          <CardContent className="flex flex-wrap items-center gap-3 p-4">
            <User className="h-4 w-4 text-white/50" />
            <label className="text-xs text-white/50">Swimmer:</label>
            <select
              value={swimmerId}
              onChange={(e) => setSwimmerId(e.target.value)}
              className="bg-transparent text-white text-sm border border-white/20 rounded px-2 py-1 min-w-[130px]"
            >
              <option value="" className="bg-gray-900">All swimmers</option>
              {availableSwimmers.map((sid) => (
                <option key={sid} value={sid} className="bg-gray-900">{sid}</option>
              ))}
            </select>

            <span className="w-px h-5 bg-white/10 mx-1" />
            <label className="text-xs text-white/50">Mode:</label>
            <select
              value={analysisMode}
              onChange={(e) => setAnalysisMode(e.target.value)}
              className="bg-transparent text-white text-sm border border-white/20 rounded px-2 py-1 min-w-[100px]"
            >
              <option value="" className="bg-gray-900">All modes</option>
              {availableModes.map((mode) => (
                <option key={mode} value={mode} className="bg-gray-900 capitalize">{mode}</option>
              ))}
            </select>

            <span className="w-px h-5 bg-white/10 mx-1" />
            <BarChart3 className="h-4 w-4 text-white/50" />
            <label className="text-xs text-white/50">Group:</label>
            <select
              value={aggregation}
              onChange={(e) => setAggregation(e.target.value)}
              className="bg-transparent text-white text-sm border border-white/20 rounded px-2 py-1 min-w-[100px]"
            >
              <option value="" className="bg-gray-900">Per session</option>
              <option value="week" className="bg-gray-900">By week</option>
              <option value="month" className="bg-gray-900">By month</option>
            </select>

            <span className="w-px h-5 bg-white/10 mx-1" />
            <Calendar className="h-4 w-4 text-white/50" />
            <label className="text-xs text-white/50">From:</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="bg-transparent text-white text-sm border border-white/20 rounded px-2 py-1 w-[140px] [color-scheme:dark]"
            />
            <label className="text-xs text-white/50">To:</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="bg-transparent text-white text-sm border border-white/20 rounded px-2 py-1 w-[140px] [color-scheme:dark]"
            />
            {(startDate || endDate) && (
              <button
                onClick={() => { setStartDate(""); setEndDate(""); }}
                className="text-xs text-white/40 hover:text-white/70 transition"
              >
                <X className="h-3 w-3" />
              </button>
            )}

            {/* Goal value input */}
            <span className="w-px h-5 bg-white/10 mx-1" />
            <label className="text-xs text-white/50">Goal:</label>
            <input
              type="number"
              step="any"
              value={goalValue}
              onChange={(e) => setGoalValue(e.target.value)}
              placeholder={METRIC_CONFIG[primaryMetric]?.unit || ""}
              className="bg-transparent text-white text-sm border border-white/20 rounded px-2 py-1 w-[80px] placeholder:text-white/20"
            />
            {goalValue && (
              <button
                onClick={() => setGoalValue("")}
                className="text-xs text-white/40 hover:text-white/70 transition"
              >
                <X className="h-3 w-3" />
              </button>
            )}

            <span className="w-px h-5 bg-white/10 mx-1" />
            <button
              onClick={handleExportCSV}
              disabled={!sessions.length}
              className="inline-flex items-center gap-1.5 text-xs rounded-full px-3 py-1 transition text-white/50 hover:text-white border border-white/10 hover:border-white/20 disabled:opacity-30"
            >
              <Download className="h-3.5 w-3.5" />
              CSV
            </button>

            <button
              onClick={handlePrintPDF}
              disabled={!summary}
              className="inline-flex items-center gap-1.5 text-xs rounded-full px-3 py-1 transition text-white/50 hover:text-white border border-white/10 hover:border-white/20 disabled:opacity-30"
            >
              <Printer className="h-3.5 w-3.5" />
              PDF
            </button>

            <button
              onClick={handleCopyReport}
              disabled={!summary}
              className={`inline-flex items-center gap-1.5 text-xs rounded-full px-3 py-1 transition border disabled:opacity-30 ${
                copied
                  ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
                  : "text-white/50 hover:text-white border-white/10 hover:border-white/20"
              }`}
            >
              {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
              {copied ? "Copied!" : "Share"}
            </button>

            <button
              onClick={() => { setCompareMode(!compareMode); if (compareMode) setComparison(null); }}
              className={`inline-flex items-center gap-1.5 text-xs rounded-full px-3 py-1 transition ${
                compareMode
                  ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                  : "text-white/50 hover:text-white border border-white/10 hover:border-white/20"
              }`}
            >
              <GitCompare className="h-3.5 w-3.5" />
              Compare
            </button>

            {/* Theme toggle */}
            <span className="w-px h-5 bg-white/10 mx-1" />
            <button
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="inline-flex items-center gap-1.5 text-xs rounded-full px-3 py-1 transition text-white/50 hover:text-white border border-white/10 hover:border-white/20"
              title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            >
              {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
              {theme === "dark" ? "Light" : "Dark"}
            </button>

            {(swimmerId || analysisMode || startDate || endDate || aggregation) && (
              <span className="text-xs text-white/40 ml-auto">
                {summary?.num_sessions || 0} {aggregation ? aggregation : "session"}{(summary?.num_sessions || 0) !== 1 ? "s" : ""}
              </span>
            )}
          </CardContent>
        </Card>

        {/* Comparison Panel */}
        {compareMode && (
          <Card className="mb-6 border-blue-500/20 no-print">
            <CardContent className="p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <GitCompare className="h-4 w-4 text-blue-400" />
                  <h2 className="text-lg font-semibold text-white">Swimmer Comparison</h2>
                </div>
                <button
                  onClick={() => { setCompareMode(false); setComparison(null); }}
                  className="text-white/40 hover:text-white/70 transition"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <label className="text-xs text-white/50">Swimmer A:</label>
                <select
                  value={swimmerA}
                  onChange={(e) => setSwimmerA(e.target.value)}
                  className="bg-transparent text-white text-sm border border-white/20 rounded px-2 py-1 min-w-[130px]"
                >
                  <option value="" className="bg-gray-900">Select swimmer</option>
                  {availableSwimmers.map((sid) => (
                    <option key={sid} value={sid} className="bg-gray-900">{sid}</option>
                  ))}
                </select>

                <ArrowRight className="h-4 w-4 text-white/30" />

                <label className="text-xs text-white/50">Swimmer B:</label>
                <select
                  value={swimmerB}
                  onChange={(e) => setSwimmerB(e.target.value)}
                  className="bg-transparent text-white text-sm border border-white/20 rounded px-2 py-1 min-w-[130px]"
                >
                  <option value="" className="bg-gray-900">Select swimmer</option>
                  {availableSwimmers.map((sid) => (
                    <option key={sid} value={sid} className="bg-gray-900">{sid}</option>
                  ))}
                </select>

                <button
                  onClick={handleCompare}
                  disabled={compareLoading || !swimmerA || !swimmerB}
                  className="inline-flex items-center gap-1.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30 px-4 py-1.5 text-xs transition hover:bg-blue-500/30 disabled:opacity-40"
                >
                  {compareLoading ? "Comparing..." : "Run comparison"}
                </button>
              </div>

              {compareError && (
                <p className="text-sm text-red-400">{compareError}</p>
              )}

              {comparison && (
                <div className="space-y-4 pt-2 border-t border-white/10">
                  {comparison.comparison && (
                    <div className="grid gap-3 md:grid-cols-3">
                      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
                        <p className="text-xs text-white/40 mb-1">{comparison.comparison.swimmer_a_name}</p>
                        <p className="text-xl text-white font-semibold">
                          {comparison.comparison.swimmer_a_mean?.toFixed(1) || "—"}
                        </p>
                        <TrendDirection
                          direction={comparison.comparison.swimmer_a_direction}
                          change={comparison.swimmer_a?.trend_summary?.primary_trend?.change}
                        />
                      </div>
                      <div className="rounded-xl border border-blue-500/20 bg-blue-500/[0.04] p-4 text-center">
                        <p className="text-xs text-blue-400/60 mb-1">Difference</p>
                        <p className="text-xl text-blue-400 font-semibold">
                          {(comparison.comparison.diff > 0 ? "+" : "")}{comparison.comparison.diff?.toFixed(1) || "—"}
                        </p>
                        <p className="text-xs text-white/40">
                          {primaryMetric in METRIC_CONFIG ? METRIC_CONFIG[primaryMetric].unit : ""}
                        </p>
                      </div>
                      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
                        <p className="text-xs text-white/40 mb-1">{comparison.comparison.swimmer_b_name}</p>
                        <p className="text-xl text-white font-semibold">
                          {comparison.comparison.swimmer_b_mean?.toFixed(1) || "—"}
                        </p>
                        <TrendDirection
                          direction={comparison.comparison.swimmer_b_direction}
                          change={comparison.swimmer_b?.trend_summary?.primary_trend?.change}
                        />
                      </div>
                    </div>
                  )}

                  {comparison.swimmer_a?.metric_trends && comparison.swimmer_b?.metric_trends && (
                    <div className="space-y-2 pt-2">
                      <p className="text-xs text-white/40 mb-2">Metric Breakdown</p>
                      {Object.keys(comparison.swimmer_a.metric_trends).map((metric) => {
                        const cfg = METRIC_CONFIG[metric] || { label: metric, unit: "", color: "#60a5fa" };
                        const valA = comparison.swimmer_a.metric_trends[metric]?.mean;
                        const valB = comparison.swimmer_b.metric_trends[metric]?.mean;
                        const maxVal = Math.max(valA || 0, valB || 0, 1);
                        return (
                          <ComparisonBar
                            key={metric}
                            valueA={valA}
                            valueB={valB}
                            maxValue={maxVal}
                            color={cfg.color}
                            label={`${cfg.label} (${cfg.unit})`}
                          />
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {loading && (
          <Card>
            <CardContent className="p-8 text-center text-white/55">
              <Activity className="mx-auto mb-3 h-8 w-8 animate-pulse" />
              Loading trend data...
            </CardContent>
          </Card>
        )}

        {error && (
          <Card>
            <CardContent className="p-8 text-center text-red-400">
              {error}
            </CardContent>
          </Card>
        )}

        {!loading && !error && !summary && (
          <Card>
            <CardContent className="p-8 text-center text-white/55">
              No completed sessions found. Run some analyses first to see trends.
            </CardContent>
          </Card>
        )}

        {summary && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid gap-4 md:grid-cols-5">
              <Card>
                <CardContent className="p-5">
                  <p className="text-xs text-white/40 uppercase tracking-wider">Sessions</p>
                  <p className="mt-1 text-2xl font-semibold text-white">{summary.num_sessions}</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-5">
                  <p className="text-xs text-white/40 uppercase tracking-wider">Worst Severity</p>
                  <p className="mt-1 text-2xl font-semibold text-white">{summary.overall_worst_severity}</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-5">
                  <p className="text-xs text-white/40 uppercase tracking-wider">Metrics Tracked</p>
                  <p className="mt-1 text-2xl font-semibold text-white">{summary.metrics_with_trends}</p>
                </CardContent>
              </Card>
              <Card className={outlierInfo.count > 0 ? "border-amber-500/30" : ""}>
                <CardContent className="p-5">
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-white/40 uppercase tracking-wider">Outliers</p>
                    <select
                      value={outlierMethod}
                      onChange={(e) => setOutlierMethod(e.target.value)}
                      className="bg-transparent text-[10px] text-white/40 border border-white/10 rounded px-1 py-0"
                    >
                      <option value="2sigma" className="bg-gray-900">2σ</option>
                      <option value="iqr" className="bg-gray-900">IQR</option>
                    </select>
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <p className="text-2xl font-semibold text-white">{outlierInfo.count}</p>
                    {outlierInfo.count > 0 && (
                      <AlertTriangle className="h-4 w-4 text-amber-400" />
                    )}
                  </div>
                  {outlierInfo.bounds && (
                    <p className="text-[10px] text-white/30 mt-1">
                      μ={outlierInfo.bounds.mean.toFixed(1)} bounds=[{outlierInfo.bounds.lower.toFixed(1)}, {outlierInfo.bounds.upper.toFixed(1)}]
                    </p>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-5">
                  <p className="text-xs text-white/40 uppercase tracking-wider">Primary Metric</p>
                  <select
                    value={primaryMetric}
                    onChange={(e) => setPrimaryMetric(e.target.value)}
                    className="mt-1 bg-transparent text-white text-sm border border-white/20 rounded px-2 py-1"
                  >
                    {Object.keys(METRIC_CONFIG).map((key) => (
                      <option key={key} value={key} className="bg-gray-900">
                        {METRIC_CONFIG[key].label}
                      </option>
                    ))}
                  </select>
                </CardContent>
              </Card>
            </div>

            {/* Primary Trend with Sparkline */}
            {summary.primary_trend && (
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      {(() => {
                        const cfg = METRIC_CONFIG[primaryMetric] || { icon: Activity, color: "#60a5fa" };
                        const Icon = cfg.icon;
                        return <Icon className="h-5 w-5" style={{ color: cfg.color }} />;
                      })()}
                      <h2 className="text-lg font-semibold text-white">
                        {METRIC_CONFIG[primaryMetric]?.label || primaryMetric}
                      </h2>
                      <Sparkline
                        values={sparklineData[primaryMetric]}
                        dates={sparklineDates}
                        color={METRIC_CONFIG[primaryMetric]?.color || "#60a5fa"}
                        metric={primaryMetric}
                        width={160}
                        height={48}
                        unit={METRIC_CONFIG[primaryMetric]?.unit || ""}
                        goalValue={parsedGoal}
                      />
                    </div>
                    <TrendDirection
                      direction={summary.primary_trend.direction}
                      change={summary.primary_trend.change}
                    />
                  </div>

                  <div className="grid gap-4 md:grid-cols-3 mb-4">
                    <div>
                      <p className="text-xs text-white/40">Mean</p>
                      <p className="text-xl text-white">{summary.primary_trend.mean?.toFixed(1)}</p>
                    </div>
                    <div>
                      <p className="text-xs text-white/40">Range</p>
                      <p className="text-xl text-white">
                        {summary.primary_trend.min?.toFixed(1)} – {summary.primary_trend.max?.toFixed(1)}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-white/40">Slope / Session</p>
                      <p className="text-xl text-white">{summary.primary_trend.slope?.toFixed(3)}</p>
                    </div>
                  </div>

                  {summary.date_range?.first && (
                    <p className="text-xs text-white/40 mb-3">
                      {summary.date_range.first} — {summary.date_range.last}
                    </p>
                  )}

                  <p className="text-sm text-white/60 italic">{summary.summary_verdict}</p>
                </CardContent>
              </Card>
            )}

            {/* All Metric Trends with Sparklines */}
            <Card>
              <CardContent className="p-6">
                <h2 className="text-lg font-semibold text-white mb-4">All Metric Trends</h2>
                <div className="space-y-3">
                  {Object.entries(metricTrends).map(([metric, trend]) => {
                    const cfg = METRIC_CONFIG[metric] || { label: metric, unit: "", color: "#60a5fa" };
                    return (
                      <TrendBar
                        key={metric}
                        value={trend.mean || 0}
                        maxValue={Math.max(...Object.values(metricTrends).map((t) => t.mean || 0), 1)}
                        color={cfg.color}
                        label={`${cfg.label} (${cfg.unit})`}
                        sparkValues={sparklineData[metric]}
                        sparkDates={sparklineDates}
                        sparkMetric={metric}
                        unit={cfg.unit}
                        goalValue={metric === primaryMetric ? parsedGoal : null}
                      />
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* Session Table */}
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-white">Session History</h2>
                  <div className="flex gap-2 no-print">
                    {sessions.length > 0 && (
                      <>
                        <button
                          onClick={handleExportCSV}
                          className="inline-flex items-center gap-1.5 text-xs text-white/50 hover:text-white transition"
                        >
                          <Download className="h-3.5 w-3.5" />
                          CSV
                        </button>
                        <button
                          onClick={handlePrintPDF}
                          className="inline-flex items-center gap-1.5 text-xs text-white/50 hover:text-white transition"
                        >
                          <Printer className="h-3.5 w-3.5" />
                          Print
                        </button>
                        <button
                          onClick={handleCopyReport}
                          className={`inline-flex items-center gap-1.5 text-xs transition ${
                            copied ? "text-emerald-400" : "text-white/50 hover:text-white"
                          }`}
                        >
                          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                          {copied ? "Copied!" : "Share"}
                        </button>
                      </>
                    )}
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/10 text-white/50">
                        <th className="py-2 text-left font-medium">Session</th>
                        <th className="py-2 text-left font-medium">Date</th>
                        <th className="py-2 text-left font-medium">Mode</th>
                        <th className="py-2 text-left font-medium">Severity</th>
                        <th className="py-2 text-right font-medium">{METRIC_CONFIG[primaryMetric]?.label || primaryMetric}</th>
                        <th className="py-2 text-left font-medium w-20 no-print"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {sessions.map((session, idx) => {
                        const isOutlier = outlierInfo.outlierIds.has(session.session_id || idx);
                        return (
                          <tr
                            key={idx}
                            className={`border-b border-white/5 hover:bg-white/[0.02] ${isOutlier ? "bg-amber-500/[0.04]" : ""}`}
                          >
                            <td className="py-3 text-white/80">{session.session_id}</td>
                            <td className="py-3 text-white/50">{session.date || "—"}</td>
                            <td className="py-3">
                              <span className="text-white/60">{session.analysis_mode}</span>
                            </td>
                            <td className="py-3">
                              <Badge
                                className={
                                  session.overall_severity === "CRITICAL"
                                    ? "bg-red-500/15 text-red-400 border-red-500/30"
                                    : session.overall_severity === "SIGNIFICANT"
                                    ? "bg-orange-500/15 text-orange-400 border-orange-500/30"
                                    : session.overall_severity === "MINOR"
                                    ? "bg-yellow-500/15 text-yellow-400 border-yellow-500/30"
                                    : "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                                }
                              >
                                {session.overall_severity}
                              </Badge>
                            </td>
                            <td className={`py-3 text-right ${isOutlier ? "text-amber-400 font-medium" : "text-white/70"}`}>
                              {session.metrics?.[primaryMetric]?.toFixed(1) || "—"}
                            </td>
                            <td className="py-3 no-print">
                              {isOutlier && (
                                <Badge className="bg-amber-500/15 text-amber-400 border-amber-500/30 gap-1 ml-2">
                                  <AlertTriangle className="h-3 w-3" />
                                  {outlierMethod === "2sigma" ? "2σ" : "IQR"} outlier
                                </Badge>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                {outlierInfo.count > 0 && (
                  <p className="mt-3 text-xs text-amber-400/60">
                    {outlierInfo.count} session{outlierInfo.count !== 1 ? "s" : ""} flagged as {outlierMethod === "2sigma" ? "2σ" : "IQR"} outliers on {METRIC_CONFIG[primaryMetric]?.label || primaryMetric}
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
}
