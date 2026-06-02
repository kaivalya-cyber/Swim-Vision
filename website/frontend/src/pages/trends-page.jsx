import React, { useEffect, useState, useMemo, useCallback } from "react";
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
  Eye,
  Bell,
  Maximize2,
  GitCommit,
  ChevronRight,
  Layers,
  Trash2,
  Square,
  CheckSquare,
  Wrench,
} from "lucide-react";
import { Link } from "react-router-dom";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

import { fetchTrends, compareSwimmers } from "@/api";
import { SiteHeader } from "@/components/site-header";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Filler, Legend);

const METRIC_CONFIG = {
  stroke_rate: { label: "Stroke Rate", unit: "spm", icon: Activity, color: "#60a5fa", higherIsBetter: true },
  body_roll: { label: "Body Roll", unit: "°", icon: Gauge, color: "#f472b6", higherIsBetter: true },
  symmetry_index: { label: "Symmetry", unit: "%", icon: Dumbbell, color: "#a78bfa", higherIsBetter: false },
  cycle_duration_seconds: { label: "Cycle Duration", unit: "s", icon: Timer, color: "#34d399", higherIsBetter: false },
  left_elbow_flexion: { label: "L Elbow Flexion", unit: "°", icon: Activity, color: "#fbbf24", higherIsBetter: true },
  right_elbow_flexion: { label: "R Elbow Flexion", unit: "°", icon: Activity, color: "#fb923c", higherIsBetter: true },
};

const METRIC_CONFIG_CB = {
  stroke_rate: { label: "Stroke Rate", unit: "spm", icon: Activity, color: "#0072B2", higherIsBetter: true },
  body_roll: { label: "Body Roll", unit: "°", icon: Gauge, color: "#E69F00", higherIsBetter: true },
  symmetry_index: { label: "Symmetry", unit: "%", icon: Dumbbell, color: "#009E73", higherIsBetter: false },
  cycle_duration_seconds: { label: "Cycle Duration", unit: "s", icon: Timer, color: "#CC79A7", higherIsBetter: false },
  left_elbow_flexion: { label: "L Elbow Flexion", unit: "°", icon: Activity, color: "#56B4E9", higherIsBetter: true },
  right_elbow_flexion: { label: "R Elbow Flexion", unit: "°", icon: Activity, color: "#D55E00", higherIsBetter: true },
};

function parsePhaseMetrics(metrics, analysisMode) {
  if (!metrics) return [];
  const phaseMap = {};
  Object.entries(metrics).forEach(([key, value]) => {
    if (analysisMode === "dive") {
      const phaseMatch = key.match(/^(block|flight|entry)_phase_(.+)$/);
      if (phaseMatch) {
        const phase = phaseMatch[1];
        const metricName = phaseMatch[2].replace(/_/g, " ");
        if (!phaseMap[phase]) phaseMap[phase] = [];
        phaseMap[phase].push({ name: metricName, value });
      }
    } else {
      if (!phaseMap["stroke"]) phaseMap["stroke"] = [];
      phaseMap["stroke"].push({ name: key.replace(/_/g, " "), value });
    }
  });
  return Object.entries(phaseMap).map(([phase, items]) => ({
    phase: phase.charAt(0).toUpperCase() + phase.slice(1).replace(/_/g, " "),
    metrics: items,
  }));
}

function Sparkline({ values, dates, color, metric, height = 40, width = 120, unit, goalValue, crosshairIdx, onCrosshairChange, onZoom }) {
  const [localHover, setLocalHover] = useState(null);

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
      gY = Math.max(p, Math.min(p + hgt, gY));
    }

    return { points: pts, dotCoords: dots, pathLength: len, gradientId: gId, safeColor: c, h: hgt, padding: p, goalY: gY };
  }, [values, color, metric, width, height, goalValue]);

  const activeIdx = crosshairIdx != null ? crosshairIdx : localHover;

  const handleKeyDown = useCallback((e) => {
    if (!dotCoords.length) return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      const next = (prev) => {
        if (prev == null) return e.key === "ArrowLeft" ? dotCoords.length - 1 : 0;
        const n = e.key === "ArrowLeft" ? prev - 1 : prev + 1;
        if (n < 0) return dotCoords.length - 1;
        if (n >= dotCoords.length) return 0;
        return n;
      };
      if (onCrosshairChange) {
        onCrosshairChange(prev => (prev == null ? next(0) : next(prev)));
      } else {
        setLocalHover(next);
      }
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (onCrosshairChange) {
        onCrosshairChange(prev => (prev != null ? null : 0));
      } else {
        setLocalHover(prev => (prev != null ? null : 0));
      }
    } else if (e.key === "Escape") {
      if (onCrosshairChange) onCrosshairChange(null);
      setLocalHover(null);
    }
  }, [dotCoords.length, onCrosshairChange]);

  if (!points || !pathLength) return null;

  return (
    <div
      className="relative shrink-0 outline-none focus-visible:ring-2 focus-visible:ring-white/40 focus-visible:rounded"
      style={{ width, height }}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onMouseLeave={() => { setLocalHover(null); if (onCrosshairChange) onCrosshairChange(null); }}
      aria-label={`Sparkline for ${metric || "metric"}`}
    >
      <svg width={width} height={height} style={{ overflow: "visible" }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={safeColor} stopOpacity="0.25" />
            <stop offset="100%" stopColor={safeColor} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {/* Crosshair vertical line */}
        {activeIdx != null && dotCoords[activeIdx] && (
          <line x1={dotCoords[activeIdx].x} y1={0} x2={dotCoords[activeIdx].x} y2={h + padding * 2} stroke="rgba(255,255,255,0.35)" strokeWidth="0.5" strokeDasharray="3 2" pointerEvents="none" />
        )}
        {goalY != null && (
          <>
            <line x1={0} y1={goalY} x2={width} y2={goalY} stroke={safeColor} strokeWidth="1" strokeDasharray="4 3" opacity="0.5" />
            <text x={width - 2} y={goalY - 3} textAnchor="end" fill={safeColor} fontSize="8" opacity="0.7">{goalValue.toFixed(1)}</text>
          </>
        )}
        <polygon points={`${padding},${h + padding} ${points} ${width - padding},${h + padding}`} fill={`url(#${gradientId})`} className="spark-fill-appear" />
        <polyline points={points} fill="none" stroke={safeColor} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" strokeDasharray={pathLength} strokeDashoffset={pathLength} className="spark-line-animate" />
        {dotCoords.map((d) => (
          <circle key={d.idx} cx={d.x} cy={d.y} r={6} fill="transparent" stroke="transparent" style={{ cursor: "pointer" }}
            onMouseEnter={() => onCrosshairChange ? onCrosshairChange(d.idx) : setLocalHover(d.idx)}
            onClick={() => {
              if (onCrosshairChange) onCrosshairChange(d.idx === activeIdx ? null : d.idx);
              else setLocalHover(d.idx === activeIdx ? null : d.idx);
            }} />
        ))}
        {activeIdx != null && dotCoords[activeIdx] && (
          <circle cx={dotCoords[activeIdx].x} cy={dotCoords[activeIdx].y} r={3} fill={safeColor} stroke="#fff" strokeWidth="1" />
        )}
      </svg>
      {activeIdx != null && dotCoords[activeIdx] && (
        <div className="absolute z-20 rounded-lg border border-white/10 bg-gray-900/95 backdrop-blur px-2 py-1 text-[10px] text-white shadow-lg pointer-events-none whitespace-nowrap" style={{ left: Math.min(dotCoords[activeIdx].x, width - 80), top: Math.max(0, dotCoords[activeIdx].y - 32) }}>
          <p className="font-medium">{dotCoords[activeIdx].v.toFixed(1)} {unit || ""}</p>
          {dates && dates[activeIdx] && <p className="text-white/50">{dates[activeIdx]}</p>}
        </div>
      )}
    </div>
  );
}

function TrendBar({ value, maxValue, color, label, sparkValues, sparkMetric, sparkDates, unit, goalValue, crosshairIdx, onCrosshairChange, onZoom }) {
  const pct = maxValue > 0 ? Math.min((value / maxValue) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-1 sm:gap-2 group">
      <span className="w-20 sm:w-28 text-[10px] sm:text-xs text-white/50 truncate">{label}</span>
      <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <Sparkline values={sparkValues} dates={sparkDates} color={color} metric={sparkMetric} unit={unit} goalValue={goalValue} crosshairIdx={crosshairIdx} onCrosshairChange={onCrosshairChange} />
      <span className="w-10 sm:w-14 text-[10px] sm:text-xs text-white/70 text-right">{value.toFixed(1)}</span>
      {onZoom && (
        <button onClick={(e) => { e.stopPropagation(); onZoom(sparkMetric); }} className="opacity-0 group-hover:opacity-100 transition ml-0.5" title="Zoom chart">
          <Maximize2 className="h-3 w-3 text-white/40 hover:text-white/70" />
        </button>
      )}
    </div>
  );
}

function TrendDirection({ direction, change }) {
  if (direction === "improving") {
    return <Badge className="bg-emerald-500/15 text-emerald-400 border-emerald-500/30 gap-1"><TrendingUp className="h-3 w-3" />Improving{change !== undefined && <span className="ml-1 opacity-70">+{change.toFixed(1)}</span>}</Badge>;
  }
  if (direction === "declining") {
    return <Badge className="bg-red-500/15 text-red-400 border-red-500/30 gap-1"><TrendingDown className="h-3 w-3" />Declining{change !== undefined && <span className="ml-1 opacity-70">{change.toFixed(1)}</span>}</Badge>;
  }
  return <Badge className="bg-white/10 text-white/60 border-white/20 gap-1">Stable</Badge>;
}

function ComparisonBar({ valueA, valueB, maxValue, color, label }) {
  const pctA = maxValue > 0 ? Math.min((valueA / maxValue) * 100, 100) : 0;
  const pctB = maxValue > 0 ? Math.min((valueB / maxValue) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-2">
      <span className="w-20 sm:w-28 text-[10px] sm:text-xs text-white/50 truncate">{label}</span>
      <div className="flex-1 space-y-1">
        <div className="flex items-center gap-2"><span className="w-6 sm:w-8 text-[10px] text-white/40 text-right">A</span><div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden"><div className="h-full rounded-full transition-all duration-500" style={{ width: `${pctA}%`, backgroundColor: color }} /></div><span className="w-10 sm:w-12 text-[10px] text-white/70 text-right">{valueA?.toFixed(1) || "—"}</span></div>
        <div className="flex items-center gap-2"><span className="w-6 sm:w-8 text-[10px] text-white/40 text-right">B</span><div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden"><div className="h-full rounded-full transition-all duration-500" style={{ width: `${pctB}%`, backgroundColor: "#fb923c" }} /></div><span className="w-10 sm:w-12 text-[10px] text-white/70 text-right">{valueB?.toFixed(1) || "—"}</span></div>
      </div>
    </div>
  );
}

function computeBaselinePct(curVal, baseVal, higherIsBetter) {
  if (baseVal == null || curVal == null) return null;
  if (Math.abs(baseVal) < 0.001) return { pct: 0, cls: "text-white/40" };
  const diff = ((curVal - baseVal) / Math.abs(baseVal)) * 100;
  const improves = higherIsBetter ? diff > 0 : diff < 0;
  const declines = higherIsBetter ? diff < 0 : diff > 0;
  if (improves) return { pct: diff, cls: "text-emerald-400" };
  if (declines) return { pct: diff, cls: "text-red-400" };
  return { pct: diff, cls: "text-white/40" };
}

function evaluateExpression(expression, metrics) {
  if (!expression || !metrics) return null;
  try {
    const safeExpr = expression.replace(/[a-z_][a-z0-9_]*/gi, (match) => {
      const val = metrics[match];
      if (val != null && typeof val === "number") return String(val);
      if (match === "pi") return String(Math.PI);
      if (match === "e") return String(Math.E);
      throw new Error(`Unknown metric: ${match}`);
    });
    const result = Function(`"use strict"; return (${safeExpr})`)();
    return typeof result === "number" && isFinite(result) ? result : null;
  } catch {
    return null;
  }
}

const CUSTOM_METRIC_COLORS = ["#f472b6", "#a78bfa", "#34d399", "#fbbf24", "#fb923c", "#60a5fa", "#f87171", "#2dd4bf"];

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
  const [thresholdValue, setThresholdValue] = useState("");
  const [expandedSessionIdx, setExpandedSessionIdx] = useState(null);
  const [baselineIdx, setBaselineIdx] = useState(null);
  const [crosshairIdx, setCrosshairIdx] = useState(null);
  const [zoomedMetric, setZoomedMetric] = useState(null);
  const [selectedSessions, setSelectedSessions] = useState(new Set());
  const [showMetricBuilder, setShowMetricBuilder] = useState(false);
  const [customMetrics, setCustomMetrics] = useState(() => {
    try { return typeof window !== "undefined" ? JSON.parse(localStorage.getItem("swimvision-custom-metrics") || "[]") : []; }
    catch { return []; }
  });
  const [newMetricName, setNewMetricName] = useState("");
  const [newMetricExpr, setNewMetricExpr] = useState("");
  const [newMetricUnit, setNewMetricUnit] = useState("");

  const [cbPalette, setCbPalette] = useState(() => {
    return typeof window !== "undefined" ? localStorage.getItem("swimvision-cb-palette") === "true" : false;
  });
  const toggleCb = useCallback(() => {
    setCbPalette(prev => { const n = !prev; localStorage.setItem("swimvision-cb-palette", String(n)); return n; });
  }, []);

  const activeMetricConfig = cbPalette ? METRIC_CONFIG_CB : METRIC_CONFIG;

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
        setExpandedSessionIdx(null);
        setBaselineIdx(null);
        setCrosshairIdx(null);
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
    if (!swimmerA || !swimmerB) { setCompareError("Select two swimmers to compare."); return; }
    if (swimmerA === swimmerB) { setCompareError("Select two different swimmers to compare."); return; }
    setCompareLoading(true); setCompareError("");
    try { const data = await compareSwimmers(swimmerA, swimmerB, primaryMetric); setComparison(data); }
    catch (err) { setCompareError(err.message); }
    finally { setCompareLoading(false); }
  }

  function handleExportCSV() {
    if (!sessions.length) return;
    const metricKeys = Object.keys(metricTrends);
    const headers = ["Session", "Date", "Mode", "Severity", ...metricKeys];
    const rows = sessions.map((s) => {
      const row = [s.session_id, s.date || "", s.analysis_mode, s.overall_severity];
      metricKeys.forEach((k) => { row.push(s.metrics?.[k] != null ? s.metrics[k] : ""); });
      return row;
    });
    const csvContent = [headers.join(","), ...rows.map((r) => r.map((v) => (typeof v === "string" && v.includes(",") ? `"${v}"` : v)).join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a"); link.href = url; link.download = `swimvision-trends-${new Date().toISOString().slice(0, 10)}.csv`; link.click();
    URL.revokeObjectURL(url);
  }

  function handlePrintPDF() { window.print(); }

  const handleCopyReport = useCallback(async () => {
    if (!summary) return;
    const cfg = activeMetricConfig[primaryMetric] || { label: primaryMetric, unit: "" };
    const lines = ["SwimVision — Performance Trends Report", "========================================", "", `Primary Metric: ${cfg.label} (${cfg.unit})`, `Sessions: ${summary.num_sessions}`, `Date Range: ${summary.date_range?.first || "—"} — ${summary.date_range?.last || "—"}`, `Overall Severity: ${summary.overall_worst_severity}`, `Metrics with Trends: ${summary.metrics_with_trends}`, `Outliers (${outlierMethod === "2sigma" ? "2σ" : "IQR"}): ${outlierInfo.count}`, "", "Primary Trend:", `  Mean: ${summary.primary_trend?.mean?.toFixed(1) || "—"}`, `  Range: ${summary.primary_trend?.min?.toFixed(1) || "—"} – ${summary.primary_trend?.max?.toFixed(1) || "—"}`, `  Direction: ${summary.primary_trend?.direction || "—"}`, `  Slope: ${summary.primary_trend?.slope?.toFixed(3) || "—"}`, "", summary.summary_verdict || "", "", "Metric Breakdown:"];
    Object.entries(metricTrends).forEach(([metric, trend]) => { const mCfg = activeMetricConfig[metric] || { label: metric, unit: "" }; lines.push(`  ${mCfg.label}: mean=${trend.mean?.toFixed(1) || "—"} ${mCfg.unit}`); });
    lines.push("", `Generated: ${new Date().toISOString().slice(0, 10)}`);
    try { await navigator.clipboard.writeText(lines.join("\n")); setCopied(true); setTimeout(() => setCopied(false), 2000); }
    catch { const textarea = document.createElement("textarea"); textarea.value = lines.join("\n"); textarea.style.position = "fixed"; textarea.style.opacity = "0"; document.body.appendChild(textarea); textarea.select(); try { document.execCommand("copy"); } catch {} document.body.removeChild(textarea); setCopied(true); setTimeout(() => setCopied(false), 2000); }
  }, [summary, primaryMetric, metricTrends, outlierInfo, outlierMethod, activeMetricConfig]);

  const summary = trends?.trend_summary;
  const sessions = trends?.sessions || [];
  const metricTrends = trends?.metric_trends || {};
  const availableSwimmers = trends?.available_swimmer_ids || [];
  const availableModes = trends?.available_analysis_modes || [];
  const primaryCfg = activeMetricConfig[primaryMetric] || { label: primaryMetric, unit: "", color: "#60a5fa", higherIsBetter: true };

  const sparklineData = useMemo(() => {
    const data = {};
    Object.keys(metricTrends).forEach((metric) => { data[metric] = sessions.map((s) => s.metrics?.[metric]).filter((v) => v != null); });
    return data;
  }, [sessions, metricTrends]);

  const sparklineDates = useMemo(() => sessions.map((s) => s.date || ""), [sessions]);

  const parsedGoal = useMemo(() => { const v = parseFloat(goalValue); return isNaN(v) ? null : v; }, [goalValue]);
  const parsedThreshold = useMemo(() => { const v = parseFloat(thresholdValue); return isNaN(v) ? null : v; }, [thresholdValue]);

  const thresholdInfo = useMemo(() => {
    if (parsedThreshold == null) return { alertIds: new Set(), count: 0 };
    const alertIds = new Set();
    sessions.forEach((s, idx) => {
      const v = s.metrics?.[primaryMetric];
      if (v != null && v < parsedThreshold) alertIds.add(s.session_id || idx);
    });
    return { alertIds, count: alertIds.size };
  }, [sessions, primaryMetric, parsedThreshold]);

  const outlierInfo = useMemo(() => {
    const vals = sessions.map((s) => s.metrics?.[primaryMetric]).filter((v) => v != null);
    if (vals.length < 3) return { outlierIds: new Set(), count: 0, bounds: null, method: outlierMethod };
    let lower, upper;
    const sorted = [...vals].sort((a, b) => a - b);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    if (outlierMethod === "iqr") {
      const q1 = sorted[Math.floor(sorted.length * 0.25)];
      const q3 = sorted[Math.floor(sorted.length * 0.75)];
      const iqr = q3 - q1;
      lower = q1 - 1.5 * iqr;
      upper = q3 + 1.5 * iqr;
    } else {
      const std = Math.sqrt(vals.reduce((sum, v) => sum + (v - mean) ** 2, 0) / vals.length);
      lower = mean - 2 * std;
      upper = mean + 2 * std;
    }
    const outlierIds = new Set();
    sessions.forEach((s, idx) => { const v = s.metrics?.[primaryMetric]; if (v != null && (v < lower || v > upper)) outlierIds.add(s.session_id || idx); });
    return { outlierIds, count: outlierIds.size, bounds: { lower, upper, mean }, method: outlierMethod };
  }, [sessions, primaryMetric, outlierMethod]);

  // Zoom modal data
  const zoomData = useMemo(() => {
    if (!zoomedMetric) return null;
    const allCfg = { ...activeMetricConfig };
    customMetrics.forEach((cm) => { allCfg[cm.name] = { label: cm.name, unit: cm.unit || "", color: cm.color || "#60a5fa", higherIsBetter: true }; });
    const cfg = allCfg[zoomedMetric] || { label: zoomedMetric, unit: "", color: "#60a5fa" };
    const vals = allSparklineData[zoomedMetric];
    const dates = sparklineDates;
    if (!vals || vals.length < 2) return null;
    return { metric: zoomedMetric, cfg, vals, dates, color: cfg.color };
  }, [zoomedMetric, activeMetricConfig, customMetrics, allSparklineData, sparklineDates]);

  // Custom metrics values per session
  const customMetricValues = useMemo(() => {
    if (!customMetrics.length) return {};
    const result = {};
    customMetrics.forEach((cm) => {
      result[cm.name] = sessions.map((s) => evaluateExpression(cm.expression, s.metrics));
    });
    return result;
  }, [customMetrics, sessions]);

  // Merge custom metrics into sparklineData
  const allSparklineData = useMemo(() => {
    return { ...sparklineData, ...customMetricValues };
  }, [sparklineData, customMetricValues]);

  function toggleSelectSession(idx) {
    setSelectedSessions((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }

  function toggleSelectAll() {
    if (selectedSessions.size === sessions.length) {
      setSelectedSessions(new Set());
    } else {
      setSelectedSessions(new Set(sessions.map((_, i) => i)));
    }
  }

  function handleExportSelectedCSV() {
    if (!selectedSessions.size) return;
    const metricKeys = Object.keys(metricTrends);
    const selSessions = [...selectedSessions].sort((a, b) => a - b).map((i) => sessions[i]);
    const headers = ["Session", "Date", "Mode", "Severity", ...metricKeys];
    const rows = selSessions.map((s) => {
      const row = [s.session_id, s.date || "", s.analysis_mode, s.overall_severity];
      metricKeys.forEach((k) => { row.push(s.metrics?.[k] != null ? s.metrics[k] : ""); });
      return row;
    });
    const csvContent = [headers.join(","), ...rows.map((r) => r.map((v) => (typeof v === "string" && v.includes(",") ? `"${v}"` : v)).join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a"); link.href = url; link.download = `swimvision-selected-${new Date().toISOString().slice(0, 10)}.csv`; link.click();
    URL.revokeObjectURL(url);
  }

  function handleAddCustomMetric() {
    const name = newMetricName.trim();
    const expr = newMetricExpr.trim();
    if (!name || !expr) return;
    if (customMetrics.find((m) => m.name === name)) return;
    const testVal = evaluateExpression(expr, sessions[0]?.metrics || {});
    if (testVal == null) return;
    const newCm = {
      name,
      expression: expr,
      unit: newMetricUnit.trim(),
      color: CUSTOM_METRIC_COLORS[customMetrics.length % CUSTOM_METRIC_COLORS.length],
    };
    const updated = [...customMetrics, newCm];
    setCustomMetrics(updated);
    localStorage.setItem("swimvision-custom-metrics", JSON.stringify(updated));
    setNewMetricName("");
    setNewMetricExpr("");
    setNewMetricUnit("");
  }

  function handleRemoveCustomMetric(name) {
    const updated = customMetrics.filter((m) => m.name !== name);
    setCustomMetrics(updated);
    localStorage.setItem("swimvision-custom-metrics", JSON.stringify(updated));
  }

  return (
    <div className="min-h-screen pb-16 sm:pb-24">
      <style>{`
        @keyframes sparkline-draw { to { stroke-dashoffset: 0; } }
        .spark-line-animate { animation: sparkline-draw 0.8s ease-out forwards; }
        @keyframes spark-fill-in { from { opacity: 0; } to { opacity: 1; } }
        .spark-fill-appear { animation: spark-fill-in 0.6s ease-out 0.3s forwards; opacity: 0; }
        @keyframes slide-down { from { opacity: 0; max-height: 0; } to { opacity: 1; max-height: 600px; } }
        .expand-panel { animation: slide-down 0.3s ease-out forwards; overflow: hidden; }
        :root, [data-theme="dark"] { --bg-color: #0a0a0a; --card-bg: rgba(255,255,255,0.03); --text-primary: #ffffff; --text-secondary: rgba(255,255,255,0.7); --text-muted: rgba(255,255,255,0.5); --border-color: rgba(255,255,255,0.1); --border-light: rgba(255,255,255,0.05); }
        [data-theme="light"] { --bg-color: #f8f9fa; --card-bg: #ffffff; --text-primary: #1a1a2e; --text-secondary: #4a4a6a; --text-muted: #6b7280; --border-color: #e5e7eb; --border-light: #f0f0f0; }
        body { background-color: var(--bg-color); color: var(--text-primary); transition: background-color 0.3s, color 0.3s; }
        [data-theme="light"] .glass-line { background: rgba(255,255,255,0.8) !important; border-color: #e5e7eb !important; }
        [data-theme="light"] .backdrop-blur-xl { backdrop-filter: blur(12px); }
        [data-theme="light"] .hero-veil { opacity: 0.15 !important; }
        [data-theme="light"] .bg-white\\/10, [data-theme="light"] .bg-white\\/\\[0\\.03\\], [data-theme="light"] .bg-white\\/\\[0\\.04\\], [data-theme="light"] .bg-white\\/\\[0\\.05\\], [data-theme="light"] .bg-white\\/\\[0\\.06\\] { background: #f0f0f0 !important; }
        [data-theme="light"] .border-white\\/10, [data-theme="light"] .border-white\\/20, [data-theme="light"] .border-white\\/30 { border-color: #e5e7eb !important; }
        [data-theme="light"] .text-white { color: #1a1a2e !important; }
        [data-theme="light"] .text-white\\/40, [data-theme="light"] .text-white\\/50, [data-theme="light"] .text-white\\/55, [data-theme="light"] .text-white\\/58, [data-theme="light"] .text-white\\/60, [data-theme="light"] .text-white\\/72, [data-theme="light"] .text-white\\/70, [data-theme="light"] .text-white\\/80 { color: #4a4a6a !important; }
        [data-theme="light"] .bg-black\\/35 { background: rgba(255,255,255,0.8) !important; }
        [data-theme="light"] select option { background: #fff; color: #1a1a2e; }
        [data-theme="light"] input[type="date"] { color-scheme: light; }
        [data-theme="light"] input { color: #1a1a2e; }
        [data-theme="light"] ::placeholder { color: #9ca3af; }
        @media (max-width: 640px) {
          .hide-mobile { display: none !important; }
        }
        @media print {
          body * { visibility: hidden; } .print-area, .print-area * { visibility: visible; } .print-area { position: absolute; left: 0; top: 0; width: 100%; } .no-print { display: none !important; }
          .print-area .container { max-width: 100% !important; padding: 1cm !important; } .print-area table { font-size: 10px !important; }
          .print-area .text-white\\/40, .print-area .text-white\\/50, .print-area .text-white\\/55, .print-area .text-white\\/60, .print-area .text-white\\/70 { color: #555 !important; } .print-area .text-white { color: #000 !important; }
          .print-area .bg-white\\/10, .print-area .bg-white\\/[0\\.03\\], .print-area .bg-white\\/[0\\.04\\], .print-area .bg-white\\/[0\\.05\\] { background: #f5f5f5 !important; }
          .print-area .border-white\\/10, .print-area .border-white\\/20 { border-color: #ddd !important; } .print-area .bg-black\\/35 { background: transparent !important; }
          .print-area .backdrop-blur-xl { backdrop-filter: none !important; } .print-area svg { max-width: 100%; } .print-area .flex-wrap { flex-wrap: wrap !important; }
        }
      `}</style>

      <div className="no-print"><SiteHeader /></div>
      <main className="container pt-6 sm:pt-12 px-3 sm:px-4 print-area">
        <Link to="/" className="mb-4 sm:mb-6 inline-flex items-center gap-2 text-xs sm:text-sm text-white/55 transition hover:text-white no-print"><ArrowLeft className="h-3 w-3 sm:h-4 sm:w-4" />Back to home</Link>

        <div className="mb-4 sm:mb-8">
          <h1 className="text-2xl sm:text-4xl font-semibold tracking-tight text-white">Performance Trends</h1>
          <p className="mt-1 sm:mt-2 text-xs sm:text-base text-white/50">Longitudinal analysis across completed swim analysis sessions.</p>
        </div>

        {/* Filter Bar */}
        <Card className="mb-4 sm:mb-6 no-print">
          <CardContent className="flex flex-wrap items-center gap-2 sm:gap-3 p-3 sm:p-4">
            <User className="h-3 w-3 sm:h-4 sm:w-4 text-white/50" />
            <select value={swimmerId} onChange={(e) => setSwimmerId(e.target.value)} className="bg-transparent text-white text-xs sm:text-sm border border-white/20 rounded px-2 py-1 min-w-[100px] sm:min-w-[130px]">
              <option value="" className="bg-gray-900">All</option>
              {availableSwimmers.map((sid) => (<option key={sid} value={sid} className="bg-gray-900">{sid}</option>))}
            </select>

            <span className="w-px h-4 sm:h-5 bg-white/10 mx-0.5 sm:mx-1 hide-mobile" />
            <select value={analysisMode} onChange={(e) => setAnalysisMode(e.target.value)} className="bg-transparent text-white text-xs sm:text-sm border border-white/20 rounded px-2 py-1 min-w-[80px] sm:min-w-[100px] hide-mobile">
              <option value="" className="bg-gray-900">All modes</option>
              {availableModes.map((mode) => (<option key={mode} value={mode} className="bg-gray-900 capitalize">{mode}</option>))}
            </select>

            <span className="w-px h-4 sm:h-5 bg-white/10 mx-0.5 sm:mx-1" />
            <select value={aggregation} onChange={(e) => setAggregation(e.target.value)} className="bg-transparent text-white text-xs sm:text-sm border border-white/20 rounded px-2 py-1 min-w-[90px] sm:min-w-[100px] hide-mobile">
              <option value="" className="bg-gray-900">Per session</option>
              <option value="week" className="bg-gray-900">By week</option>
              <option value="month" className="bg-gray-900">By month</option>
            </select>

            <span className="w-px h-4 sm:h-5 bg-white/10 mx-0.5 sm:mx-1 hide-mobile" />
            <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="bg-transparent text-white text-xs sm:text-sm border border-white/20 rounded px-1 sm:px-2 py-1 w-[110px] sm:w-[140px] [color-scheme:dark] hide-mobile" />
            <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="bg-transparent text-white text-xs sm:text-sm border border-white/20 rounded px-1 sm:px-2 py-1 w-[110px] sm:w-[140px] [color-scheme:dark] hide-mobile" />

            <span className="w-px h-4 sm:h-5 bg-white/10 mx-0.5 sm:mx-1" />
            <input type="number" step="any" value={goalValue} onChange={(e) => setGoalValue(e.target.value)} placeholder={primaryCfg.unit || "Goal"} className="bg-transparent text-white text-xs sm:text-sm border border-white/20 rounded px-2 py-1 w-[65px] sm:w-[80px] placeholder:text-white/20" />

            <span className="w-px h-4 sm:h-5 bg-white/10 mx-0.5 sm:mx-1" />
            <input type="number" step="any" value={thresholdValue} onChange={(e) => setThresholdValue(e.target.value)} placeholder="Alert" className="bg-transparent text-white text-xs sm:text-sm border border-white/20 rounded px-2 py-1 w-[65px] sm:w-[80px] placeholder:text-white/20" />
            {thresholdValue && <button onClick={() => setThresholdValue("")} className="text-xs text-white/40 hover:text-white/70 transition"><X className="h-3 w-3 sm:h-3.5 sm:w-3.5" /></button>}

            <span className="w-px h-4 sm:h-5 bg-white/10 mx-0.5 sm:mx-1" />
            <button onClick={handleExportCSV} disabled={!sessions.length} className="inline-flex items-center gap-1 text-[10px] sm:text-xs rounded-full px-2 sm:px-3 py-1 transition text-white/50 hover:text-white border border-white/10 hover:border-white/20 disabled:opacity-30"><Download className="h-3 w-3 sm:h-3.5 sm:w-3.5" /><span className="hidden sm:inline">CSV</span></button>
            <button onClick={handlePrintPDF} disabled={!summary} className="inline-flex items-center gap-1 text-[10px] sm:text-xs rounded-full px-2 sm:px-3 py-1 transition text-white/50 hover:text-white border border-white/10 hover:border-white/20 disabled:opacity-30 hide-mobile"><Printer className="h-3 w-3 sm:h-3.5 sm:w-3.5" /><span className="hidden sm:inline">PDF</span></button>
            <button onClick={handleCopyReport} disabled={!summary} className={`inline-flex items-center gap-1 text-[10px] sm:text-xs rounded-full px-2 sm:px-3 py-1 transition border disabled:opacity-30 ${copied ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : "text-white/50 hover:text-white border-white/10 hover:border-white/20"}`}>{copied ? <Check className="h-3 w-3 sm:h-3.5 sm:w-3.5" /> : <Copy className="h-3 w-3 sm:h-3.5 sm:w-3.5" />}<span className="hidden sm:inline">{copied ? "Copied!" : "Share"}</span></button>
            <button onClick={() => { setCompareMode(!compareMode); if (compareMode) setComparison(null); }} className={`inline-flex items-center gap-1 text-[10px] sm:text-xs rounded-full px-2 sm:px-3 py-1 transition ${compareMode ? "bg-blue-500/20 text-blue-400 border border-blue-500/30" : "text-white/50 hover:text-white border border-white/10 hover:border-white/20"}`}><GitCompare className="h-3 w-3 sm:h-3.5 sm:w-3.5" /><span className="hidden sm:inline">Compare</span></button>
            <button onClick={() => setTheme(theme === "dark" ? "light" : "dark")} className="inline-flex items-center gap-1 text-[10px] sm:text-xs rounded-full px-2 sm:px-3 py-1 transition text-white/50 hover:text-white border border-white/10 hover:border-white/20 hide-mobile" title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>{theme === "dark" ? <Sun className="h-3 w-3 sm:h-3.5 sm:w-3.5" /> : <Moon className="h-3 w-3 sm:h-3.5 sm:w-3.5" />}</button>
            <button onClick={toggleCb} className={`inline-flex items-center gap-1 text-[10px] sm:text-xs rounded-full px-2 sm:px-3 py-1 transition ${cbPalette ? "bg-purple-500/20 text-purple-400 border-purple-500/30" : "text-white/50 hover:text-white border border-white/10 hover:border-white/20"}`} title="Toggle color-blind friendly palette"><Eye className="h-3 w-3 sm:h-3.5 sm:w-3.5" /><span className="hidden sm:inline">{cbPalette ? "CB" : ""}</span></button>
            <button onClick={() => setShowMetricBuilder(true)} className={`inline-flex items-center gap-1 text-[10px] sm:text-xs rounded-full px-2 sm:px-3 py-1 transition ${customMetrics.length > 0 ? "bg-blue-500/20 text-blue-400 border-blue-500/30" : "text-white/50 hover:text-white border border-white/10 hover:border-white/20"}`} title="Build custom metric"><Wrench className="h-3 w-3 sm:h-3.5 sm:w-3.5" /><span className="hidden sm:inline">{customMetrics.length > 0 ? customMetrics.length : "Build"}</span></button>
            {baselineIdx != null && sessions[baselineIdx] && (
              <button onClick={() => setBaselineIdx(null)} className="inline-flex items-center gap-1 text-[10px] sm:text-xs rounded-full px-2 sm:px-3 py-1 transition bg-emerald-500/15 text-emerald-400 border border-emerald-500/30" title="Clear baseline">
                <GitCommit className="h-3 w-3 sm:h-3.5 sm:w-3.5" />
                <span className="hidden sm:inline">Clear Baseline</span>
              </button>
            )}
          </CardContent>
        </Card>

        {/* Comparison Panel */}
        {compareMode && (
          <Card className="mb-4 sm:mb-6 border-blue-500/20 no-print">
            <CardContent className="p-4 sm:p-5 space-y-3 sm:space-y-4">
              <div className="flex items-center justify-between"><div className="flex items-center gap-2"><GitCompare className="h-3 w-3 sm:h-4 sm:w-4 text-blue-400" /><h2 className="text-base sm:text-lg font-semibold text-white">Swimmer Comparison</h2></div><button onClick={() => { setCompareMode(false); setComparison(null); }} className="text-white/40 hover:text-white/70 transition"><X className="h-3 w-3 sm:h-4 sm:w-4" /></button></div>
              <div className="flex flex-wrap items-center gap-2 sm:gap-3">
                <select value={swimmerA} onChange={(e) => setSwimmerA(e.target.value)} className="bg-transparent text-white text-xs sm:text-sm border border-white/20 rounded px-2 py-1 min-w-[100px] sm:min-w-[130px]"><option value="" className="bg-gray-900">Swimmer A</option>{availableSwimmers.map((sid) => (<option key={sid} value={sid} className="bg-gray-900">{sid}</option>))}</select>
                <ArrowRight className="h-3 w-3 sm:h-4 sm:w-4 text-white/30" />
                <select value={swimmerB} onChange={(e) => setSwimmerB(e.target.value)} className="bg-transparent text-white text-xs sm:text-sm border border-white/20 rounded px-2 py-1 min-w-[100px] sm:min-w-[130px]"><option value="" className="bg-gray-900">Swimmer B</option>{availableSwimmers.map((sid) => (<option key={sid} value={sid} className="bg-gray-900">{sid}</option>))}</select>
                <button onClick={handleCompare} disabled={compareLoading || !swimmerA || !swimmerB} className="inline-flex items-center gap-1.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30 px-3 sm:px-4 py-1 sm:py-1.5 text-[10px] sm:text-xs transition hover:bg-blue-500/30 disabled:opacity-40">{compareLoading ? "Comparing..." : "Run comparison"}</button>
              </div>
              {compareError && <p className="text-xs sm:text-sm text-red-400">{compareError}</p>}
              {comparison && (
                <div className="space-y-3 sm:space-y-4 pt-2 border-t border-white/10">
                  {comparison.comparison && (
                    <div className="grid gap-2 sm:gap-3 grid-cols-3">
                      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 sm:p-4"><p className="text-[10px] sm:text-xs text-white/40 mb-1">{comparison.comparison.swimmer_a_name}</p><p className="text-lg sm:text-xl text-white font-semibold">{comparison.comparison.swimmer_a_mean?.toFixed(1) || "—"}</p><TrendDirection direction={comparison.comparison.swimmer_a_direction} change={comparison.swimmer_a?.trend_summary?.primary_trend?.change} /></div>
                      <div className="rounded-xl border border-blue-500/20 bg-blue-500/[0.04] p-3 sm:p-4 text-center"><p className="text-[10px] sm:text-xs text-blue-400/60 mb-1">Difference</p><p className="text-lg sm:text-xl text-blue-400 font-semibold">{(comparison.comparison.diff > 0 ? "+" : "")}{comparison.comparison.diff?.toFixed(1) || "—"}</p><p className="text-[10px] sm:text-xs text-white/40">{primaryCfg.unit}</p></div>
                      <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 sm:p-4"><p className="text-[10px] sm:text-xs text-white/40 mb-1">{comparison.comparison.swimmer_b_name}</p><p className="text-lg sm:text-xl text-white font-semibold">{comparison.comparison.swimmer_b_mean?.toFixed(1) || "—"}</p><TrendDirection direction={comparison.comparison.swimmer_b_direction} change={comparison.swimmer_b?.trend_summary?.primary_trend?.change} /></div>
                    </div>
                  )}
                  {comparison.swimmer_a?.metric_trends && comparison.swimmer_b?.metric_trends && (
                    <div className="space-y-2 pt-2"><p className="text-[10px] sm:text-xs text-white/40 mb-2">Metric Breakdown</p>{Object.keys(comparison.swimmer_a.metric_trends).map((metric) => { const cfg = activeMetricConfig[metric] || { label: metric, unit: "", color: "#60a5fa" }; const valA = comparison.swimmer_a.metric_trends[metric]?.mean; const valB = comparison.swimmer_b.metric_trends[metric]?.mean; return (<ComparisonBar key={metric} valueA={valA} valueB={valB} maxValue={Math.max(valA || 0, valB || 0, 1)} color={cfg.color} label={`${cfg.label} (${cfg.unit})`} />); })}</div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {loading && (<Card><CardContent className="p-6 sm:p-8 text-center text-white/55"><Activity className="mx-auto mb-3 h-6 w-6 sm:h-8 sm:w-8 animate-pulse" />Loading trend data...</CardContent></Card>)}
        {error && (<Card><CardContent className="p-6 sm:p-8 text-center text-red-400">{error}</CardContent></Card>)}
        {!loading && !error && !summary && (<Card><CardContent className="p-6 sm:p-8 text-center text-white/55">No completed sessions found. Run some analyses first to see trends.</CardContent></Card>)}

        {summary && (
          <div className="space-y-4 sm:space-y-6">
            {/* Summary Cards - responsive grid */}
            <div className="grid gap-3 sm:gap-4 grid-cols-2 sm:grid-cols-3 md:grid-cols-5">
              <Card><CardContent className="p-3 sm:p-5"><p className="text-[10px] sm:text-xs text-white/40 uppercase tracking-wider">Sessions</p><p className="mt-1 text-lg sm:text-2xl font-semibold text-white">{summary.num_sessions}</p></CardContent></Card>
              <Card><CardContent className="p-3 sm:p-5"><p className="text-[10px] sm:text-xs text-white/40 uppercase tracking-wider">Worst Severity</p><p className="mt-1 text-lg sm:text-2xl font-semibold text-white">{summary.overall_worst_severity}</p></CardContent></Card>
              <Card><CardContent className="p-3 sm:p-5"><p className="text-[10px] sm:text-xs text-white/40 uppercase tracking-wider">Metrics</p><p className="mt-1 text-lg sm:text-2xl font-semibold text-white">{summary.metrics_with_trends}</p></CardContent></Card>
              <Card className={outlierInfo.count > 0 ? "border-amber-500/30" : ""}>
                <CardContent className="p-3 sm:p-5"><div className="flex items-center justify-between"><p className="text-[10px] sm:text-xs text-white/40 uppercase tracking-wider">Outliers</p><select value={outlierMethod} onChange={(e) => setOutlierMethod(e.target.value)} className="bg-transparent text-[9px] sm:text-[10px] text-white/40 border border-white/10 rounded px-1 py-0"><option value="2sigma" className="bg-gray-900">2σ</option><option value="iqr" className="bg-gray-900">IQR</option></select></div><div className="flex items-center gap-2 mt-1"><p className="text-lg sm:text-2xl font-semibold text-white">{outlierInfo.count}</p>{outlierInfo.count > 0 && <AlertTriangle className="h-3 w-3 sm:h-4 sm:w-4 text-amber-400" />}</div>{outlierInfo.bounds && <p className="text-[8px] sm:text-[10px] text-white/30 mt-1">μ={outlierInfo.bounds.mean.toFixed(1)} [{outlierInfo.bounds.lower.toFixed(1)}, {outlierInfo.bounds.upper.toFixed(1)}]</p>}</CardContent>
              </Card>
              <Card><CardContent className="p-3 sm:p-5"><p className="text-[10px] sm:text-xs text-white/40 uppercase tracking-wider">Metric</p><select value={primaryMetric} onChange={(e) => setPrimaryMetric(e.target.value)} className="mt-1 bg-transparent text-white text-xs sm:text-sm border border-white/20 rounded px-1 sm:px-2 py-1 w-full">{Object.keys(activeMetricConfig).map((key) => (<option key={key} value={key} className="bg-gray-900">{activeMetricConfig[key].label}</option>))}</select></CardContent></Card>
            </div>

            {/* Primary Trend */}
            {summary.primary_trend && (
              <Card><CardContent className="p-4 sm:p-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3 sm:mb-4 gap-2">
                  <div className="flex items-center gap-2 sm:gap-3 group">
                    {(() => { const Icon = primaryCfg.icon || Activity; return <Icon className="h-4 w-4 sm:h-5 sm:w-5" style={{ color: primaryCfg.color }} />; })()}
                    <h2 className="text-base sm:text-lg font-semibold text-white">{primaryCfg.label}</h2>
                    <Sparkline values={allSparklineData[primaryMetric]} dates={sparklineDates} color={primaryCfg.color} metric={primaryMetric} width={120} height={40} unit={primaryCfg.unit} goalValue={parsedGoal} crosshairIdx={crosshairIdx} onCrosshairChange={setCrosshairIdx} />
                    <button onClick={() => setZoomedMetric(primaryMetric)} className="opacity-0 group-hover:opacity-100 transition ml-0.5" title="Zoom chart">
                      <Maximize2 className="h-3 w-3 sm:h-3.5 sm:w-3.5 text-white/40 hover:text-white/70" />
                    </button>
                    {customMetrics.map((cm) => (
                      <span key={cm.name} className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition">
                        <button onClick={() => setZoomedMetric(cm.name)} title={`Zoom ${cm.name}`}><Maximize2 className="h-3 w-3 text-white/30 hover:text-white/60" /></button>
                      </span>
                    ))}
                  </div>
                  <TrendDirection direction={summary.primary_trend.direction} change={summary.primary_trend.change} />
                </div>
                <div className="grid gap-3 sm:gap-4 grid-cols-3 mb-3 sm:mb-4">
                  <div><p className="text-[10px] sm:text-xs text-white/40">Mean</p><p className="text-base sm:text-xl text-white">{summary.primary_trend.mean?.toFixed(1)}</p></div>
                  <div><p className="text-[10px] sm:text-xs text-white/40">Range</p><p className="text-base sm:text-xl text-white">{summary.primary_trend.min?.toFixed(1)} – {summary.primary_trend.max?.toFixed(1)}</p></div>
                  <div><p className="text-[10px] sm:text-xs text-white/40">Slope / Session</p><p className="text-base sm:text-xl text-white">{summary.primary_trend.slope?.toFixed(3)}</p></div>
                </div>
                {summary.date_range?.first && <p className="text-[10px] sm:text-xs text-white/40 mb-2 sm:mb-3">{summary.date_range.first} — {summary.date_range.last}</p>}
                <p className="text-xs sm:text-sm text-white/60 italic">{summary.summary_verdict}</p>
              </CardContent></Card>
            )}

            {/* All Metric Trends */}
            <Card><CardContent className="p-4 sm:p-6">
              <h2 className="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4">All Metric Trends</h2>
              <div className="space-y-2 sm:space-y-3">
                {Object.entries(metricTrends).map(([metric, trend]) => {
                  const cfg = activeMetricConfig[metric] || { label: metric, unit: "", color: "#60a5fa" };
                  return (<TrendBar key={metric} value={trend.mean || 0} maxValue={Math.max(...Object.values(metricTrends).map((t) => t.mean || 0), 1)} color={cfg.color} label={`${cfg.label} (${cfg.unit})`} sparkValues={sparklineData[metric]} sparkDates={sparklineDates} sparkMetric={metric} unit={cfg.unit} goalValue={metric === primaryMetric ? parsedGoal : null} crosshairIdx={crosshairIdx} onCrosshairChange={setCrosshairIdx} onZoom={setZoomedMetric} />);
                })}
                {customMetrics.map((cm) => {
                  const vals = customMetricValues[cm.name] || [];
                  const validVals = vals.filter((v) => v != null);
                  const mean = validVals.length > 0 ? validVals.reduce((a, b) => a + b, 0) / validVals.length : 0;
                  const allMax = Math.max(...Object.values(metricTrends).map((t) => t.mean || 0), ...Object.values(customMetricValues).map((vs) => Math.max(...(vs || []).filter((v) => v != null), 0)), 1);
                  return (<TrendBar key={cm.name} value={mean} maxValue={allMax} color={cm.color} label={`${cm.name} (${cm.unit || "—"})`} sparkValues={vals} sparkDates={sparklineDates} sparkMetric={cm.name} unit={cm.unit || "—"} crosshairIdx={crosshairIdx} onCrosshairChange={setCrosshairIdx} onZoom={setZoomedMetric} />);
                })}
              </div>
            </CardContent></Card>

            {/* Session Table */}
            <Card><CardContent className="p-4 sm:p-6">
              <div className="flex items-center justify-between mb-3 sm:mb-4">
                <h2 className="text-base sm:text-lg font-semibold text-white">Session History</h2>
                <div className="flex gap-1 sm:gap-2 no-print">
                  {sessions.length > 0 && (<>
                    <button onClick={handleExportCSV} className="inline-flex items-center gap-1 text-[10px] sm:text-xs text-white/50 hover:text-white transition"><Download className="h-3 w-3 sm:h-3.5 sm:w-3.5" /><span className="hidden sm:inline">CSV</span></button>
                    <button onClick={handlePrintPDF} className="inline-flex items-center gap-1 text-[10px] sm:text-xs text-white/50 hover:text-white transition hide-mobile"><Printer className="h-3 w-3 sm:h-3.5 sm:w-3.5" /><span className="hidden sm:inline">Print</span></button>
                    <button onClick={handleCopyReport} className={`inline-flex items-center gap-1 text-[10px] sm:text-xs transition ${copied ? "text-emerald-400" : "text-white/50 hover:text-white"}`}>{copied ? <Check className="h-3 w-3 sm:h-3.5 sm:w-3.5" /> : <Copy className="h-3 w-3 sm:h-3.5 sm:w-3.5" />}<span className="hidden sm:inline">{copied ? "Copied!" : "Share"}</span></button>
                  </>)}
                </div>
              </div>
              {/* Batch Action Bar */}
              {selectedSessions.size > 0 && (
                <div className="mb-3 flex items-center gap-2 sm:gap-3 px-3 py-2 rounded-lg bg-blue-500/[0.06] border border-blue-500/20">
                  <span className="text-[10px] sm:text-xs text-blue-400">{selectedSessions.size} selected</span>
                  <button onClick={handleExportSelectedCSV} className="inline-flex items-center gap-1 text-[10px] sm:text-xs text-blue-400 hover:text-blue-300 transition"><Download className="h-3 w-3" />Export CSV</button>
                  <button onClick={() => setSelectedSessions(new Set())} className="inline-flex items-center gap-1 text-[10px] sm:text-xs text-white/40 hover:text-white/70 transition"><X className="h-3 w-3" />Clear</button>
                </div>
              )}
              <div className="overflow-x-auto -mx-4 sm:mx-0">
                <table className="w-full text-[11px] sm:text-sm">
                  <thead><tr className="border-b border-white/10 text-white/50">
                    <th className="py-1.5 sm:py-2 text-left font-medium px-2 sm:px-0 w-6">
                      <button onClick={toggleSelectAll} className="text-white/30 hover:text-white/60 transition">
                        {selectedSessions.size === sessions.length && sessions.length > 0 ? <CheckSquare className="h-3.5 w-3.5" /> : <Square className="h-3.5 w-3.5" />}
                      </button>
                    </th>
                    <th className="py-1.5 sm:py-2 text-left font-medium px-2 sm:px-0">Session</th>
                    <th className="py-1.5 sm:py-2 text-left font-medium px-1">Date</th>
                    <th className="py-1.5 sm:py-2 text-left font-medium px-1 hide-mobile">Mode</th>
                    <th className="py-1.5 sm:py-2 text-left font-medium px-1">Severity</th>
                    <th className="py-1.5 sm:py-2 text-right font-medium px-1">{primaryCfg.label}</th>
                    {customMetrics.map((cm) => (<th key={cm.name} className="py-1.5 sm:py-2 text-right font-medium px-1">{cm.name}</th>))}
                    {baselineIdx != null && <th className="py-1.5 sm:py-2 text-right font-medium px-1">% vs Base</th>}
                    <th className="py-1.5 sm:py-2 text-left font-medium w-16 sm:w-28 no-print px-1"></th>
                  </tr></thead>
                  <tbody>
                    {sessions.map((session, idx) => {
                      const isOutlier = outlierInfo.outlierIds.has(session.session_id || idx);
                      const isAlert = thresholdInfo.alertIds.has(session.session_id || idx);
                      const isExpanded = expandedSessionIdx === idx;
                      const isBase = baselineIdx === idx;
                      const phases = parsePhaseMetrics(session.metrics, session.analysis_mode);
                      const baselineInfo = baselineIdx != null && baselineIdx !== idx
                        ? computeBaselinePct(session.metrics?.[primaryMetric], sessions[baselineIdx]?.metrics?.[primaryMetric], primaryCfg.higherIsBetter)
                        : null;

                      return (
                        <React.Fragment key={idx}>
                          <tr className={`border-b border-white/5 hover:bg-white/[0.04] cursor-pointer transition-colors ${isOutlier ? "bg-amber-500/[0.04]" : ""} ${isAlert ? "bg-red-500/[0.06]" : ""} ${isBase ? "bg-emerald-500/[0.06]" : ""} ${selectedSessions.has(idx) ? "bg-blue-500/[0.06]" : ""}`} onClick={() => setExpandedSessionIdx(prev => prev === idx ? null : idx)}>
                            <td className="py-2 sm:py-3 px-2 sm:px-0" onClick={(e) => e.stopPropagation()}>
                              <button onClick={() => toggleSelectSession(idx)} className="text-white/30 hover:text-white/60 transition">
                                {selectedSessions.has(idx) ? <CheckSquare className="h-3.5 w-3.5" /> : <Square className="h-3.5 w-3.5" />}
                              </button>
                            </td>
                            <td className="py-2 sm:py-3 text-white/80 px-2 sm:px-0 truncate max-w-[80px] sm:max-w-none">
                              <div className="flex items-center gap-1">
                                <ChevronRight className={`h-3 w-3 text-white/30 transition-transform flex-shrink-0 ${isExpanded ? "rotate-90" : ""}`} />
                                {session.session_id}
                              </div>
                            </td>
                            <td className="py-2 sm:py-3 text-white/50 px-1">{session.date || "—"}</td>
                            <td className="py-2 sm:py-3 px-1 hide-mobile"><span className="text-white/60">{session.analysis_mode}</span></td>
                            <td className="py-2 sm:py-3 px-1"><Badge className={session.overall_severity === "CRITICAL" ? "bg-red-500/15 text-red-400 border-red-500/30" : session.overall_severity === "SIGNIFICANT" ? "bg-orange-500/15 text-orange-400 border-orange-500/30" : session.overall_severity === "MINOR" ? "bg-yellow-500/15 text-yellow-400 border-yellow-500/30" : "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"}>{session.overall_severity}</Badge></td>
                            <td className={`py-2 sm:py-3 text-right px-1 ${isOutlier ? "text-amber-400 font-medium" : isAlert ? "text-red-400 font-medium" : isBase ? "text-emerald-400 font-medium" : "text-white/70"}`}>{session.metrics?.[primaryMetric]?.toFixed(1) || "—"}</td>
                            {customMetrics.map((cm) => {
                              const val = evaluateExpression(cm.expression, session.metrics);
                              return (<td key={cm.name} className="py-2 sm:py-3 text-right px-1 text-white/60 font-mono tabular-nums">{val != null ? val.toFixed(2) : "—"}</td>);
                            })}
                            {baselineIdx != null && (
                              <td className={`py-2 sm:py-3 text-right px-1 font-medium ${isBase ? "text-emerald-400" : baselineInfo?.cls || "text-white/40"}`}>
                                {isBase ? "Baseline" : baselineInfo ? `${baselineInfo.pct > 0 ? "+" : ""}${baselineInfo.pct.toFixed(1)}%` : "—"}
                              </td>
                            )}
                            <td className="py-2 sm:py-3 no-print px-1">
                              <div className="flex items-center gap-0.5 sm:gap-1">
                                {isOutlier && <Badge className="bg-amber-500/15 text-amber-400 border-amber-500/30 gap-0.5 text-[9px] sm:text-xs"><AlertTriangle className="h-2 w-2 sm:h-3 sm:w-3" />{outlierMethod === "2sigma" ? "2σ" : "IQR"}</Badge>}
                                {isAlert && <Badge className="bg-red-500/15 text-red-400 border-red-500/30 gap-0.5 text-[9px] sm:text-xs"><Bell className="h-2 w-2 sm:h-3 sm:w-3" />Alert</Badge>}
                                {isBase && <Badge className="bg-emerald-500/15 text-emerald-400 border-emerald-500/30 gap-0.5 text-[9px] sm:text-xs"><GitCommit className="h-2 w-2 sm:h-3 sm:w-3" />Base</Badge>}
                                <button onClick={(e) => { e.stopPropagation(); setBaselineIdx(prev => prev === idx ? null : idx); }} className={`ml-0.5 p-0.5 rounded transition ${isBase ? "text-emerald-400 bg-emerald-500/10" : "text-white/25 hover:text-white/60 hover:bg-white/10"}`} title={isBase ? "Remove baseline" : "Set as baseline"}><GitCommit className="h-3 w-3 sm:h-3.5 sm:w-3.5" /></button>
                              </div>
                            </td>
                          </tr>
                          {/* Session Explorer Panel */}
                          {isExpanded && (
                            <tr className="border-b border-white/5 bg-black/20">
                              <td colSpan={7 + customMetrics.length + (baselineIdx != null ? 1 : 0)} className="p-3 sm:p-4 expand-panel">
                                <div className="space-y-3">
                                  <div className="flex items-center gap-2 text-[10px] sm:text-xs text-white/40">
                                    <Layers className="h-3 w-3 sm:h-3.5 sm:w-3.5" />
                                    <span>Session detail — {session.session_id}</span>
                                    <span className="text-white/20">|</span>
                                    <span>{session.date || "No date"}</span>
                                    <span className="text-white/20">|</span>
                                    <span className="capitalize">{session.analysis_mode} mode</span>
                                    {session.num_cycles > 0 && <><span className="text-white/20">|</span><span>{session.num_cycles} cycles</span></>}
                                  </div>
                                  {phases.length === 0 ? (
                                    <p className="text-[10px] sm:text-xs text-white/30 italic">No phase metrics available for this session.</p>
                                  ) : (
                                    <div className="grid gap-2 sm:gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
                                      {phases.map((group) => (
                                        <div key={group.phase} className="rounded-lg border border-white/10 bg-white/[0.03] p-2 sm:p-3">
                                          <h4 className="text-[10px] sm:text-xs font-medium text-white/60 mb-1.5 sm:mb-2 uppercase tracking-wider">{group.phase}</h4>
                                          <div className="space-y-1">
                                            {group.metrics.map((m, mi) => (
                                              <div key={mi} className="flex items-center justify-between gap-2">
                                                <span className="text-[10px] sm:text-xs text-white/50 capitalize truncate">{m.name}</span>
                                                <span className="text-[10px] sm:text-xs text-white/80 font-mono tabular-nums">{typeof m.value === "number" ? m.value.toFixed(2) : String(m.value)}</span>
                                              </div>
                                            ))}
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {(outlierInfo.count > 0 || thresholdInfo.count > 0) && (
                <div className="mt-2 sm:mt-3 flex flex-wrap gap-x-3 gap-y-1 text-[10px] sm:text-xs">
                  {outlierInfo.count > 0 && <p className="text-amber-400/60">{outlierInfo.count} {outlierMethod === "2sigma" ? "2σ" : "IQR"} outlier{outlierInfo.count !== 1 ? "s" : ""}</p>}
                  {thresholdInfo.count > 0 && <p className="text-red-400/60">{thresholdInfo.count} threshold alert{thresholdInfo.count !== 1 ? "s" : ""} ({primaryCfg.label} &lt; {parsedThreshold})</p>}
                  {baselineIdx != null && <p className="text-emerald-400/60">Baseline: {sessions[baselineIdx]?.session_id || "—"}</p>}
                </div>
              )}
            </CardContent></Card>
          </div>
        )}

        {/* Zoom Modal with Chart.js */}
        {zoomedMetric && zoomData && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 no-print" onClick={() => setZoomedMetric(null)}>
            <Card className="w-full max-w-3xl max-h-[90vh] overflow-auto" onClick={e => e.stopPropagation()}>
              <CardContent className="p-4 sm:p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {(() => { const Icon = zoomData.cfg.icon || Activity; return <Icon className="h-5 w-5" style={{ color: zoomData.color }} />; })()}
                    <h2 className="text-lg sm:text-xl font-semibold text-white">{zoomData.cfg.label}</h2>
                    <Badge className="bg-white/10 text-white/60 border-white/20">{zoomData.cfg.unit}</Badge>
                  </div>
                  <button onClick={() => setZoomedMetric(null)} className="text-white/40 hover:text-white/70 transition"><X className="h-4 w-4 sm:h-5 sm:w-5" /></button>
                </div>
                <div className="h-64 sm:h-80">
                  <Line
                    data={{
                      labels: zoomData.dates || zoomData.vals.map((_, i) => i),
                      datasets: [{
                        label: zoomData.cfg.label,
                        data: zoomData.vals,
                        borderColor: zoomData.color,
                        backgroundColor: zoomData.color + "20",
                        fill: true,
                        tension: 0.3,
                        pointRadius: 4,
                        pointBackgroundColor: zoomData.color,
                        pointBorderColor: "#fff",
                        pointBorderWidth: 1.5,
                        pointHoverRadius: 6,
                        borderWidth: 2.5,
                      }],
                    }}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      interaction: { mode: "index", intersect: false },
                      plugins: {
                        legend: { display: false },
                        tooltip: {
                          backgroundColor: "rgba(17,17,25,0.95)",
                          titleColor: "rgba(255,255,255,0.7)",
                          bodyColor: "#fff",
                          borderColor: "rgba(255,255,255,0.1)",
                          borderWidth: 1,
                          padding: 10,
                          cornerRadius: 8,
                          displayColors: false,
                          callbacks: {
                            label: (ctx) => `${ctx.parsed.y.toFixed(2)} ${zoomData.cfg.unit}`,
                          },
                        },
                      },
                      scales: {
                        x: {
                          grid: { color: "rgba(255,255,255,0.05)" },
                          ticks: { color: "rgba(255,255,255,0.3)", font: { size: 10 }, maxRotation: 45, maxTicksLimit: 8 },
                        },
                        y: {
                          grid: { color: "rgba(255,255,255,0.05)" },
                          ticks: { color: "rgba(255,255,255,0.3)", font: { size: 10 }, callback: (v) => v.toFixed(1) },
                        },
                      },
                    }}
                  />
                </div>
                <div className="grid grid-cols-4 gap-2 sm:gap-3 text-center pt-2 border-t border-white/10">
                  <div><p className="text-[10px] sm:text-xs text-white/40">Min</p><p className="text-sm sm:text-base text-white">{Math.min(...zoomData.vals).toFixed(1)}</p></div>
                  <div><p className="text-[10px] sm:text-xs text-white/40">Max</p><p className="text-sm sm:text-base text-white">{Math.max(...zoomData.vals).toFixed(1)}</p></div>
                  <div><p className="text-[10px] sm:text-xs text-white/40">Mean</p><p className="text-sm sm:text-base text-white">{(zoomData.vals.reduce((a, b) => a + b, 0) / zoomData.vals.length).toFixed(1)}</p></div>
                  <div><p className="text-[10px] sm:text-xs text-white/40">Range</p><p className="text-sm sm:text-base text-white">{(Math.max(...zoomData.vals) - Math.min(...zoomData.vals)).toFixed(1)}</p></div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Custom Metric Builder Modal */}
        {showMetricBuilder && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 no-print" onClick={() => setShowMetricBuilder(false)}>
            <Card className="w-full max-w-md" onClick={e => e.stopPropagation()}>
              <CardContent className="p-4 sm:p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <h2 className="text-base sm:text-lg font-semibold text-white flex items-center gap-2"><Wrench className="h-4 w-4" />Custom Metric Builder</h2>
                  <button onClick={() => setShowMetricBuilder(false)} className="text-white/40 hover:text-white/70 transition"><X className="h-4 w-4" /></button>
                </div>
                <p className="text-[10px] sm:text-xs text-white/50">Create derived metrics using existing metric keys (e.g., <code className="text-blue-400">stroke_rate * 2</code>, <code className="text-blue-400">left_elbow_flexion / right_elbow_flexion</code>)</p>

                <div className="space-y-2">
                  <input type="text" value={newMetricName} onChange={(e) => setNewMetricName(e.target.value)} placeholder="Metric name (e.g. Elbow Ratio)" className="w-full bg-transparent text-white text-sm border border-white/20 rounded px-3 py-2 placeholder:text-white/20" />
                  <input type="text" value={newMetricExpr} onChange={(e) => setNewMetricExpr(e.target.value)} placeholder="Expression (e.g. left_elbow_flexion / right_elbow_flexion)" className="w-full bg-transparent text-white text-sm border border-white/20 rounded px-3 py-2 placeholder:text-white/20" />
                  <input type="text" value={newMetricUnit} onChange={(e) => setNewMetricUnit(e.target.value)} placeholder="Unit (e.g. ratio)" className="w-full bg-transparent text-white text-sm border border-white/20 rounded px-3 py-2 placeholder:text-white/20" />
                </div>

                <div className="flex gap-2">
                  <button onClick={handleAddCustomMetric} disabled={!newMetricName.trim() || !newMetricExpr.trim()} className="flex-1 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30 px-4 py-2 text-xs sm:text-sm transition hover:bg-blue-500/30 disabled:opacity-40">Add Metric</button>
                  <button onClick={() => setShowMetricBuilder(false)} className="rounded-full text-white/50 hover:text-white border border-white/10 px-4 py-2 text-xs sm:text-sm transition">Cancel</button>
                </div>

                {customMetrics.length > 0 && (
                  <div className="space-y-2 pt-2 border-t border-white/10">
                    <p className="text-[10px] sm:text-xs text-white/40">Current Custom Metrics</p>
                    {customMetrics.map((cm) => (
                      <div key={cm.name} className="flex items-center justify-between text-xs sm:text-sm bg-white/[0.03] rounded-lg px-3 py-2 border border-white/5">
                        <div>
                          <span className="text-white/80">{cm.name}</span>
                          <span className="text-white/30 ml-2">{cm.expression}</span>
                          {cm.unit && <span className="text-white/20 ml-1">({cm.unit})</span>}
                        </div>
                        <button onClick={() => handleRemoveCustomMetric(cm.name)} className="text-white/30 hover:text-red-400 transition"><Trash2 className="h-3 w-3" /></button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}

      </main>
    </div>
  );
}


