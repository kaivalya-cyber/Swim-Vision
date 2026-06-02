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
} from "lucide-react";
import { useEffect, useState, useMemo } from "react";
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

function Sparkline({ values, color, metric, height = 40, width = 120 }) {
  if (!values || values.length < 2) return null;
  const vals = values.filter(v => v != null);
  if (vals.length < 2) return null;

  const safeColor = color || "#60a5fa";
  const gradientId = `spark-fill-${metric || "default"}-${safeColor.replace("#", "")}`;
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const range = max - min || 1;
  const padding = 2;
  const w = width - padding * 2;
  const h = height - padding * 2;

  const points = vals.map((v, i) => {
    const x = padding + (i / (vals.length - 1)) * w;
    const y = padding + h - ((v - min) / range) * h;
    return `${x},${y}`;
  }).join(" ");

  return (
    <svg width={width} height={height} className="shrink-0">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={safeColor} stopOpacity="0.25" />
          <stop offset="100%" stopColor={safeColor} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <polygon
        points={`${padding},${h + padding} ${points} ${width - padding},${h + padding}`}
        fill={`url(#${gradientId})`}
      />
      <polyline
        points={points}
        fill="none"
        stroke={safeColor}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function TrendBar({ value, maxValue, color, label, sparkValues, sparkMetric }) {
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
      <Sparkline values={sparkValues} color={color} metric={sparkMetric} />
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

  // Comparison mode state
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

  const summary = trends?.trend_summary;
  const sessions = trends?.sessions || [];
  const metricTrends = trends?.metric_trends || {};
  const availableSwimmers = trends?.available_swimmer_ids || [];
  const availableModes = trends?.available_analysis_modes || [];

  // Build sparkline values per metric from sessions
  const sparklineData = useMemo(() => {
    const data = {};
    Object.keys(metricTrends).forEach((metric) => {
      data[metric] = sessions.map((s) => s.metrics?.[metric]).filter((v) => v != null);
    });
    return data;
  }, [sessions, metricTrends]);

  return (
    <div className="min-h-screen pb-24">
      <SiteHeader />
      <main className="container pt-12">
        <Link to="/" className="mb-6 inline-flex items-center gap-2 text-sm text-white/55 transition hover:text-white">
          <ArrowLeft className="h-4 w-4" />
          Back to home
        </Link>

        <div className="mb-8">
          <h1 className="text-4xl font-semibold tracking-tight text-white">Performance Trends</h1>
          <p className="mt-2 text-white/50">Longitudinal analysis across completed swim analysis sessions.</p>
        </div>

        {/* Filter Bar */}
        <Card className="mb-6">
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

            {/* Aggregation toggle */}
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

            {/* Date range */}
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

            {/* CSV Export */}
            <span className="w-px h-5 bg-white/10 mx-1" />
            <button
              onClick={handleExportCSV}
              disabled={!sessions.length}
              className="inline-flex items-center gap-1.5 text-xs rounded-full px-3 py-1 transition text-white/50 hover:text-white border border-white/10 hover:border-white/20 disabled:opacity-30"
            >
              <Download className="h-3.5 w-3.5" />
              CSV
            </button>

            {/* Compare toggle */}
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

            {(swimmerId || analysisMode || startDate || endDate || aggregation) && (
              <span className="text-xs text-white/40 ml-auto">
                {summary?.num_sessions || 0} {aggregation ? aggregation : "session"}{(summary?.num_sessions || 0) !== 1 ? "s" : ""}
              </span>
            )}
          </CardContent>
        </Card>

        {/* Comparison Panel */}
        {compareMode && (
          <Card className="mb-6 border-blue-500/20">
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
            <div className="grid gap-4 md:grid-cols-4">
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
                      <Sparkline values={sparklineData[primaryMetric]} color={METRIC_CONFIG[primaryMetric]?.color || "#60a5fa"} metric={primaryMetric} width={160} height={48} />
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
                        sparkMetric={metric}
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
                  {sessions.length > 0 && (
                    <button
                      onClick={handleExportCSV}
                      className="inline-flex items-center gap-1.5 text-xs text-white/50 hover:text-white transition"
                    >
                      <Download className="h-3.5 w-3.5" />
                      Export CSV
                    </button>
                  )}
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
                      </tr>
                    </thead>
                    <tbody>
                      {sessions.map((session, idx) => (
                        <tr key={idx} className="border-b border-white/5 hover:bg-white/[0.02]">
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
                          <td className="py-3 text-right text-white/70">
                            {session.metrics?.[primaryMetric]?.toFixed(1) || "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
}
