import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Activity,
  Gauge,
  Timer,
  Dumbbell,
  User,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { fetchTrends } from "@/api";
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

function TrendBar({ value, maxValue, color, label }) {
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

export function TrendsPage() {
  const [trends, setTrends] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [primaryMetric, setPrimaryMetric] = useState("stroke_rate");
  const [swimmerId, setSwimmerId] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const data = await fetchTrends(primaryMetric, swimmerId);
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
  }, [primaryMetric, swimmerId]);

  const summary = trends?.trend_summary;
  const sessions = trends?.sessions || [];
  const metricTrends = trends?.metric_trends || {};

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

        {/* Swimmer filter */}
        {trends?.available_swimmer_ids?.length > 0 && (
          <Card className="mb-6">
            <CardContent className="flex items-center gap-4 p-4">
              <User className="h-4 w-4 text-white/50" />
              <label className="text-sm text-white/60">Swimmer:</label>
              <select
                value={swimmerId}
                onChange={(e) => setSwimmerId(e.target.value)}
                className="bg-transparent text-white text-sm border border-white/20 rounded px-3 py-1.5 min-w-[160px]"
              >
                <option value="" className="bg-gray-900">All swimmers</option>
                {trends.available_swimmer_ids.map((sid) => (
                  <option key={sid} value={sid} className="bg-gray-900">
                    {sid}
                  </option>
                ))}
              </select>
              {swimmerId && (
                <span className="text-xs text-white/40">
                  Showing {trends.trend_summary?.num_sessions || 0} sessions for {swimmerId}
                </span>
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

            {/* Primary Trend */}
            {summary.primary_trend && (
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      {(() => {
                        const cfg = METRIC_CONFIG[primaryMetric] || { icon: Activity, color: "#60a5fa" };
                        const Icon = cfg.icon;
                        return <Icon className="h-5 w-5" style={{ color: cfg.color }} />;
                      })()}
                      <h2 className="text-lg font-semibold text-white">
                        {METRIC_CONFIG[primaryMetric]?.label || primaryMetric}
                      </h2>
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

                  <p className="text-sm text-white/60 italic">{summary.summary_verdict}</p>
                </CardContent>
              </Card>
            )}

            {/* All Metric Trends */}
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
                      />
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* Session Table */}
            <Card>
              <CardContent className="p-6">
                <h2 className="text-lg font-semibold text-white mb-4">Session History</h2>
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
