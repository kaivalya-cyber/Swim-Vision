import { AlertCircle, ArrowUpRight, FileDown, Video } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { formatLabel, severityTone } from "@/lib/utils";

function statusVariant(status) {
  if (status === "completed") return "completed";
  if (status === "running") return "running";
  if (status === "failed") return "failed";
  return "queued";
}

export function JobStatusPanel({ job }) {
  const progress = job.total_steps ? (job.step_index / job.total_steps) * 100 : 0;

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="space-y-6 p-6 md:p-8">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-white/45">Live Analysis</p>
              <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">{job.clip_id}</h1>
              <p className="mt-2 text-sm text-white/45">{job.original_filename}</p>
            </div>
            <Badge variant={statusVariant(job.status)}>{job.status}</Badge>
          </div>

          <div className="space-y-3">
            <div className="flex flex-col gap-2 text-sm text-white/65 md:flex-row md:items-center md:justify-between">
              <span>{job.current_step}</span>
              <span>{job.step_index}/{job.total_steps} steps</span>
            </div>
            <Progress value={progress} />
            {job.error ? (
              <div className="flex items-start gap-2 rounded-2xl border border-red-400/15 bg-red-500/10 px-4 py-3 text-sm text-red-100">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{job.error}</span>
              </div>
            ) : null}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <Card>
          <CardContent className="space-y-6 p-6 md:p-8">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-white/45">Report Summary</p>
              <h2 className="mt-2 text-2xl font-semibold text-white">What SwimVision found</h2>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-white/45">Overall Severity</p>
                <p className="mt-3 text-2xl font-medium text-white">
                  {job.summary?.overall_severity || "Pending"}
                </p>
              </div>
              <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-4">
                <p className="text-xs uppercase tracking-[0.2em] text-white/45">Phase Boundaries</p>
                <p className="mt-3 text-2xl font-medium text-white">
                  {Object.keys(job.summary?.phase_boundaries || {}).length || 0}
                </p>
              </div>
            </div>

            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.2em] text-white/45">Detected boundaries</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(job.summary?.phase_boundaries || {}).length > 0 ? (
                  Object.entries(job.summary.phase_boundaries).map(([key, value]) => (
                    <div key={key} className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-white/72">
                      {formatLabel(key)}: {value}
                    </div>
                  ))
                ) : (
                  <span className="text-sm text-white/45">Phase boundaries will appear when the run completes.</span>
                )}
              </div>
            </div>

            <div className="space-y-3">
              <p className="text-xs uppercase tracking-[0.2em] text-white/45">Flagged metrics</p>
              <div className="space-y-3">
                {job.summary?.flagged_metrics?.length ? (
                  job.summary.flagged_metrics.map((metric) => (
                    <div
                      key={`${metric.phase}-${metric.metric}`}
                      className={`rounded-[24px] border px-4 py-4 ${severityTone(metric.flag)}`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="font-medium text-white">
                            {formatLabel(metric.phase)} / {formatLabel(metric.metric)}
                          </p>
                          <p className="mt-1 text-sm text-white/70">Measured: {Number(metric.measured).toFixed(2)}</p>
                        </div>
                        <span className="rounded-full border border-current/20 px-3 py-1 text-xs uppercase tracking-[0.18em]">
                          {metric.flag}
                        </span>
                      </div>
                    </div>
                  ))
                ) : (
                  <span className="text-sm text-white/45">No flagged metrics yet.</span>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-5 p-6 md:p-8">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-white/45">Artifacts</p>
              <h2 className="mt-2 text-2xl font-semibold text-white">Open the generated outputs</h2>
            </div>

            <div className="space-y-3">
              {job.artifact_urls && Object.keys(job.artifact_urls).length > 0 ? (
                Object.entries(job.artifact_urls).map(([name, url]) => (
                  <a
                    key={name}
                    href={url}
                    target="_blank"
                    rel="noreferrer"
                    className="group flex items-center justify-between rounded-[24px] border border-white/10 bg-white/[0.03] px-4 py-4 transition hover:bg-white/[0.05]"
                  >
                    <div className="flex items-center gap-3">
                      <div className="rounded-full border border-white/10 bg-white/[0.03] p-2">
                        {name.includes("video") ? <Video className="h-4 w-4" /> : <FileDown className="h-4 w-4" />}
                      </div>
                      <div>
                        <p className="font-medium text-white">{formatLabel(name)}</p>
                        <p className="text-sm text-white/45">{job.outputs?.[name]?.split("/").pop()}</p>
                      </div>
                    </div>
                    <ArrowUpRight className="h-4 w-4 text-white/35 transition group-hover:text-white/80" />
                  </a>
                ))
              ) : (
                <div className="rounded-[24px] border border-dashed border-white/10 bg-white/[0.02] px-4 py-8 text-sm text-white/45">
                  Artifacts will appear here when processing completes.
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
