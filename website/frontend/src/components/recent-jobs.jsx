import { ArrowRight, Clock3 } from "lucide-react";
import { Link } from "react-router-dom";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

function statusVariant(status) {
  if (status === "completed") return "completed";
  if (status === "running") return "running";
  if (status === "failed") return "failed";
  return "queued";
}

export function RecentJobs({ jobs }) {
  return (
    <Card>
      <CardContent className="space-y-6 p-6 md:p-8">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-white/45">Recent Jobs</p>
            <h3 className="mt-2 text-2xl font-semibold text-white">Jump back into prior runs</h3>
          </div>
          <Clock3 className="h-5 w-5 text-white/45" />
        </div>

        <div className="space-y-3">
          {jobs.length > 0 ? (
            jobs.map((job) => (
              <Link
                key={job.id}
                to={`/jobs/${job.id}`}
                className="group flex items-center justify-between gap-4 rounded-[24px] border border-white/10 bg-white/[0.03] px-4 py-4 transition hover:bg-white/[0.05]"
              >
                <div className="space-y-1">
                  <p className="font-medium text-white">{job.clip_id}</p>
                  <p className="text-sm text-white/45">{job.original_filename}</p>
                </div>
                <div className="flex items-center gap-3">
                  <Badge variant={statusVariant(job.status)}>{job.status}</Badge>
                  <ArrowRight className="h-4 w-4 text-white/35 transition group-hover:translate-x-0.5 group-hover:text-white/75" />
                </div>
              </Link>
            ))
          ) : (
            <div className="rounded-[24px] border border-dashed border-white/10 bg-white/[0.02] px-4 py-8 text-sm text-white/45">
              No browser-side jobs yet. Start with a race or practice clip.
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
