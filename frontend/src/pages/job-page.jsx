import { ArrowLeft } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { fetchJob } from "@/api";
import { JobStatusPanel } from "@/components/job-status-panel";
import { SiteHeader } from "@/components/site-header";
import { Card, CardContent } from "@/components/ui/card";

export function JobPage() {
  const { jobId } = useParams();
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let timeoutId;

    async function poll() {
      try {
        const payload = await fetchJob(jobId);
        if (cancelled) return;
        setJob(payload);
        setError("");

        const existing = JSON.parse(window.localStorage.getItem("swimvisionRecentJobs") || "[]");
        const next = [payload.id, ...existing.filter((id) => id !== payload.id)].slice(0, 8);
        window.localStorage.setItem("swimvisionRecentJobs", JSON.stringify(next));

        if (payload.status === "queued" || payload.status === "running") {
          timeoutId = window.setTimeout(poll, 2500);
        }
      } catch (err) {
        if (cancelled) return;
        setError(err.message);
      }
    }

    poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [jobId]);

  return (
    <div className="min-h-screen pb-24">
      <SiteHeader />
      <main className="container pt-12">
        <Link to="/" className="mb-6 inline-flex items-center gap-2 text-sm text-white/55 transition hover:text-white">
          <ArrowLeft className="h-4 w-4" />
          Back to home
        </Link>

        {job ? (
          <JobStatusPanel job={job} />
        ) : (
          <Card>
            <CardContent className="p-8">
              <p className="text-sm text-white/55">
                {error || "Loading the analysis job..."}
              </p>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
}
