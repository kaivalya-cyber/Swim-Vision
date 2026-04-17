import {
  ArrowDown,
  ArrowRight,
  ChevronRight,
  FileVideo2,
  Radar,
  ScanSearch,
  Waves,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { fetchJob } from "@/api";
import { RecentJobs } from "@/components/recent-jobs";
import { SiteHeader } from "@/components/site-header";
import { UploadForm } from "@/components/upload-form";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const featureCards = [
  {
    icon: ScanSearch,
    title: "Pose extraction, phase segmentation, and overlays",
    body: "Run the full biomechanics pipeline from one browser flow without switching between Python scripts.",
  },
  {
    icon: Radar,
    title: "Deviation scoring against literature targets",
    body: "Spot where a swimmer drifts from optimal block, flight, and entry mechanics in seconds.",
  },
  {
    icon: FileVideo2,
    title: "Artifact-ready outputs for coaches and athletes",
    body: "Get an annotated video, JSON reports, angle CSVs, and a PDF summary from the same run.",
  },
];

export function HomePage() {
  const [recentJobs, setRecentJobs] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const ids = JSON.parse(window.localStorage.getItem("swimvisionRecentJobs") || "[]");
    if (!Array.isArray(ids) || ids.length === 0) {
      return;
    }

    let cancelled = false;
    Promise.all(ids.slice(0, 8).map((jobId) => fetchJob(jobId).catch(() => null))).then((jobs) => {
      if (cancelled) return;
      setRecentJobs(jobs.filter(Boolean));
    });

    return () => {
      cancelled = true;
    };
  }, []);

  function handleJobCreated(jobId) {
    const existing = JSON.parse(window.localStorage.getItem("swimvisionRecentJobs") || "[]");
    const next = [jobId, ...existing.filter((id) => id !== jobId)].slice(0, 8);
    window.localStorage.setItem("swimvisionRecentJobs", JSON.stringify(next));
    navigate(`/jobs/${jobId}`);
  }

  return (
    <div className="relative overflow-hidden">
      <SiteHeader />

      <main>
        <section className="relative pt-12 md:pt-20">
          <div className="container relative min-h-[840px] pb-20">
            <div className="absolute inset-x-0 top-0 -z-10 h-[720px] overflow-hidden">
              <div className="hero-veil left-[6%] top-[12%] h-[120px] w-[380px] rotate-[-48deg] animate-shimmer md:h-[160px] md:w-[520px]" />
              <div className="hero-veil right-[3%] top-[18%] h-[150px] w-[480px] rotate-[32deg] animate-float md:h-[200px] md:w-[620px]" />
              <div className="hero-veil left-[20%] bottom-[12%] h-[160px] w-[640px] rotate-[10deg] animate-float md:h-[200px] md:w-[760px]" />
              <div className="hero-veil right-[8%] bottom-[4%] h-[130px] w-[440px] rotate-[-18deg] animate-shimmer md:h-[180px] md:w-[560px]" />
            </div>

            <div className="relative mx-auto flex max-w-5xl flex-col items-center text-center">
              <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.05] px-4 py-2 text-[11px] uppercase tracking-[0.24em] text-white/72">
                <Waves className="h-3.5 w-3.5" />
                Swim start analysis, redesigned for the browser
              </div>

              <h1 className="max-w-4xl text-balance text-5xl font-medium leading-[0.95] tracking-[-0.06em] text-white md:text-7xl">
                Biomechanics feedback that feels as sharp as the race itself.
              </h1>

              <p className="mt-8 max-w-2xl text-pretty text-base leading-7 text-white/58 md:text-lg">
                Upload one swim-start clip and get annotated motion, phase boundaries, and deviation scoring in a premium
                studio experience instead of a pile of scripts and output folders.
              </p>

              <div className="mt-10 flex flex-col gap-3 sm:flex-row">
                <a href="#studio">
                  <Button size="lg" className="min-w-[180px]">
                    Start analysis
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </a>
                <a href="#features">
                  <Button variant="secondary" size="lg" className="min-w-[180px]">
                    Explore features
                  </Button>
                </a>
              </div>

              <div className="mt-16 flex items-center gap-3 text-sm text-white/38">
                <span>Scroll to studio</span>
                <ArrowDown className="h-4 w-4" />
              </div>
            </div>
          </div>
        </section>

        <section id="features" className="pb-14">
          <div className="container grid gap-5 lg:grid-cols-3">
            {featureCards.map((feature) => (
              <Card key={feature.title} className="bg-white/[0.035]">
                <CardContent className="space-y-4 p-6 md:p-7">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full border border-white/10 bg-white/[0.04]">
                    <feature.icon className="h-5 w-5 text-white" />
                  </div>
                  <div className="space-y-2">
                    <h2 className="text-xl font-medium text-white">{feature.title}</h2>
                    <p className="text-sm leading-6 text-white/52">{feature.body}</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        <section id="workflow" className="pb-14">
          <div className="container">
            <Card className="overflow-hidden">
              <CardContent className="grid gap-8 p-6 md:grid-cols-[0.95fr_1.05fr] md:p-8 xl:p-10">
                <div className="space-y-5">
                  <p className="text-xs uppercase tracking-[0.22em] text-white/45">Workflow</p>
                  <h2 className="max-w-lg text-3xl font-semibold tracking-tight text-white md:text-4xl">
                    One surface for upload, progress, and biomechanical review.
                  </h2>
                  <p className="max-w-md text-sm leading-6 text-white/55">
                    The new frontend keeps the underlying SwimVision pipeline intact, but wraps it in a cleaner experience
                    that is easier to share with coaches, swimmers, and teammates.
                  </p>
                </div>

                <div className="grid gap-4">
                  {[
                    "Upload a race or training clip and optionally crop to a single lane.",
                    "Track each pipeline stage live while extraction, segmentation, and reporting run in sequence.",
                    "Review flagged metrics and open the annotated MP4, JSON outputs, and PDF report directly.",
                  ].map((step, index) => (
                    <div key={step} className="flex gap-4 rounded-[24px] border border-white/10 bg-white/[0.03] p-4">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] text-sm font-medium text-white">
                        0{index + 1}
                      </div>
                      <div className="pt-1 text-sm leading-6 text-white/60">{step}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        <section id="studio" className="pb-24">
          <div className="container space-y-6">
            <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-white/45">Studio</p>
                <h2 className="mt-2 text-3xl font-semibold tracking-tight text-white md:text-4xl">
                  Launch a run and keep everything in view
                </h2>
              </div>
              <div className="flex items-center gap-2 text-sm text-white/45">
                Backed by the existing SwimVision pipeline
                <ChevronRight className="h-4 w-4" />
              </div>
            </div>

            <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
              <UploadForm onJobCreated={handleJobCreated} />
              <RecentJobs jobs={recentJobs} />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
