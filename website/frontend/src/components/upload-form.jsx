import { LoaderCircle, Sparkles } from "lucide-react";
import { useState } from "react";

import { createJob } from "@/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

const cropFields = [
  { name: "crop_x", label: "X", placeholder: "0" },
  { name: "crop_y", label: "Y", placeholder: "0" },
  { name: "crop_w", label: "Width", placeholder: "430" },
  { name: "crop_h", label: "Height", placeholder: "730" },
];

export function UploadForm({ onJobCreated }) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setIsSubmitting(true);
    setMessage("");

    const formData = new FormData(event.currentTarget);
    try {
      const payload = await createJob(formData);
      setMessage("Analysis started. Redirecting to the live job view...");
      onJobCreated(payload.job_id);
    } catch (error) {
      setMessage(error.message);
      setIsSubmitting(false);
    }
  }

  return (
    <Card className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,255,255,0.12),transparent_35%)]" />
      <CardContent className="relative space-y-6 p-6 md:p-8">
        <div className="space-y-3">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/6 px-3 py-1 text-[11px] uppercase tracking-[0.22em] text-white/72">
            <Sparkles className="h-3.5 w-3.5" />
            Analysis Studio
          </div>
          <div>
            <h3 className="text-2xl font-semibold tracking-tight text-white">Upload a swim-start clip</h3>
            <p className="mt-2 max-w-lg text-sm leading-6 text-white/55">
              Start a new SwimVision run with optional crop coordinates to isolate a single lane before pose extraction.
            </p>
          </div>
        </div>

        <form className="space-y-5" onSubmit={handleSubmit}>
          <div className="space-y-2">
            <label className="text-sm text-white/72">Video file</label>
            <Input
              type="file"
              name="video"
              accept=".mp4,.mov,.m4v,.avi,.mpg,.mpeg,.webm"
              required
              className="file:mr-4 file:rounded-full file:border-0 file:bg-white file:px-4 file:py-2 file:text-sm file:font-medium file:text-black hover:file:bg-white/90"
            />
          </div>

          <div className="grid gap-5 md:grid-cols-[1fr_1fr_0.9fr]">
            <div className="space-y-2">
              <label className="text-sm text-white/72">Swimmer ID</label>
              <Input name="swimmer_id" placeholder="dressel" />
            </div>
            <div className="space-y-2">
              <label className="text-sm text-white/72">Clip ID</label>
              <Input name="clip_id" placeholder="dressel-lane4-semifinal" />
            </div>
            <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-4">
              <p className="text-xs uppercase tracking-[0.18em] text-white/48">Crop override</p>
              <p className="mt-2 text-sm text-white/55">
                Leave blank for full frame, or enter all four values.
              </p>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-4">
            {cropFields.map((field) => (
              <div key={field.name} className="space-y-2">
                <label className="text-sm text-white/72">{field.label}</label>
                <Input type="number" min="0" name={field.name} placeholder={field.placeholder} />
              </div>
            ))}
          </div>

          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <Button size="lg" type="submit" disabled={isSubmitting} className="min-w-[180px]">
              {isSubmitting ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
              {isSubmitting ? "Starting analysis" : "Start analysis"}
            </Button>
            <p className="text-sm text-white/55" aria-live="polite">
              {message || "Outputs include the annotated video, JSON metrics, CSV angles, and PDF report."}
            </p>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
