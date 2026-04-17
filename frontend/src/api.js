export async function createJob(formData) {
  const response = await fetch("/api/jobs", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Could not start the analysis.");
  }
  return payload;
}

export async function fetchJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error("Could not load the analysis job.");
  }
  return response.json();
}
