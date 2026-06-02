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

export async function fetchTrends(primaryMetric = "stroke_rate", swimmerId = "", analysisMode = "", startDate = "", endDate = "") {
  let url = `/api/trends?primary_metric=${encodeURIComponent(primaryMetric)}`;
  if (swimmerId) {
    url += `&swimmer_id=${encodeURIComponent(swimmerId)}`;
  }
  if (analysisMode) {
    url += `&analysis_mode=${encodeURIComponent(analysisMode)}`;
  }
  if (startDate) {
    url += `&start_date=${encodeURIComponent(startDate)}`;
  }
  if (endDate) {
    url += `&end_date=${encodeURIComponent(endDate)}`;
  }
  const response = await fetch(url);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || "Could not load trend data.");
  }
  return response.json();
}

export async function compareSwimmers(swimmerA, swimmerB, primaryMetric = "stroke_rate") {
  const url = `/api/trends/compare?swimmer_a=${encodeURIComponent(swimmerA)}&swimmer_b=${encodeURIComponent(swimmerB)}&primary_metric=${encodeURIComponent(primaryMetric)}`;
  const response = await fetch(url);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || "Could not load comparison data.");
  }
  return response.json();
}
