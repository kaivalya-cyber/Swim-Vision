function methodNotAllowed(res) {
  return res.status(405).json({ error: "Method not allowed" });
}

export default async function handler(req, res) {
  if (req.method !== "GET") {
    return methodNotAllowed(res);
  }

  const { jobId } = req.query;
  return res.status(404).json({
    error: `Job '${jobId}' not found in Vercel-lite mode.`,
  });
}
