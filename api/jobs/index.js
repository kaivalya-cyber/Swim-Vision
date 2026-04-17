function methodNotAllowed(res) {
  return res.status(405).json({ error: "Method not allowed" });
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return methodNotAllowed(res);
  }

  return res.status(501).json({
    error: "Video pipeline jobs are disabled in this Vercel deployment.",
    hint: "Deploy a dedicated compute backend for extraction and analysis jobs.",
  });
}
