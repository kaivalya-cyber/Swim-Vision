import { Route, Routes } from "react-router-dom";

import { HomePage } from "@/pages/home-page";
import { JobPage } from "@/pages/job-page";
import { TrendsPage } from "@/pages/trends-page";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/jobs/:jobId" element={<JobPage />} />
      <Route path="/trends" element={<TrendsPage />} />
    </Routes>
  );
}
