import { Activity, ArrowUpRight } from "lucide-react";
import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-30">
      <div className="container pt-5">
        <div className="glass-line flex items-center justify-between rounded-full border border-white/10 bg-black/35 px-4 py-3 backdrop-blur-xl md:px-6">
          <Link to="/" className="flex items-center gap-2 text-sm font-medium tracking-wide text-white">
            <span className="rounded-full border border-white/10 bg-white/5 p-2">
              <Activity className="h-4 w-4" />
            </span>
            SwimVision
          </Link>

          <nav className="hidden items-center gap-6 text-xs text-white/55 md:flex">
            <a href="#features" className="transition hover:text-white">Features</a>
            <a href="#workflow" className="transition hover:text-white">Workflow</a>
            <a href="#studio" className="transition hover:text-white">Studio</a>
            <Link to="/trends" className="transition hover:text-white">Trends</Link>
          </nav>

          <a href="#studio">
            <Button variant="secondary" size="sm" className="h-10 px-5">
              Open Studio
              <ArrowUpRight className="h-4 w-4" />
            </Button>
          </a>
        </div>
      </div>
    </header>
  );
}
