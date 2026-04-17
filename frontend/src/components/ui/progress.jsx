import { cn } from "@/lib/utils";

export function Progress({ value, className }) {
  return (
    <div className={cn("h-2.5 w-full overflow-hidden rounded-full bg-white/10", className)}>
      <div
        className="h-full rounded-full bg-[linear-gradient(90deg,rgba(255,255,255,0.55),rgba(255,255,255,0.95))] transition-all duration-500"
        style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
      />
    </div>
  );
}
