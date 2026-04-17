import { cva } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium tracking-[0.16em] uppercase",
  {
    variants: {
      variant: {
        default: "border-white/12 bg-white/7 text-white/80",
        running: "border-amber-300/20 bg-amber-200/12 text-amber-50",
        completed: "border-emerald-300/20 bg-emerald-200/12 text-emerald-50",
        failed: "border-red-400/20 bg-red-400/12 text-red-50",
        queued: "border-white/12 bg-white/7 text-white/70",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export function Badge({ className, variant, ...props }) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}
