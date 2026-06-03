import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export function formatLabel(value) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function severityTone(flag) {
  switch (flag) {
    case "CRITICAL":
      return "bg-red-500/15 text-red-200 border-red-500/20";
    case "SIGNIFICANT":
      return "bg-orange-400/15 text-orange-100 border-orange-400/20";
    case "MINOR":
      return "bg-amber-300/15 text-amber-50 border-amber-200/20";
    case "OPTIMAL":
      return "bg-emerald-400/15 text-emerald-100 border-emerald-400/20";
    default:
      return "bg-white/10 text-white/80 border-white/10";
  }
}
