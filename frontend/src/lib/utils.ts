import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function masteryColor(score: number): string {
  if (score >= 75) return "text-emerald-600";
  if (score >= 50) return "text-amber-600";
  if (score >= 25) return "text-orange-600";
  return "text-red-600";
}

export function categoryBadgeClass(category: string): string {
  const map: Record<string, string> = {
    Reading: "bg-blue-100 text-blue-900 border-blue-200",
    Evidence: "bg-purple-100 text-purple-900 border-purple-200",
    Grammar: "bg-emerald-100 text-emerald-900 border-emerald-200",
    Vocabulary: "bg-amber-100 text-amber-900 border-amber-200",
    Writing: "bg-sky-100 text-sky-900 border-sky-200",
    Rhetoric: "bg-rose-100 text-rose-900 border-rose-200",
  };
  return map[category] ?? "bg-muted text-foreground";
}
