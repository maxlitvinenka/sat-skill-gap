import * as React from "react";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";

export function PageHeading({
  eyebrow,
  title,
  description,
  action,
  icon,
  className,
}: {
  eyebrow?: React.ReactNode;
  title: React.ReactNode;
  description?: React.ReactNode;
  action?: React.ReactNode;
  icon?: React.ReactNode;
  className?: string;
}) {
  return (
    <header
      className={cn(
        "flex flex-col gap-3 md:flex-row md:items-end md:justify-between",
        className,
      )}
    >
      <div className="min-w-0 space-y-1.5">
        {eyebrow && <div className="text-eyebrow">{eyebrow}</div>}
        <div className="flex items-center gap-3 min-w-0">
          {icon && <div className="shrink-0 text-primary">{icon}</div>}
          <h1 className="text-display text-foreground">{title}</h1>
        </div>
        {description && (
          <p className="text-body text-muted-foreground max-w-2xl">{description}</p>
        )}
      </div>
      {action && <div className="flex items-center gap-2 shrink-0">{action}</div>}
    </header>
  );
}

export function SectionHeading({
  title,
  description,
  className,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("space-y-1", className)}>
      <h2 className="text-h2 text-foreground">{title}</h2>
      {description && <p className="text-body-sm text-muted-foreground">{description}</p>}
    </div>
  );
}

type StatTone = "default" | "primary" | "accent" | "warning";

const TONE: Record<StatTone, { iconBg: string; iconText: string }> = {
  default: { iconBg: "bg-muted", iconText: "text-muted-foreground" },
  primary: { iconBg: "bg-primary/10", iconText: "text-primary" },
  accent: { iconBg: "bg-accent/15", iconText: "text-brand-gold" },
  warning: { iconBg: "bg-amber-500/10", iconText: "text-amber-600" },
};

export function StatCard({
  icon,
  label,
  value,
  sublabel,
  tone = "default",
  className,
}: {
  icon?: React.ReactNode;
  label: React.ReactNode;
  value: React.ReactNode;
  sublabel?: React.ReactNode;
  tone?: StatTone;
  className?: string;
}) {
  const palette = TONE[tone];
  return (
    <Card className={cn("p-5 flex items-center gap-4", className)}>
      {icon && (
        <div
          className={cn(
            "shrink-0 w-12 h-12 rounded-md flex items-center justify-center",
            palette.iconBg,
            palette.iconText,
          )}
        >
          {icon}
        </div>
      )}
      <div className="min-w-0 flex-1">
        <div className="text-meta">{label}</div>
        <div className="text-metric mt-1">{value}</div>
        {sublabel && (
          <div className="text-body-sm text-muted-foreground mt-0.5">{sublabel}</div>
        )}
      </div>
    </Card>
  );
}
