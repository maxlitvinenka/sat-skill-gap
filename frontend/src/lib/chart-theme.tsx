import type { TooltipProps } from "recharts";

export const chartPalette = [
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-1))",
] as const;

export const chartGrid = "hsl(var(--chart-grid))";
export const chartAxis = "hsl(var(--chart-axis))";

export const chartGridProps = {
  stroke: chartGrid,
  strokeDasharray: "2 6",
  vertical: false as const,
};

export const chartAxisProps = {
  stroke: chartAxis,
  tick: {
    fill: chartAxis,
    fontSize: 11,
    fontFamily: "var(--font-sans)",
  },
  tickLine: false,
  axisLine: { stroke: chartGrid },
} as const;

interface ReadingRoomsTooltipProps extends TooltipProps<number, string> {
  valueFormatter?: (value: number, name: string) => string;
}

export function ReadingRoomsTooltip({
  active,
  payload,
  label,
  valueFormatter,
}: ReadingRoomsTooltipProps) {
  if (!active || !payload?.length) return null;

  return (
    <div
      style={{
        background: "hsl(var(--chart-tooltip-bg))",
        border: "1px solid hsl(var(--chart-tooltip-border))",
        borderLeft: "3px solid hsl(var(--accent))",
        borderRadius: 8,
        padding: "10px 12px",
        minWidth: 140,
      }}
    >
      {label && (
        <div className="text-meta mb-1">{String(label)}</div>
      )}
      {payload.map((entry, idx) => {
        const value = typeof entry.value === "number" ? entry.value : Number(entry.value ?? 0);
        const name = String(entry.name ?? "");
        const display =
          valueFormatter?.(value, name) ?? `${Math.round(value * 100) / 100}`;
        return (
          <div key={`${name}-${idx}`} className="text-body-sm flex gap-2">
            <span>{name}</span>
            <span className="ml-auto font-semibold">{display}</span>
          </div>
        );
      })}
    </div>
  );
}
