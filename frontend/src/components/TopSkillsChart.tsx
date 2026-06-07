import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  chartAxisProps,
  chartGridProps,
  chartPalette,
  ReadingRoomsTooltip,
} from "@/lib/chart-theme";
import type { Recommendation } from "@/types";

export function TopSkillsChart({ rows }: { rows: Recommendation[] }) {
  const data = rows.slice(0, 10).map((r) => ({
    name: r.skillName.length > 22 ? `${r.skillName.slice(0, 20)}…` : r.skillName,
    priority: r.priorityScore,
    fullName: r.skillName,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Top recommended skills</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={360} aria-label="Top priority skills bar chart">
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
            <CartesianGrid {...chartGridProps} horizontal={false} />
            <XAxis type="number" {...chartAxisProps} />
            <YAxis type="category" dataKey="name" width={160} {...chartAxisProps} />
            <Tooltip
              content={
                <ReadingRoomsTooltip
                  valueFormatter={(v) => v.toFixed(1)}
                />
              }
            />
            <Bar dataKey="priority" name="Priority" fill={chartPalette[0]} radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
