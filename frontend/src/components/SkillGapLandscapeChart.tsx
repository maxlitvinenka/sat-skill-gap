import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SectionHeading } from "@/components/ui/section-heading";
import { chartAxisProps, chartGridProps } from "@/lib/chart-theme";
import type { Recommendation } from "@/types";

const TESTED_FILL = "hsl(var(--chart-3))";
const PREDICTED_FILL = "hsl(var(--chart-2))";
const TOP_PRIORITY_STROKE = "hsl(var(--accent))";

type LandscapeRow = {
  rank: number;
  name: string;
  fullName: string;
  mastery: number;
  gap: number;
  priority: number;
  isTested: boolean;
  category: string;
  level: string;
};

function LandscapeTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: LandscapeRow }[];
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;

  return (
    <div
      style={{
        background: "hsl(var(--chart-tooltip-bg))",
        border: "1px solid hsl(var(--chart-tooltip-border))",
        borderLeft: "3px solid hsl(var(--accent))",
        borderRadius: 8,
        padding: "12px 14px",
        maxWidth: 320,
      }}
    >
      <p className="text-body-sm font-semibold mb-2">{row.fullName}</p>
      <dl className="text-body-sm space-y-1">
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">Rank</dt>
          <dd className="font-medium tabular-nums">#{row.rank}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">Status</dt>
          <dd className="font-medium">{row.isTested ? "Tested (observed)" : "Untested (predicted)"}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">Mastery</dt>
          <dd className="font-medium tabular-nums">{row.mastery.toFixed(0)}%</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">Gap (100 − mastery)</dt>
          <dd className="font-medium tabular-nums">{row.gap.toFixed(0)}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">Priority score</dt>
          <dd className="font-medium tabular-nums">{row.priority.toFixed(1)}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">Category</dt>
          <dd className="font-medium text-right">{row.category}</dd>
        </div>
      </dl>
    </div>
  );
}

export function SkillGapLandscapeChart({
  rows,
  studentId,
}: {
  rows: Recommendation[];
  studentId: string;
}) {
  const data: LandscapeRow[] = rows.map((r) => ({
    rank: r.rank,
    name:
      r.skillName.length > 30 ? `${r.skillName.slice(0, 28)}…` : r.skillName,
    fullName: r.skillName,
    mastery: r.predictedMastery,
    gap: 100 - r.predictedMastery,
    priority: r.priorityScore,
    isTested: r.isTested ?? false,
    category: r.category,
    level: r.level,
  }));

  const chartHeight = Math.min(2200, Math.max(880, data.length * 26));
  const testedCount = data.filter((d) => d.isTested).length;
  const untestedCount = data.length - testedCount;

  return (
    <section id="conclusion" className="scroll-mt-8 space-y-4">
      <SectionHeading
        title="Complete skill landscape"
        description={`All ${data.length} SAT skills for ${studentId}, sorted by practice priority (highest at top). Bar length = mastery %; color = observed from items vs predicted from k-NN + propagation.`}
      />
      <Card className="border-2 border-brand-gold/35 shadow-md">
        <CardHeader className="pb-2">
          <CardTitle className="text-h2">Where to focus next — full picture</CardTitle>
          <p className="text-body-sm text-muted-foreground">
            {testedCount} tested · {untestedCount} predicted · priority = (100 − mastery) ×
            foundational weight
          </p>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto rounded-lg border bg-card/50">
            <ResponsiveContainer
              width="100%"
              height={chartHeight}
              aria-label={`Complete skill gap landscape for ${studentId}`}
            >
              <BarChart
                data={data}
                layout="vertical"
                margin={{ top: 8, right: 32, left: 8, bottom: 8 }}
                barCategoryGap={4}
              >
                <CartesianGrid {...chartGridProps} horizontal={false} />
                <XAxis
                  type="number"
                  domain={[0, 100]}
                  tickFormatter={(v) => `${v}%`}
                  label={{
                    value: "Mastery %",
                    position: "insideBottom",
                    offset: -4,
                    style: { fill: "hsl(var(--chart-axis))", fontSize: 12 },
                  }}
                  {...chartAxisProps}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={200}
                  stroke={chartAxisProps.stroke}
                  tick={{ fill: chartAxisProps.tick.fill, fontSize: 10, fontFamily: chartAxisProps.tick.fontFamily }}
                  tickLine={chartAxisProps.tickLine}
                  axisLine={chartAxisProps.axisLine}
                />
                <Tooltip content={<LandscapeTooltip />} />
                <Legend
                  verticalAlign="top"
                  align="right"
                  wrapperStyle={{ paddingBottom: 12, fontSize: 12 }}
                  payload={[
                    { value: "Tested (observed from items)", type: "square", color: TESTED_FILL },
                    { value: "Untested (predicted)", type: "square", color: PREDICTED_FILL },
                  ]}
                />
                <Bar dataKey="mastery" name="Mastery %" radius={[0, 4, 4, 0]} maxBarSize={18}>
                  {data.map((entry) => (
                    <Cell
                      key={entry.rank}
                      fill={entry.isTested ? TESTED_FILL : PREDICTED_FILL}
                      fillOpacity={entry.isTested ? 1 : 0.9}
                      stroke={entry.rank === 1 ? TOP_PRIORITY_STROKE : undefined}
                      strokeWidth={entry.rank === 1 ? 2 : 0}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-meta text-muted-foreground mt-4 text-center">
            #1 priority skill has a gold outline — scroll this chart during your conclusion to
            show how observed gaps and predicted gaps combine into one ranked plan.
          </p>
        </CardContent>
      </Card>
    </section>
  );
}
