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
import type { Peer } from "@/types";

export function PeerSimilarityChart({ peers }: { peers: Peer[] }) {
  const data = peers.map((p) => ({
    name: p.studentId,
    similarity: p.similarity,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle>Nearest similar students</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={280} aria-label="Peer cosine similarity chart">
          <BarChart data={data} margin={{ bottom: 8 }}>
            <CartesianGrid {...chartGridProps} />
            <XAxis dataKey="name" {...chartAxisProps} />
            <YAxis domain={[0, 1]} {...chartAxisProps} />
            <Tooltip
              content={
                <ReadingRoomsTooltip valueFormatter={(v) => v.toFixed(3)} />
              }
            />
            <Bar dataKey="similarity" name="Cosine similarity" fill={chartPalette[1]} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
