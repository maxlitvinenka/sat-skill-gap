import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Recommendation } from "@/types";
import { categoryBadgeClass, cn, masteryColor } from "@/lib/utils";

export function RecommendationsTable({ rows }: { rows: Recommendation[] }) {
  const top = rows.slice(0, 15);

  return (
    <div className="rounded-lg border bg-card">
      <Table aria-label="Priority skill recommendations">
        <TableHeader>
          <TableRow>
            <TableHead>Rank</TableHead>
            <TableHead>Skill</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Category</TableHead>
            <TableHead>Level</TableHead>
            <TableHead>Score</TableHead>
            <TableHead>Weight</TableHead>
            <TableHead>Priority</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {top.map((row) => (
            <TableRow key={row.skillId}>
              <TableCell className="font-medium">{row.rank}</TableCell>
              <TableCell>
                <div className="font-medium max-w-[220px]">{row.skillName}</div>
                <div className="text-meta mt-0.5 max-w-[280px]">{row.reason}</div>
              </TableCell>
              <TableCell>
                <Badge
                  variant="outline"
                  className={cn(
                    row.isTested
                      ? "border-blue-500/40 text-blue-700 bg-blue-50"
                      : "border-amber-500/40 text-amber-800 bg-amber-50",
                  )}
                >
                  {row.isTested ? "Tested" : "Untested"}
                </Badge>
              </TableCell>
              <TableCell>
                <Badge variant="outline" className={cn(categoryBadgeClass(row.category))}>
                  {row.category}
                </Badge>
              </TableCell>
              <TableCell>{row.level}</TableCell>
              <TableCell className={cn("tabular-nums font-medium", masteryColor(row.predictedMastery))}>
                {row.predictedMastery.toFixed(0)}%
              </TableCell>
              <TableCell className="tabular-nums">{row.foundationalWeight.toFixed(1)}</TableCell>
              <TableCell className="tabular-nums font-semibold text-brand-gold">
                {row.priorityScore.toFixed(1)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
