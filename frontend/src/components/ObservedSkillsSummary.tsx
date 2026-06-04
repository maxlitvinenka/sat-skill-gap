import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ObservedSkillWork } from "@/types";
import { categoryBadgeClass, cn, masteryColor } from "@/lib/utils";

export function ObservedSkillsSummary({ work }: { work: ObservedSkillWork[] }) {
  if (work.length === 0) {
    return <p className="text-body-sm text-muted-foreground">No tested skills for this student.</p>;
  }

  return (
    <div className="rounded-lg border bg-card overflow-auto">
      <Table aria-label="Tested skills summary from item responses">
        <TableHeader>
          <TableRow>
            <TableHead>Skill</TableHead>
            <TableHead>Category</TableHead>
            <TableHead>Items</TableHead>
            <TableHead>Correct</TableHead>
            <TableHead>Mastery</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {work.map((row) => (
            <TableRow key={row.skillId}>
              <TableCell className="font-medium max-w-[200px]">{row.skillName}</TableCell>
              <TableCell>
                <Badge variant="outline" className={cn(categoryBadgeClass(row.category))}>
                  {row.category}
                </Badge>
              </TableCell>
              <TableCell className="tabular-nums">{row.attempted}</TableCell>
              <TableCell className="tabular-nums">
                {row.correct}/{row.attempted}
              </TableCell>
              <TableCell className={cn("tabular-nums font-semibold", masteryColor(row.mastery))}>
                {row.mastery.toFixed(0)}%
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
