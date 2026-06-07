import { useState } from "react";
import { CheckCircle2, ChevronDown, ChevronRight, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { ObservedSkillWork, ResponseItem } from "@/types";
import { categoryBadgeClass, cn, masteryColor } from "@/lib/utils";

function ItemRow({ item }: { item: ResponseItem }) {
  const choiceLabel = (key: string, text: string) => (
    <div
      className={cn(
        "text-body-sm py-1 px-2 rounded",
        key === item.correctChoice && "bg-emerald-50 font-medium",
        key === item.chosen && !item.isCorrect && "bg-red-50 line-through",
        key === item.chosen && item.isCorrect && "bg-emerald-100 font-medium",
      )}
    >
      <span className="font-mono mr-2">{key}.</span>
      {text}
    </div>
  );

  return (
    <div className="border rounded-md p-3 space-y-2 bg-background/80">
      <div className="flex items-start gap-2">
        {item.isCorrect ? (
          <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" aria-label="Correct" />
        ) : (
          <XCircle className="h-4 w-4 text-red-600 shrink-0 mt-0.5" aria-label="Incorrect" />
        )}
        <p className="text-body-sm">{item.stem}</p>
      </div>
      <div className="grid gap-1 pl-6">
        {choiceLabel("A", item.choiceA)}
        {choiceLabel("B", item.choiceB)}
        {choiceLabel("C", item.choiceC)}
        {choiceLabel("D", item.choiceD)}
      </div>
      <p className="text-meta pl-6">
        Student chose <strong>{item.chosen}</strong>
        {!item.isCorrect && (
          <> — correct answer was <strong>{item.correctChoice}</strong></>
        )}
      </p>
    </div>
  );
}

function SkillRow({ skill, defaultOpen }: { skill: ObservedSkillWork; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen ?? false);

  return (
    <div className="border-b last:border-b-0">
      <button
        type="button"
        className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-muted/40 transition-colors"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        {open ? (
          <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
        )}
        <span className="flex-1 min-w-0 font-medium text-body-sm truncate">{skill.skillName}</span>
        <Badge
          variant="outline"
          className={cn("hidden sm:inline-flex shrink-0", categoryBadgeClass(skill.category))}
        >
          {skill.category}
        </Badge>
        <span className="text-meta tabular-nums shrink-0 w-14 text-right hidden md:inline">
          {skill.correct}/{skill.attempted}
        </span>
        <span
          className={cn(
            "text-body-sm font-semibold tabular-nums shrink-0 w-12 text-right",
            masteryColor(skill.mastery),
          )}
        >
          {skill.mastery.toFixed(0)}%
        </span>
      </button>
      {open && (
        <div className="px-4 pb-4 pt-0 pl-11 space-y-3 bg-muted/20">
          <p className="text-meta sm:hidden">
            {skill.category} · {skill.correct}/{skill.attempted} correct
          </p>
          {skill.isCustomInput || skill.items.length === 0 ? (
            <p className="text-body-sm text-muted-foreground italic">
              Custom live input — mastery entered directly ({skill.correct}/{skill.attempted}{" "}
              equivalent).
            </p>
          ) : (
            skill.items.map((item) => <ItemRow key={item.itemId} item={item} />)
          )}
        </div>
      )}
    </div>
  );
}

export function TestedSkillsWork({ work }: { work: ObservedSkillWork[] }) {
  if (work.length === 0) {
    return (
      <p className="text-body-sm text-muted-foreground rounded-lg border bg-card p-4">
        No tested skills for this student.
      </p>
    );
  }

  return (
    <div className="rounded-lg border bg-card overflow-hidden">
      <div
        className="hidden md:grid grid-cols-[1fr_auto_auto_auto] gap-3 px-4 py-2 border-b bg-muted/30 text-meta font-medium"
        aria-hidden
      >
        <span className="pl-7">Skill</span>
        <span className="w-24">Category</span>
        <span className="w-14 text-right">Score</span>
        <span className="w-12 text-right">%</span>
      </div>
      <div role="list" aria-label="Tested skills and item responses">
        {work.map((skill, i) => (
          <div key={skill.skillId} role="listitem">
            <SkillRow skill={skill} defaultOpen={i === 0} />
          </div>
        ))}
      </div>
    </div>
  );
}
