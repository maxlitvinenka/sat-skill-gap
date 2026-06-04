import { useState } from "react";
import { CheckCircle2, ChevronDown, ChevronRight, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ObservedSkillWork, ResponseItem } from "@/types";
import { cn, masteryColor } from "@/lib/utils";

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
    <div className="border rounded-md p-3 space-y-2 bg-background/50">
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

function SkillPanel({ skill }: { skill: ObservedSkillWork }) {
  const [open, setOpen] = useState(false);

  return (
    <Card>
      <CardHeader className="py-4">
        <button
          type="button"
          className="flex w-full items-center gap-3 text-left hover-elevate rounded-md -m-2 p-2"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
        >
          {open ? (
            <ChevronDown className="h-4 w-4 shrink-0" />
          ) : (
            <ChevronRight className="h-4 w-4 shrink-0" />
          )}
          <div className="flex-1 min-w-0">
            <CardTitle className="text-base">{skill.skillName}</CardTitle>
            <p className="text-meta mt-0.5">
              {skill.correct}/{skill.attempted} items correct
            </p>
          </div>
          <Badge variant="outline" className={cn("shrink-0", masteryColor(skill.mastery))}>
            {skill.mastery.toFixed(0)}%
          </Badge>
        </button>
      </CardHeader>
      {open && (
        <CardContent className="space-y-3 pt-0">
          {skill.items.map((item) => (
            <ItemRow key={item.itemId} item={item} />
          ))}
        </CardContent>
      )}
    </Card>
  );
}

export function SkillResponsePanel({ work }: { work: ObservedSkillWork[] }) {
  return (
    <div className="space-y-3">
      {work.map((skill) => (
        <SkillPanel key={skill.skillId} skill={skill} />
      ))}
    </div>
  );
}
