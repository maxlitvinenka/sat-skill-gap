import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { MathFormula } from "@/components/MathFormula";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ComputationStep } from "@/types";

export function ComputationTrace({
  steps,
  studentId,
  live,
}: {
  steps: ComputationStep[];
  studentId: string;
  live?: boolean;
}) {
  const [open, setOpen] = useState(true);

  if (steps.length === 0) return null;

  return (
    <Card id="computation-trace" className="scroll-mt-8 border-2 border-emerald-600/30">
      <CardHeader className="pb-2">
        <button
          type="button"
          className="flex w-full items-center gap-2 text-left"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
        >
          {open ? (
            <ChevronDown className="h-5 w-5 shrink-0 text-emerald-600" />
          ) : (
            <ChevronRight className="h-5 w-5 shrink-0 text-emerald-600" />
          )}
          <CardTitle className="text-h2">Computation trace</CardTitle>
        </button>
        <p className="text-body-sm text-muted-foreground pl-7">
          {live ? "Live run — " : ""}
          Step-by-step provenance for <strong>{studentId}</strong>: where each mastery %,
          kernel weight, neighbor contribution, and priority score comes from.
        </p>
      </CardHeader>
      {open && (
        <CardContent className="space-y-4 pt-2">
          {steps.map((step) => (
            <div key={step.step} className="rounded-md border bg-muted/20 p-4 space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline">Step {step.step}</Badge>
                <span className="font-medium text-body-sm">{step.title}</span>
                <code className="text-meta ml-auto">{step.function}</code>
              </div>
              {step.explanation && (
                <p className="text-body-sm leading-relaxed">{step.explanation}</p>
              )}
              <p className="text-meta font-mono">{step.codeRef}</p>
              {step.details && step.details.length > 0 && (
                <dl className="grid gap-1.5 sm:grid-cols-2 text-body-sm rounded-md border bg-background/60 px-3 py-2">
                  {step.details.map((d) => (
                    <div key={d.label} className="flex gap-2 justify-between sm:block">
                      <dt className="text-muted-foreground shrink-0">{d.label}</dt>
                      <dd className="font-medium tabular-nums text-right sm:text-left">
                        {d.value}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}
              {step.table && step.table.rows.length > 0 && (
                <div className="overflow-x-auto rounded-md border">
                  <table className="w-full text-body-sm">
                    <thead className="bg-muted/50 text-meta">
                      <tr>
                        {step.table.headers.map((h) => (
                          <th key={h} className="text-left px-3 py-2 font-medium">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {step.table.rows.map((row, i) => (
                        <tr key={i} className="border-t">
                          {step.table!.headers.map((h) => (
                            <td key={h} className="px-3 py-2 tabular-nums">
                              {row[h] ?? "—"}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <MathFormula latex={step.latex} />
              <p className="text-body-sm">
                Result: <strong className="text-brand-gold">{step.output}</strong>
              </p>
            </div>
          ))}
        </CardContent>
      )}
    </Card>
  );
}
