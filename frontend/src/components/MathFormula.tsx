import katex from "katex";
import "katex/dist/katex.min.css";
import { cn } from "@/lib/utils";

type MathFormulaProps = {
  latex: string;
  display?: boolean;
  className?: string;
};

export function MathFormula({ latex, display = true, className }: MathFormulaProps) {
  const html = katex.renderToString(latex, {
    throwOnError: false,
    displayMode: display,
    trust: false,
  });

  if (display) {
    return (
      <div
        className={cn(
          "rounded-md border border-border/60 bg-muted/25 px-4 py-3 overflow-x-auto",
          "[&_.katex]:text-[1.05rem] [&_.katex-display]:my-0",
          className,
        )}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }

  return (
    <span
      className={cn("inline [&_.katex]:text-[0.95em]", className)}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
