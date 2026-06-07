import { useState } from "react";
import { ChevronDown, ChevronRight, ExternalLink } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MathFormula } from "@/components/MathFormula";
import type { DashboardData, MethodologyDemo } from "@/types";
import { cn } from "@/lib/utils";

function LiveExample({ children }: { children: React.ReactNode }) {
  return (
    <div className="border-l-4 border-l-brand-gold bg-brand-gold/5 rounded-r-md px-4 py-3 text-body-sm space-y-2">
      <p className="text-meta font-medium uppercase tracking-wide">Live example for this student</p>
      {children}
    </div>
  );
}

function SeeBelow({ href }: { href: string }) {
  return (
    <a
      href={href}
      className="inline-flex items-center gap-1 text-body-sm text-primary hover:underline mt-2"
    >
      See this on the page below
      <ExternalLink className="h-3 w-3" />
    </a>
  );
}

function Step({
  number,
  title,
  description,
  formula,
  live,
  seeBelow,
}: {
  number: number;
  title: string;
  description: string;
  formula?: React.ReactNode;
  live?: React.ReactNode;
  seeBelow?: string;
}) {
  return (
    <div className="flex gap-4">
      <div
        className="shrink-0 w-8 h-8 rounded-full bg-brand-gold/20 text-brand-gold flex items-center justify-center font-semibold text-sm"
        aria-hidden
      >
        {number}
      </div>
      <div className="flex-1 min-w-0 space-y-2 pb-6 border-b last:border-b-0 last:pb-0">
        <h3 className="text-h2 text-base font-semibold">{title}</h3>
        <p className="text-body-sm text-muted-foreground">{description}</p>
        {formula}
        {live}
        {seeBelow && <SeeBelow href={seeBelow} />}
      </div>
    </div>
  );
}

export function MethodologyWalkthrough({
  meta,
  demo,
  studentId,
}: {
  meta: DashboardData["meta"];
  demo?: MethodologyDemo;
  studentId: string;
}) {
  const [open, setOpen] = useState(true);
  const sigma = demo?.similarityExample?.sigma ?? meta.kernelSigma ?? 25;
  const alpha = demo?.similarityExample?.alpha ?? meta.propagationAlpha ?? 0.7;
  const k = meta.kNeighbors ?? 15;
  return (
    <Card className="border-2 border-brand-gold/30">
      <CardHeader className="pb-2">
        <button
          type="button"
          className="flex w-full items-center gap-2 text-left"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
        >
          {open ? (
            <ChevronDown className="h-5 w-5 shrink-0 text-brand-gold" />
          ) : (
            <ChevronRight className="h-5 w-5 shrink-0 text-brand-gold" />
          )}
          <CardTitle className="text-h2">How this data is built</CardTitle>
        </button>
        <p className="text-body-sm text-muted-foreground pl-7">
          These numbers are not typed into the frontend. They are computed offline in Python and
          exported to <code className="text-meta">dashboard.json</code>. Selecting a student
          updates the live examples.
        </p>
      </CardHeader>
      {open && (
        <CardContent className="space-y-2 pt-2">
          {!demo && (
            <p className="text-body-sm text-muted-foreground">
              Run <code className="text-meta">python predict.py --export-frontend</code> to load
              methodology examples.
            </p>
          )}

          <Step
            number={1}
            title="Where the initial data comes from"
            description="The showcase uses synthetic SAT-style data only — no real students. Python scripts generate an item bank, simulate MCQ responses for 100 students (~40 tested skills each), and write CSV files before this page loads."
            live={
              <LiveExample>
                <p>
                  <strong>{meta.nStudents}</strong> students · <strong>{meta.nSkills}</strong> skills
                  · <strong>{meta.nItems ?? 216}</strong> MCQ items
                </p>
                <p>Random seed: <strong>{demo?.randomSeed ?? meta.randomSeed ?? 42}</strong> (reproducible)</p>
                <p>Generated: {new Date(meta.generatedAt).toLocaleString()}</p>
                <p className="text-meta">Source files: {(demo?.dataFiles ?? meta.dataFiles ?? []).join(", ")}</p>
              </LiveExample>
            }
          />

          <Step
            number={2}
            title="Item responses → response matrix R"
            description={`Each student attempts ~3 questions per tested skill. Matrix R has shape ${demo?.matrixShapes.R ?? "100×216"}: row = one student, column = one item. Entry is 1 (correct), 0 (wrong), or missing if never attempted.`}
            formula={
              <MathFormula latex={`R_{i,k} \\in \\{0,\\,1,\\,\\text{NaN}\\} \\quad \\text{(student } i \\text{, item } k \\text{)}`} />
            }
            live={
              demo && (
                <LiveExample>
                  <p>
                    <strong>{studentId}</strong> attempted{" "}
                    <strong>{demo.studentVector.totalItemsAttempted}</strong> items total.
                  </p>
                </LiveExample>
              )
            }
            seeBelow="#tested-skills"
          />

          <Step
            number={3}
            title="Aggregate to skill matrix S (matrix multiplication view)"
            description="Each item maps to exactly one skill via matrix Q. Skill mastery is the percentage of items correct for that skill — not a random number."
            formula={
              <MathFormula
                latex={`S_{i,j} = 100 \\cdot \\frac{\\#\\text{ correct items for skill } j}{\\#\\text{ attempted items for skill } j}

\\text{Conceptually:} \\quad S = \\text{normalize}(R \\cdot Q)`}
              />
            }
            live={
              demo?.skillAggregationExample && (
                <LiveExample>
                  <p>
                    <strong>{demo.skillAggregationExample.skillName}</strong>:{" "}
                    {demo.skillAggregationExample.correct}/{demo.skillAggregationExample.attempted}{" "}
                    correct → <strong>{demo.skillAggregationExample.mastery}%</strong>
                  </p>
                  <MathFormula
                    display={false}
                    latex={`100 \\cdot \\frac{${demo.skillAggregationExample.correct}}{${demo.skillAggregationExample.attempted}} = ${demo.skillAggregationExample.mastery}\\%`}
                  />
                </LiveExample>
              )
            }
            seeBelow="#tested-skills"
          />

          <Step
            number={4}
            title="Each student is a vector in ℝ⁷²"
            description="Row i of S is a 72-dimensional skill vector. Coordinates are known only for tested skills (~40); untested skills are missing (not filled with random values)."
            formula={
              <MathFormula latex={`\\mathbf{s}_i = (S_{i,1},\\, S_{i,2},\\, \\ldots,\\, S_{i,72}) \\in \\mathbb{R}^{72}`} />
            }
            live={
              demo && (
                <LiveExample>
                  <p>
                    <strong>{studentId}</strong>: {demo.studentVector.observedCount} known coordinates,{" "}
                    {demo.studentVector.untestedCount} missing (untested skills).
                  </p>
                </LiveExample>
              )
            }
          />

          <Step
            number={5}
            title="Gaussian kernel on shared skills (distance + norm)"
            description="To compare two students, we use only skills both have been tested on (set Ω). Euclidean distance ‖u−v‖² is computed via dot product on the difference vector, then mapped to a kernel similarity."
            formula={
              <MathFormula
                latex={`\\|u - v\\|^2 = (u - v)^\\top (u - v) = \\sum_{k \\in \\Omega} (u_k - v_k)^2

K(u, v) = \\exp\\!\\left(-\\frac{\\|u - v\\|^2}{2\\sigma^2}\\right), \\quad \\sigma = ${sigma}`}
              />
            }
            live={
              demo?.similarityExample && (
                <LiveExample>
                  <p>
                    Nearest peer: <strong>{demo.similarityExample.peerStudentId}</strong> (
                    {demo.similarityExample.sharedSkills} shared tested skills)
                  </p>
                  <MathFormula
                    display={false}
                    latex={`\\|u - v\\|^2 = ${demo.similarityExample.squaredDistance}`}
                  />
                  <MathFormula
                    display={false}
                    latex={`K(u,v) = \\exp\\!\\left(-\\frac{${demo.similarityExample.squaredDistance}}{2 \\cdot ${demo.similarityExample.sigma}^2}\\right) = ${demo.similarityExample.kernelValue}`}
                  />
                </LiveExample>
              )
            }
            seeBelow="#peer-similarity"
          />

          <Step
            number={6}
            title="k-NN weighted prediction (collaborative)"
            description={`For each missing skill j, take the top-${k} peers who were tested on j. Weight their scores on j by Gaussian kernel similarity to the target student.`}
            formula={
              <MathFormula
                latex={`\\widehat{y}^{\\text{neighbor}}_j = \\frac{\\sum_i K_i \\cdot \\text{peer\\_score}_{i,j}}{\\sum_i K_i}

\\text{Higher kernel weight } K_i \\Rightarrow \\text{that peer's score counts more.}`}
              />
            }
            live={
              demo?.predictionExample && (
                <LiveExample>
                  <p>
                    Top untested gap: <strong>{demo.predictionExample.skillName}</strong>
                  </p>
                  {demo.predictionExample.neighborPrediction != null && (
                    <p>
                      k-NN neighbor prediction: <strong>{demo.predictionExample.neighborPrediction}%</strong>{" "}
                      (from {demo.predictionExample.peersWithSkill} peers who took this skill)
                    </p>
                  )}
                </LiveExample>
              )
            }
            seeBelow="#recommendations"
          />

          <Step
            number={7}
            title="Label propagation from related skills"
            description="A skill affinity graph W connects prerequisites, same-category skills, and adjacent difficulty levels. Missing skills borrow signal from the target's known related skills."
            formula={
              <MathFormula
                latex={`\\widehat{y}^{\\text{related}}_j = \\frac{\\sum_{j'} W_{j,j'} \\cdot \\text{known\\_score}_{j'}}{\\sum_{j'} W_{j,j'}}

\\widehat{y}^{\\text{final}}_j = \\alpha \\cdot \\widehat{y}^{\\text{neighbor}}_j + (1 - \\alpha) \\cdot \\widehat{y}^{\\text{related}}_j, \\quad \\alpha = ${alpha}`}
              />
            }
            live={
              demo?.predictionExample && (
                <LiveExample>
                  {demo.predictionExample.relatedPrediction != null && (
                    <p>
                      Related-skill propagation: <strong>{demo.predictionExample.relatedPrediction}%</strong>
                    </p>
                  )}
                  <p>
                    Final blended prediction: <strong>{demo.predictionExample.predictedMastery}%</strong>
                  </p>
                </LiveExample>
              )
            }
            seeBelow="#recommendations"
          />

          <Step
            number={8}
            title="Priority ranking (what to practice next)"
            description="Low predicted mastery on foundational skills ranks highest. Basic skills have larger foundational weights."
            formula={
              <MathFormula latex={`\\text{priority} = (100 - \\widehat{y}^{\\text{final}}) \\times w_{\\text{foundational}}`} />
            }
            live={
              demo?.predictionExample && (
                <LiveExample>
                  <p>{demo.predictionExample.skillName}:</p>
                  <MathFormula
                    display={false}
                    latex={`(100 - ${Math.round(demo.predictionExample.predictedMastery)}) \\times ${demo.predictionExample.foundationalWeight.toFixed(2)} = ${demo.predictionExample.priorityScore.toFixed(1)}`}
                  />
                </LiveExample>
              )
            }
            seeBelow="#recommendations"
          />

          {demo && (
            <p className={cn("text-body-sm text-muted-foreground pt-2 border-t")}>
              {demo.notRandomNote}
            </p>
          )}
        </CardContent>
      )}
    </Card>
  );
}
