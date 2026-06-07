import { useState } from "react";
import { ChevronDown, ChevronRight, ExternalLink } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DashboardData, MethodologyDemo } from "@/types";
import { cn } from "@/lib/utils";

function Formula({ children }: { children: React.ReactNode }) {
  return (
    <pre className="text-body-sm font-mono bg-muted/40 border rounded-md p-3 overflow-x-auto whitespace-pre-wrap">
      {children}
    </pre>
  );
}

function LiveExample({ children }: { children: React.ReactNode }) {
  return (
    <div className="border-l-4 border-l-brand-gold bg-brand-gold/5 rounded-r-md px-4 py-3 text-body-sm space-y-1">
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
          <CardTitle className="text-h2">
            How this data is built (linear algebra step-by-step)
          </CardTitle>
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
              <Formula>{`R[i,k] ∈ {0, 1, NaN}   (student i, item k)`}</Formula>
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
              <Formula>{`S[i,j] = 100 × (# correct items for skill j) / (# attempted items for skill j)

Conceptually:  S = normalize(R · Q)
Q maps items → skills (${demo?.matrixShapes.Q ?? "216×72"})`}</Formula>
            }
            live={
              demo?.skillAggregationExample && (
                <LiveExample>
                  <p>
                    <strong>{demo.skillAggregationExample.skillName}</strong>:{" "}
                    {demo.skillAggregationExample.correct}/{demo.skillAggregationExample.attempted}{" "}
                    correct → <strong>{demo.skillAggregationExample.mastery}%</strong>
                  </p>
                  <p className="font-mono text-meta">{demo.skillAggregationExample.formula}</p>
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
              <Formula>{`s_i = (S[i,1], S[i,2], …, S[i,72]) ∈ ℝ^72`}</Formula>
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
            description="To compare two students, we use only skills both have been tested on (set Ω). Euclidean distance ||u−v||² is computed via dot product on the difference vector, then mapped to a kernel similarity."
            formula={
              <Formula>{`||u − v||² = (u − v)ᵀ(u − v) = Σ (u_k − v_k)²   on shared skills Ω

K(u, v) = exp(−||u − v||² / (2σ²))     σ = ${sigma}`}</Formula>
            }
            live={
              demo?.similarityExample && (
                <LiveExample>
                  <p>
                    Nearest peer: <strong>{demo.similarityExample.peerStudentId}</strong> (
                    {demo.similarityExample.sharedSkills} shared tested skills)
                  </p>
                  <p className="font-mono text-meta">
                    ||u−v||² = {demo.similarityExample.squaredDistance}
                  </p>
                  <p>
                    K(u,v) = exp(−{demo.similarityExample.squaredDistance} / (2×{demo.similarityExample.sigma}²)) ={" "}
                    <strong>{demo.similarityExample.kernelValue}</strong>
                  </p>
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
              <Formula>{`neighbor_pred_j = Σ (K_i × peer_score_i) / Σ K_i

Higher kernel weight → that peer's score counts more.`}</Formula>
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
              <Formula>{`related_pred_j = Σ W[j,j′] × known_score_j′ / Σ W[j,j′]

final_j = α × neighbor_pred + (1 − α) × related_pred     α = ${alpha}`}</Formula>
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
              <Formula>{`priority = (100 − predicted) × foundational_weight`}</Formula>
            }
            live={
              demo?.predictionExample && (
                <LiveExample>
                  <p>
                    {demo.predictionExample.skillName}:{" "}
                    <strong>{demo.predictionExample.priorityFormula}</strong>
                  </p>
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
