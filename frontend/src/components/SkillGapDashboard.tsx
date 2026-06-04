import { useEffect, useMemo, useState } from "react";
import {
  BarChart3,
  ClipboardList,
  Lightbulb,
  Target,
  Users,
} from "lucide-react";
import { SyntheticDataBanner } from "@/components/SyntheticDataBanner";
import { RecommendationsTable } from "@/components/RecommendationsTable";
import { TopSkillsChart } from "@/components/TopSkillsChart";
import { PeerSimilarityChart } from "@/components/PeerSimilarityChart";
import { TestedSkillsWork } from "@/components/TestedSkillsWork";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { PageHeading, SectionHeading, StatCard } from "@/components/ui/section-heading";
import type { DashboardData } from "@/types";

export function SkillGapDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [studentId, setStudentId] = useState<string>("");

  useEffect(() => {
    fetch("/data/dashboard.json")
      .then((r) => {
        if (!r.ok) throw new Error("Missing dashboard.json — run: python predict.py --export-frontend");
        return r.json();
      })
      .then((json: DashboardData) => {
        setData(json);
        setStudentId(json.defaultStudentId);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const selectableStudents = useMemo(() => {
    if (!data) return [];
    return data.students.filter((s) => data.byStudent[s.id]);
  }, [data]);

  const studentData = data?.byStudent[studentId];

  if (error) {
    return (
      <div className="min-h-screen bg-background p-8">
        <Card className="max-w-lg mx-auto p-6">
          <p className="text-body text-destructive">{error}</p>
        </Card>
      </div>
    );
  }

  if (!data || !studentData) {
    return (
      <div className="min-h-screen bg-background p-8 space-y-4 max-w-5xl mx-auto">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  const topPeer = studentData.peers[0];
  const observedWork = studentData.observedWork ?? [];

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-6xl px-4 py-8 md:px-8 space-y-8">
        <SyntheticDataBanner />

        <PageHeading
          eyebrow="Skill Gap Prediction"
          title={`Recommendations for ${studentId}`}
          description="Skill scores are computed from item responses (% correct). Cosine similarity on partial skill vectors predicts untested gaps."
          icon={<Target className="h-8 w-8 text-brand-gold" />}
          action={
            <Select value={studentId} onValueChange={setStudentId}>
              <SelectTrigger aria-label="Select student">
                <SelectValue placeholder="Select student" />
              </SelectTrigger>
              <SelectContent>
                {selectableStudents.map((s) => (
                  <SelectItem key={s.id} value={s.id}>
                    {s.label} ({s.observedCount} skills tested)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          }
        />

        <section className="space-y-3">
          <SectionHeading
            title="Tested skills & item responses"
            description="Expand any skill to see MCQ attempts. Mastery % = correct ÷ items attempted."
          />
          <TestedSkillsWork work={observedWork} />
        </section>

        <div className="grid gap-4 md:grid-cols-3">
          <StatCard
            icon={<BarChart3 className="h-5 w-5" />}
            label="Skills tested"
            value={`${studentData.summary.observedSkills} / ${studentData.summary.observedSkills + studentData.summary.untestedSkills}`}
            sublabel={`${studentData.summary.untestedSkills} skills with no items attempted`}
            tone="primary"
          />
          <StatCard
            icon={<ClipboardList className="h-5 w-5" />}
            label="Items attempted"
            value={studentData.summary.totalItemsAttempted ?? "—"}
            sublabel={
              studentData.summary.avgItemsPerSkill
                ? `${studentData.summary.avgItemsPerSkill} avg items per tested skill`
                : undefined
            }
            tone="default"
          />
          <StatCard
            icon={<Users className="h-5 w-5" />}
            label="Nearest peer (skill sim.)"
            value={topPeer ? topPeer.similarity.toFixed(3) : "—"}
            sublabel={
              topPeer?.itemSimilarity
                ? `${topPeer.studentId} · item sim. ${topPeer.itemSimilarity.toFixed(3)}`
                : topPeer?.studentId
            }
            tone="accent"
          />
        </div>

        <StatCard
          icon={<Lightbulb className="h-5 w-5" />}
          label="Top priority gap score"
          value={studentData.summary.topPriorityScore.toFixed(1)}
          sublabel="Highest (100 − predicted mastery) × foundational weight among untested skills"
          tone="warning"
          className="max-w-md"
        />

        <Card className="border-l-4 border-l-brand-gold">
          <CardContent className="p-6">
            <SectionHeading
              title="Interpretation"
              description={studentData.summary.interpretation}
            />
          </CardContent>
        </Card>

        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-3">
            <SectionHeading
              title="Priority recommendations (untested skills)"
              description="Ranked by predicted weakness × foundational weight"
            />
            <RecommendationsTable rows={studentData.recommendations} />
          </div>
          <TopSkillsChart rows={studentData.recommendations} />
        </div>

        <PeerSimilarityChart peers={studentData.peers} />

        <p className="text-meta text-center pb-4">
          Linear algebra: response matrix R (students × items) aggregates to skill matrix S via Q;
          cosine similarity on S uses dot products and norms; predictions are weighted peer averages.
        </p>
      </div>
    </div>
  );
}
