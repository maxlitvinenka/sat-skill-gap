import { useEffect, useMemo, useState } from "react";
import {
  BarChart3,
  Lightbulb,
  Target,
  Users,
} from "lucide-react";
import { SyntheticDataBanner } from "@/components/SyntheticDataBanner";
import { RecommendationsTable } from "@/components/RecommendationsTable";
import { TopSkillsChart } from "@/components/TopSkillsChart";
import { PeerSimilarityChart } from "@/components/PeerSimilarityChart";
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

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-6xl px-4 py-8 md:px-8 space-y-8">
        <SyntheticDataBanner />

        <PageHeading
          eyebrow="Skill Gap Prediction"
          title={`Recommendations for ${studentId}`}
          description="Cosine similarity on partial skill vectors predicts untested foundational gaps."
          icon={<Target className="h-8 w-8 text-brand-gold" />}
          action={
            <Select value={studentId} onValueChange={setStudentId}>
              <SelectTrigger aria-label="Select student">
                <SelectValue placeholder="Select student" />
              </SelectTrigger>
              <SelectContent>
                {selectableStudents.map((s) => (
                  <SelectItem key={s.id} value={s.id}>
                    {s.label} ({s.observedCount} tested)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          }
        />

        <div className="grid gap-4 md:grid-cols-3">
          <StatCard
            icon={<BarChart3 className="h-5 w-5" />}
            label="Skills coverage"
            value={`${studentData.summary.observedSkills} / ${studentData.summary.observedSkills + studentData.summary.untestedSkills}`}
            sublabel={`${studentData.summary.untestedSkills} untested skills`}
            tone="primary"
          />
          <StatCard
            icon={<Lightbulb className="h-5 w-5" />}
            label="Top priority score"
            value={studentData.summary.topPriorityScore.toFixed(1)}
            sublabel="Higher = more urgent foundational gap"
            tone="accent"
          />
          <StatCard
            icon={<Users className="h-5 w-5" />}
            label="Nearest peer similarity"
            value={topPeer ? topPeer.similarity.toFixed(3) : "—"}
            sublabel={topPeer ? topPeer.studentId : "No similar peers"}
            tone="default"
          />
        </div>

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
              title="Priority recommendations"
              description="Untested skills ranked by (100 − predicted mastery) × foundational weight"
            />
            <RecommendationsTable rows={studentData.recommendations} />
          </div>
          <TopSkillsChart rows={studentData.recommendations} />
        </div>

        <PeerSimilarityChart peers={studentData.peers} />

        <p className="text-meta text-center pb-4">
          Linear algebra: students are vectors in ℝ⁷²; cosine similarity uses dot products and norms;
          missing scores are weighted averages from similar peers.
        </p>
      </div>
    </div>
  );
}
