import { useCallback, useEffect, useMemo, useState } from "react";
import {
  BarChart3,
  ClipboardList,
  Lightbulb,
  Target,
  Users,
} from "lucide-react";
import { analyzeLive, checkHealth } from "@/api/client";
import { SyntheticDataBanner } from "@/components/SyntheticDataBanner";
import { LiveShowcase } from "@/components/LiveShowcase";
import { MethodologyWalkthrough } from "@/components/MethodologyWalkthrough";
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
import type { DashboardData, StudentDashboard, StudentMeta } from "@/types";

export function SkillGapDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [studentId, setStudentId] = useState<string>("");
  const [liveMode, setLiveMode] = useState(false);
  const [studentData, setStudentData] = useState<StudentDashboard | null>(null);
  const [students, setStudents] = useState<StudentMeta[]>([]);
  const [meta, setMeta] = useState<DashboardData["meta"] | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      const apiUp = await checkHealth();
      if (cancelled) return;

      if (apiUp) {
        setLiveMode(true);
        try {
          const res = await analyzeLive({});
          if (cancelled) return;
          setMeta(res.meta);
          setStudents(res.students);
          setStudentId(res.studentId);
          setStudentData(res.dashboard);
          setData({
            meta: res.meta,
            students: res.students,
            defaultStudentId: res.studentId,
            byStudent: { [res.studentId]: res.dashboard },
          });
        } catch (e) {
          setError(e instanceof Error ? e.message : "Live API failed");
        }
        return;
      }

      fetch(`${import.meta.env.BASE_URL}data/dashboard.json`)
        .then((r) => {
          if (!r.ok) throw new Error("Missing dashboard.json — run: python predict.py --export-frontend");
          return r.json();
        })
        .then((json: DashboardData) => {
          if (cancelled) return;
          setData(json);
          setMeta(json.meta);
          setStudents(json.students);
          setStudentId(json.defaultStudentId);
          setStudentData(json.byStudent[json.defaultStudentId] ?? null);
        })
        .catch((e: Error) => {
          if (!cancelled) setError(e.message);
        });
    }

    void init();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleLiveUpdate = useCallback(
    (payload: {
      studentId: string;
      meta: DashboardData["meta"];
      students: StudentMeta[];
      dashboard: StudentDashboard;
    }) => {
      setMeta(payload.meta);
      setStudents(payload.students);
      setStudentId(payload.studentId);
      setStudentData(payload.dashboard);
      setData((prev) =>
        prev
          ? {
              ...prev,
              meta: payload.meta,
              students: payload.students,
              defaultStudentId: payload.studentId,
              byStudent: { ...prev.byStudent, [payload.studentId]: payload.dashboard },
            }
          : {
              meta: payload.meta,
              students: payload.students,
              defaultStudentId: payload.studentId,
              byStudent: { [payload.studentId]: payload.dashboard },
            },
      );
    },
    [],
  );

  const handleStudentChange = useCallback(
    async (id: string) => {
      setStudentId(id);
      if (liveMode) {
        const cached = data?.byStudent[id];
        if (cached) {
          setStudentData(cached);
          return;
        }
        try {
          const res = await analyzeLive({
            studentId: id,
            seed: meta?.randomSeed,
            sigma: meta?.kernelSigma,
            alpha: meta?.propagationAlpha,
            k: meta?.kNeighbors,
          });
          handleLiveUpdate({
            studentId: res.studentId,
            meta: res.meta,
            students: res.students,
            dashboard: res.dashboard,
          });
        } catch (e) {
          setError(e instanceof Error ? e.message : "Failed to load student");
        }
      } else {
        setStudentData(data?.byStudent[id] ?? null);
      }
    },
    [liveMode, data, meta, handleLiveUpdate],
  );

  const selectableStudents = useMemo(() => {
    return students.filter((s) => liveMode || data?.byStudent[s.id]);
  }, [students, data, liveMode]);

  if (error) {
    return (
      <div className="min-h-screen bg-background p-8">
        <Card className="max-w-lg mx-auto p-6">
          <p className="text-body text-destructive">{error}</p>
        </Card>
      </div>
    );
  }

  if (!meta || !studentData) {
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
        <SyntheticDataBanner liveMode={liveMode} />

        <PageHeading
          eyebrow="Skill Gap Prediction"
          title={`Recommendations for ${studentId}`}
          description="Skill scores are computed from item responses (% correct). Gaussian kernel k-NN plus skill-neighborhood label propagation predicts untested gaps."
          icon={<Target className="h-8 w-8 text-brand-gold" />}
          action={
            <Select value={studentId} onValueChange={(id) => void handleStudentChange(id)}>
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

        <LiveShowcase
          live={liveMode}
          studentId={studentId}
          meta={meta}
          onUpdate={handleLiveUpdate}
        />

        <MethodologyWalkthrough
          meta={meta}
          demo={studentData.methodologyDemo}
          studentId={studentId}
        />

        <section id="tested-skills" className="space-y-3 scroll-mt-8">
          <SectionHeading
            title="Tested skills & item responses"
            description={
              studentData.summary.isCustomInput
                ? "Custom input — mastery % entered live (no MCQ item drill-down)."
                : "Expand any skill to see MCQ attempts. Mastery % = correct ÷ items attempted."
            }
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
            label="Nearest peer (kernel sim.)"
            value={topPeer ? topPeer.similarity.toFixed(3) : "—"}
            sublabel={topPeer?.studentId}
            tone="accent"
          />
        </div>

        <StatCard
          icon={<Lightbulb className="h-5 w-5" />}
          label="Top priority gap score"
          value={studentData.summary.topPriorityScore.toFixed(1)}
          sublabel="Highest (100 − mastery) × foundational weight across all skills"
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

        <section id="recommendations" className="space-y-6 scroll-mt-8">
          <div className="space-y-3">
            <SectionHeading
              title="Priority recommendations (all skills)"
              description="Tested skills use observed mastery; untested use predicted mastery. Ranked by (100 − score) × foundational weight."
            />
            <RecommendationsTable rows={studentData.recommendations} />
          </div>
          <TopSkillsChart rows={studentData.recommendations} />
        </section>

        <div id="peer-similarity" className="scroll-mt-8">
          <PeerSimilarityChart peers={studentData.peers} />
        </div>
      </div>
    </div>
  );
}
