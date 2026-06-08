import { useCallback, useEffect, useState } from "react";
import { ChevronDown, ChevronRight, Loader2, Play, RefreshCw, UserPlus, Zap } from "lucide-react";
import { analyzeLive, fetchSkills, regeneratePipeline } from "@/api/client";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { AnalyzeResponse, DashboardData, SkillMeta, StudentDashboard, StudentMeta } from "@/types";

type CustomSkillRow = { skillId: string; mastery: number };

export function LiveShowcase({
  live,
  studentId,
  meta,
  onUpdate,
}: {
  live: boolean;
  studentId: string;
  meta: DashboardData["meta"];
  onUpdate: (payload: {
    studentId: string;
    meta: DashboardData["meta"];
    students: StudentMeta[];
    dashboard: StudentDashboard;
  }) => void;
}) {
  const [open, setOpen] = useState(true);
  const [seed, setSeed] = useState(meta.randomSeed ?? 42);
  const [sigma, setSigma] = useState(meta.kernelSigma ?? 25);
  const [alpha, setAlpha] = useState(meta.propagationAlpha ?? 0.7);
  const [k, setK] = useState(meta.kNeighbors ?? 15);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [skills, setSkills] = useState<SkillMeta[]>([]);
  const [customName, setCustomName] = useState("Jordan");
  const [customRows, setCustomRows] = useState<CustomSkillRow[]>([
    { skillId: "SK001", mastery: 45 },
    { skillId: "SK002", mastery: 35 },
    { skillId: "SK003", mastery: 55 },
  ]);
  const [skillPicker, setSkillPicker] = useState("SK004");

  useEffect(() => {
    if (!live) return;
    fetchSkills().then(setSkills).catch(() => setSkills([]));
  }, [live]);

  const applyResponse = useCallback(
    (res: AnalyzeResponse) => {
      onUpdate({
        studentId: res.studentId,
        meta: res.meta,
        students: res.students,
        dashboard: res.dashboard,
      });
    },
    [onUpdate],
  );

  const runAnalyze = useCallback(
    async (overrides?: { customStudent?: { name: string; observedSkills: CustomSkillRow[] } }) => {
      if (!live) return;
      setLoading(true);
      setError(null);
      try {
        const res = await analyzeLive({
          studentId: overrides?.customStudent ? undefined : studentId,
          seed,
          sigma,
          alpha,
          k,
          customStudent: overrides?.customStudent
            ? {
                name: overrides.customStudent.name,
                observedSkills: overrides.customStudent.observedSkills.map((r) => ({
                  skillId: r.skillId,
                  mastery: r.mastery,
                })),
              }
            : undefined,
        });
        applyResponse(res);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Analysis failed");
      } finally {
        setLoading(false);
      }
    },
    [live, studentId, seed, sigma, alpha, k, applyResponse],
  );

  const regenerate = async () => {
    if (!live) return;
    setLoading(true);
    setError(null);
    try {
      const pipe = await regeneratePipeline(seed);
      const res = await analyzeLive({
        studentId: pipe.defaultStudentId,
        seed,
        sigma,
        alpha,
        k,
      });
      applyResponse(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Regeneration failed");
    } finally {
      setLoading(false);
    }
  };

  const addCustomSkill = () => {
    if (customRows.some((r) => r.skillId === skillPicker)) return;
    setCustomRows([...customRows, { skillId: skillPicker, mastery: 50 }]);
  };

  const runCustom = () => {
    if (customRows.length < 5) {
      setError("Add at least 5 tested skills for reliable kernel overlap.");
      return;
    }
    if (customRows.length < 10) {
      setError(null);
    }
    void runAnalyze({
      customStudent: { name: customName.trim(), observedSkills: customRows },
    });
  };

  if (!live) return null;

  return (
    <Card className="border-2 border-emerald-600/40">
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
          <CardTitle className="text-h2 flex items-center gap-2">
            <Zap className="h-5 w-5 text-emerald-600" />
            Live showcase
          </CardTitle>
          <Badge variant="outline" className="ml-auto border-emerald-600/50 text-emerald-700">
            Python API
          </Badge>
        </button>
        <p className="text-body-sm text-muted-foreground pl-7">
          Tune parameters or add a custom student — results run through real{" "}
          <code className="text-meta">lib.py</code> on the server.
        </p>
      </CardHeader>
      {open && (
        <CardContent className="space-y-6 pt-2">
          {error && (
            <p className="text-body-sm text-destructive rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2">
              {error}
            </p>
          )}

          <section className="space-y-3">
            <h3 className="text-body font-semibold">Algorithm controls</h3>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <label className="space-y-1 text-body-sm">
                <span className="text-muted-foreground">Random seed</span>
                <input
                  type="number"
                  className="w-full rounded-md border bg-background px-3 py-2"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                />
              </label>
              <label className="space-y-1 text-body-sm">
                <span className="text-muted-foreground">σ (kernel)</span>
                <input
                  type="range"
                  min={5}
                  max={60}
                  step={1}
                  value={sigma}
                  onChange={(e) => setSigma(Number(e.target.value))}
                  className="w-full"
                />
                <span className="tabular-nums">{sigma}</span>
              </label>
              <label className="space-y-1 text-body-sm">
                <span className="text-muted-foreground">α (blend)</span>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={alpha}
                  onChange={(e) => setAlpha(Number(e.target.value))}
                  className="w-full"
                />
                <span className="tabular-nums">{alpha.toFixed(2)}</span>
              </label>
              <label className="space-y-1 text-body-sm">
                <span className="text-muted-foreground">k (neighbors)</span>
                <input
                  type="range"
                  min={3}
                  max={30}
                  step={1}
                  value={k}
                  onChange={(e) => setK(Number(e.target.value))}
                  className="w-full"
                />
                <span className="tabular-nums">{k}</span>
              </label>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button type="button" onClick={() => void runAnalyze()} disabled={loading}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                Run analysis
              </Button>
              <Button type="button" variant="outline" onClick={() => void regenerate()} disabled={loading}>
                <RefreshCw className="h-4 w-4" />
                Regenerate cohort
              </Button>
            </div>
          </section>

          <section className="space-y-3 border-t pt-4">
            <h3 className="text-body font-semibold flex items-center gap-2">
              <UserPlus className="h-4 w-4" />
              Custom student
            </h3>
            <p className="text-body-sm text-muted-foreground">
              Enter a name and tested skills with mastery %. Recommended: 10+ skills for stable k-NN.
            </p>
            <div className="flex flex-wrap gap-2 items-end">
              <label className="space-y-1 text-body-sm">
                <span className="text-muted-foreground">Name</span>
                <input
                  className="rounded-md border bg-background px-3 py-2 block"
                  value={customName}
                  onChange={(e) => setCustomName(e.target.value)}
                />
              </label>
              <label className="space-y-1 text-body-sm flex-1 min-w-[200px]">
                <span className="text-muted-foreground">Add skill</span>
                <select
                  className="w-full rounded-md border bg-background px-3 py-2"
                  value={skillPicker}
                  onChange={(e) => setSkillPicker(e.target.value)}
                >
                  {skills.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name} ({s.category})
                    </option>
                  ))}
                </select>
              </label>
              <Button type="button" variant="outline" size="sm" onClick={addCustomSkill}>
                Add skill
              </Button>
            </div>
            <div className="rounded-md border overflow-hidden max-h-48 overflow-y-auto">
              <table className="w-full text-body-sm">
                <thead className="bg-muted/40 text-meta">
                  <tr>
                    <th className="text-left px-3 py-2">Skill</th>
                    <th className="text-right px-3 py-2 w-28">Mastery %</th>
                    <th className="w-10" />
                  </tr>
                </thead>
                <tbody>
                  {customRows.map((row) => {
                    const skill = skills.find((s) => s.id === row.skillId);
                    return (
                      <tr key={row.skillId} className="border-t">
                        <td className="px-3 py-2">{skill?.name ?? row.skillId}</td>
                        <td className="px-3 py-2">
                          <input
                            type="number"
                            min={0}
                            max={100}
                            className="w-full rounded border px-2 py-1 text-right"
                            value={row.mastery}
                            onChange={(e) =>
                              setCustomRows(
                                customRows.map((r) =>
                                  r.skillId === row.skillId
                                    ? { ...r, mastery: Number(e.target.value) }
                                    : r,
                                ),
                              )
                            }
                          />
                        </td>
                        <td className="px-1">
                          <button
                            type="button"
                            className="text-muted-foreground hover:text-destructive text-xs"
                            onClick={() =>
                              setCustomRows(customRows.filter((r) => r.skillId !== row.skillId))
                            }
                          >
                            ×
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <Button type="button" onClick={runCustom} disabled={loading || customRows.length < 5}>
              Add &amp; analyze custom student
            </Button>
          </section>

        </CardContent>
      )}
    </Card>
  );
}
