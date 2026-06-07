import type {
  AnalyzeRequest,
  AnalyzeResponse,
  DashboardData,
  SkillMeta,
  StudentMeta,
} from "@/types";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = (body as { detail?: string }).detail ?? res.statusText;
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const data = await apiFetch<{ ok: boolean }>("/api/health");
    return data.ok;
  } catch {
    return false;
  }
}

export async function fetchSkills(): Promise<SkillMeta[]> {
  return apiFetch<SkillMeta[]>("/api/skills");
}

export async function regeneratePipeline(seed: number): Promise<{
  meta: DashboardData["meta"];
  students: StudentMeta[];
  defaultStudentId: string;
}> {
  return apiFetch("/api/pipeline", {
    method: "POST",
    body: JSON.stringify({ seed }),
  });
}

export async function analyzeLive(body: AnalyzeRequest): Promise<AnalyzeResponse> {
  return apiFetch<AnalyzeResponse>("/api/analyze", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
