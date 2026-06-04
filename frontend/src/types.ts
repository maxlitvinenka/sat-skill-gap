export interface Recommendation {
  rank: number;
  skillId: string;
  skillName: string;
  category: string;
  level: string;
  predictedMastery: number;
  foundationalWeight: number;
  priorityScore: number;
  reason: string;
}

export interface Peer {
  studentId: string;
  similarity: number;
}

export interface StudentSummary {
  observedSkills: number;
  untestedSkills: number;
  topPriorityScore: number;
  interpretation: string;
}

export interface StudentDashboard {
  summary: StudentSummary;
  recommendations: Recommendation[];
  peers: Peer[];
}

export interface StudentMeta {
  id: string;
  label: string;
  observedCount: number;
  hiddenCount: number;
}

export interface DashboardData {
  meta: {
    generatedAt: string;
    isSynthetic: boolean;
    algorithm: string;
    nStudents: number;
    nSkills: number;
  };
  students: StudentMeta[];
  defaultStudentId: string;
  byStudent: Record<string, StudentDashboard>;
}
