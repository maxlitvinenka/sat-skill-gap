export interface ResponseItem {
  itemId: string;
  stem: string;
  chosen: string;
  correctChoice: string;
  isCorrect: boolean;
  choiceA: string;
  choiceB: string;
  choiceC: string;
  choiceD: string;
}

export interface ObservedSkillWork {
  skillId: string;
  skillName: string;
  category: string;
  attempted: number;
  correct: number;
  mastery: number;
  items: ResponseItem[];
}

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
  itemSimilarity?: number;
}

export interface StudentSummary {
  observedSkills: number;
  untestedSkills: number;
  totalItemsAttempted?: number;
  avgItemsPerSkill?: number;
  topPriorityScore: number;
  interpretation: string;
}

export interface StudentDashboard {
  summary: StudentSummary;
  observedWork: ObservedSkillWork[];
  recommendations: Recommendation[];
  peers: Peer[];
}

export interface StudentMeta {
  id: string;
  label: string;
  observedCount: number;
  hiddenCount: number;
  itemCount?: number;
}

export interface DashboardData {
  meta: {
    generatedAt: string;
    isSynthetic: boolean;
    algorithm: string;
    scoreSource?: string;
    nStudents: number;
    nSkills: number;
    nItems?: number;
  };
  students: StudentMeta[];
  defaultStudentId: string;
  byStudent: Record<string, StudentDashboard>;
}
