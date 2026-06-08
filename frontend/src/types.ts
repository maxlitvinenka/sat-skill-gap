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
  isCustomInput?: boolean;
  items: ResponseItem[];
}

export interface Recommendation {
  rank: number;
  skillId: string;
  skillName: string;
  category: string;
  level: string;
  isTested?: boolean;
  predictedMastery: number;
  neighborPred?: number | null;
  relatedPred?: number | null;
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
  totalItemsAttempted?: number;
  avgItemsPerSkill?: number;
  topPriorityScore: number;
  interpretation: string;
  isCustomInput?: boolean;
}

export interface SkillMeta {
  id: string;
  name: string;
  category: string;
  level: string;
}

export interface TraceDetailRow {
  label: string;
  value: string;
}

export interface TraceTable {
  headers: string[];
  rows: Record<string, string | number>[];
}

export interface ComputationStep {
  step: number;
  function: string;
  codeRef: string;
  title: string;
  explanation?: string;
  details?: TraceDetailRow[];
  table?: TraceTable | null;
  inputs: Record<string, unknown>;
  output: string | number;
  latex: string;
}

export interface AnalyzeRequest {
  studentId?: string;
  seed?: number;
  sigma?: number;
  alpha?: number;
  k?: number;
  customStudent?: {
    name: string;
    observedSkills: { skillId: string; mastery: number }[];
  };
}

export interface AnalyzeResponse {
  studentId: string;
  meta: DashboardData["meta"];
  students: StudentMeta[];
  dashboard: StudentDashboard;
}

export interface MethodologyDemo {
  randomSeed: number;
  matrixShapes: { R: string; Q: string; S: string };
  dataFiles: string[];
  notRandomNote: string;
  skillAggregationExample: {
    skillName: string;
    correct: number;
    attempted: number;
    mastery: number;
    formula: string;
  } | null;
  studentVector: {
    dimension: number;
    observedCount: number;
    untestedCount: number;
    totalItemsAttempted: number;
  };
  similarityExample: {
    peerStudentId: string;
    sharedSkills: number;
    squaredDistance: number;
    meanSquaredDistance?: number;
    kernelValue: number;
    sigma: number;
    alpha: number;
  } | null;
  predictionExample: {
    skillName: string;
    predictedMastery: number;
    neighborPrediction?: number | null;
    relatedPrediction?: number | null;
    foundationalWeight: number;
    priorityScore: number;
    peersWithSkill: number;
    priorityFormula: string;
  } | null;
}

export interface StudentDashboard {
  summary: StudentSummary;
  observedWork: ObservedSkillWork[];
  recommendations: Recommendation[];
  peers: Peer[];
  methodologyDemo?: MethodologyDemo;
  computationTrace?: ComputationStep[];
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
    randomSeed?: number;
    nStudents: number;
    nSkills: number;
    nItems?: number;
    dataFiles?: string[];
    pipelineSteps?: string[];
    kernelSigma?: number;
    kNeighbors?: number;
    propagationAlpha?: number;
  };
  students: StudentMeta[];
  defaultStudentId: string;
  byStudent: Record<string, StudentDashboard>;
}
