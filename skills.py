"""Self-contained SAT Reading/Writing skill catalog (72 skills, synthetic project only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Category = Literal["Reading", "Writing", "Grammar", "Evidence", "Vocabulary", "Rhetoric"]
Level = Literal["Basic", "Intermediate", "Advanced"]


@dataclass(frozen=True)
class Skill:
    skill_id: str
    skill_name: str
    category: Category
    level: Level
    foundational_weight: float
    prerequisites: tuple[str, ...]


def _weight(level: Level, is_root: bool) -> float:
    base = {"Basic": 1.0, "Intermediate": 0.75, "Advanced": 0.45}[level]
    return min(1.0, base + (0.1 if is_root else 0.0))


def _chain(
    start: int,
    names: list[str],
    category: Category,
    levels: list[Level],
    prereq_offset: int = 1,
) -> list[Skill]:
    skills: list[Skill] = []
    for i, (name, level) in enumerate(zip(names, levels)):
        sid = f"SK{start + i:03d}"
        prereqs: tuple[str, ...] = ()
        if i > 0:
            prereqs = (f"SK{start + i - prereq_offset:03d}",)
        skills.append(
            Skill(
                skill_id=sid,
                skill_name=name,
                category=category,
                level=level,
                foundational_weight=_weight(level, is_root=i == 0),
                prerequisites=prereqs,
            )
        )
    return skills


# Reading (18): SK001–SK018
READING = _chain(
    1,
    [
        "Identify Central Claim / Thesis",
        "Locate Explicit Details",
        "Determine Paragraph Purpose",
        "Summarize Passage Main Points",
        "Understand Author's Tone",
        "Track Argument Development",
        "Compare Viewpoints Within Passage",
        "Draw Logical Inferences",
        "Resolve Conflicting Information",
        "Predict Author Response",
        "Interpret Figurative Language",
        "Analyze Cause and Effect",
        "Evaluate Argument Strength",
        "Synthesize Multi-Paragraph Ideas",
        "Identify Assumptions",
        "Analyze Chronological Structure",
        "Evaluate Counterarguments",
        "Integrate Complex Passage Themes",
    ],
    "Reading",
    ["Basic"] * 4 + ["Intermediate"] * 8 + ["Advanced"] * 6,
)

# Evidence (12): SK019–SK030
EVIDENCE = _chain(
    19,
    [
        "Identify Textual Evidence",
        "Match Evidence to Claim",
        "Evaluate Evidence Relevance",
        "Command of Evidence (Textual)",
        "Compare Quantitative and Textual Data",
        "Interpret Charts in Context",
        "Evaluate Data Support for Claims",
        "Identify Evidence Gaps",
        "Strengthen Arguments with Evidence",
        "Weaken Arguments via Counterevidence",
        "Synthesize Cross-Source Evidence",
        "Evaluate Statistical Claims in Passages",
    ],
    "Evidence",
    ["Basic"] * 3 + ["Intermediate"] * 5 + ["Advanced"] * 4,
)

# Grammar (15): SK031–SK045
GRAMMAR = _chain(
    31,
    [
        "Subject-Verb Agreement",
        "Pronoun-Antecedent Agreement",
        "Verb Tense Consistency",
        "Sentence Boundaries (Fragments/Run-ons)",
        "Comma Usage with Clauses",
        "Apostrophe and Possessives",
        "Modifier Placement",
        "Parallel Structure",
        "Punctuation with Introductory Elements",
        "Semicolon and Colon Usage",
        "Dangling and Misplaced Modifiers",
        "Idiom and Preposition Usage",
        "Comparative and Superlative Forms",
        "Subjunctive Mood",
        "Complex Sentence Revision",
    ],
    "Grammar",
    ["Basic"] * 5 + ["Intermediate"] * 6 + ["Advanced"] * 4,
)

# Vocabulary (10): SK046–SK055
VOCABULARY = _chain(
    46,
    [
        "Words in Context (Basic)",
        "Determine Word Meaning from Context",
        "Distinguish Denotation and Connotation",
        "Analyze Word Choice Effects",
        "Precision in Word Selection",
        "Domain-Specific Vocabulary",
        "Multiple-Meaning Words",
        "Tone-Shifting Vocabulary",
        "Academic Vocabulary in Arguments",
        "Nuanced Synonym Selection",
    ],
    "Vocabulary",
    ["Basic"] * 3 + ["Intermediate"] * 4 + ["Advanced"] * 3,
)

# Writing (10): SK056–SK065
WRITING = _chain(
    56,
    [
        "Sentence Combination",
        "Concision and Clarity",
        "Logical Transitions",
        "Organization of Ideas",
        "Effective Introductions",
        "Effective Conclusions",
        "Development with Supporting Details",
        "Coherence Across Paragraphs",
        "Style and Formality",
        "Global Revision for Purpose",
    ],
    "Writing",
    ["Basic"] * 3 + ["Intermediate"] * 4 + ["Advanced"] * 3,
)

# Rhetoric (7): SK066–SK072
RHETORIC = _chain(
    66,
    [
        "Identify Rhetorical Situation",
        "Analyze Author's Purpose",
        "Evaluate Persuasive Techniques",
        "Analyze Structure and Function",
        "Compare Rhetorical Strategies",
        "Evaluate Audience Awareness",
        "Synthesize Rhetorical Analysis",
    ],
    "Rhetoric",
    ["Basic"] * 2 + ["Intermediate"] * 3 + ["Advanced"] * 2,
)

SKILLS: list[Skill] = READING + EVIDENCE + GRAMMAR + VOCABULARY + WRITING + RHETORIC

assert len(SKILLS) == 72

LATENT_DIMS = ("grammar", "reading", "inference", "vocabulary", "rhetoric")

CATEGORY_LATENT_LOADINGS: dict[Category, dict[str, float]] = {
    "Reading": {"reading": 0.55, "inference": 0.35, "rhetoric": 0.1},
    "Evidence": {"reading": 0.35, "inference": 0.45, "rhetoric": 0.2},
    "Grammar": {"grammar": 0.75, "reading": 0.15, "vocabulary": 0.1},
    "Vocabulary": {"vocabulary": 0.65, "reading": 0.25, "rhetoric": 0.1},
    "Writing": {"rhetoric": 0.4, "grammar": 0.35, "reading": 0.25},
    "Rhetoric": {"rhetoric": 0.55, "inference": 0.25, "reading": 0.2},
}

LEVEL_INTERCEPT: dict[Level, float] = {
    "Basic": 58.0,
    "Intermediate": 52.0,
    "Advanced": 46.0,
}
