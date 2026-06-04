"""Synthetic MCQ item bank — 3 items per skill (216 items total)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from skills import SKILLS, Category, Level, Skill

Difficulty = Literal["easy", "medium", "hard"]

DIFFICULTY_OFFSET = {"easy": 0.12, "medium": 0.0, "hard": -0.12}


@dataclass(frozen=True)
class Item:
    item_id: str
    skill_id: str
    stem: str
    choice_a: str
    choice_b: str
    choice_c: str
    choice_d: str
    correct_choice: str
    difficulty: Difficulty


def _stem_for_skill(skill: Skill, variant: int) -> tuple[str, str, str, str, str, str]:
    """Return stem, A, B, C, D, correct for a skill variant."""
    cat = skill.category
    name = skill.skill_name
    v = variant + 1

    if cat == "Grammar":
        stems = [
            (
                f"Which revision best fixes the {name.lower()} error? "
                f'"The team of researchers were publishing their findings."',
                "The team of researchers were publishing",
                "The team of researchers was publishing",
                "The team of researchers are publishing",
                "The team of researchers have been publishing",
                "B",
            ),
            (
                f"Select the sentence that demonstrates correct {name.lower()}.",
                "Neither the students nor the teacher were ready.",
                "Neither the students nor the teacher was ready.",
                "Neither the students or the teacher was ready.",
                "Neither the students nor the teacher is ready.",
                "B",
            ),
            (
                f"Which option corrects the issue related to {name.lower()}?",
                "Running quickly, the finish line approached.",
                "Running quickly, she approached the finish line.",
                "Running quickly, the finish line was approached.",
                "Running quickly, approaching the finish line.",
                "B",
            ),
        ]
    elif cat == "Vocabulary":
        stems = [
            (
                f'In context, "{name.lower()}" — the word "scrupulous" most nearly means:',
                "careless",
                "meticulous",
                "hostile",
                "ambiguous",
                "B",
            ),
            (
                f"Which word best completes the sentence for {name.lower()}? "
                '"The author\'s ___ word choice conveyed skepticism."',
                "neutral",
                "derisive",
                "literal",
                "redundant",
                "B",
            ),
            (
                f"Which synonym best fits {name.lower()} in an academic passage?",
                "trivial",
                "nuanced",
                "obsolete",
                "uniform",
                "B",
            ),
        ]
    elif cat == "Evidence":
        stems = [
            (
                f"Which line best supports the claim for {name.lower()}?",
                "The experiment failed immediately.",
                "Results showed a 40% increase in retention.",
                "Scientists enjoy collaborative work.",
                "The lab opened in 1998.",
                "B",
            ),
            (
                f"For {name.lower()}, which evidence is most relevant to the author's argument?",
                "A personal anecdote about childhood",
                "A cited study with sample size and methodology",
                "An unrelated historical date",
                "A rhetorical question",
                "B",
            ),
            (
                f"Which choice best demonstrates {name.lower()}?",
                "The data contradicts the headline claim.",
                "The passage uses colorful metaphors.",
                "The author mentions a fictional character.",
                "The title includes a pun.",
                "A",
            ),
        ]
    elif cat == "Writing":
        stems = [
            (
                f"Which revision best improves {name.lower()}?",
                "The report was long. It was also confusing.",
                "The report was long and confusing.",
                "The report, it was long and confusing.",
                "Long and confusing, the report it was.",
                "B",
            ),
            (
                f"Select the option that best achieves {name.lower()}.",
                "However the results were mixed.",
                "However, the results were mixed.",
                "However; the results were mixed.",
                "However the results, were mixed.",
                "B",
            ),
            (
                f"Which sentence best demonstrates {name.lower()}?",
                "Things happened. Then other things.",
                "First, the team analyzed data; then they drafted conclusions.",
                "Data analysis conclusions drafting team.",
                "The team did stuff and then more stuff.",
                "B",
            ),
        ]
    elif cat == "Rhetoric":
        stems = [
            (
                f"Which choice best identifies {name.lower()} in the passage?",
                "The author lists unrelated facts.",
                "The author appeals to shared values to persuade skeptics.",
                "The author copies another text verbatim.",
                "The author avoids stating any claim.",
                "B",
            ),
            (
                f"For {name.lower()}, what is the primary rhetorical move?",
                "Shifting tone to undermine the opponent",
                "Using statistical evidence to establish credibility",
                "Ignoring the audience entirely",
                "Repeating the title word-for-word",
                "B",
            ),
            (
                f"Which option best analyzes {name.lower()}?",
                "The paragraph summarizes the introduction only.",
                "The paragraph contrasts prior views with the author's thesis.",
                "The paragraph defines a single vocabulary word.",
                "The paragraph quotes dialogue without comment.",
                "B",
            ),
        ]
    else:  # Reading
        stems = [
            (
                f"Which answer best reflects {name.lower()}?",
                "The passage focuses on a minor detail in paragraph four.",
                "The passage argues that urban gardens improve community health.",
                "The passage is primarily a biography of one scientist.",
                "The passage rejects all scientific methods.",
                "B",
            ),
            (
                f"Based on the text, which inference aligns with {name.lower()}?",
                "The author distrusts all new technology.",
                "The author implies that policy change requires public pressure.",
                "The author believes history is irrelevant.",
                "The author refuses to take a position.",
                "B",
            ),
            (
                f"Which statement demonstrates {name.lower()}?",
                "The author describes a setting without purpose.",
                "The author uses a counterexample to qualify a general claim.",
                "The author lists random dates.",
                "The author copies a dictionary definition.",
                "B",
            ),
        ]

    idx = variant % len(stems)
    s, a, b, c, d, correct = stems[idx]
    return s, a, b, c, d, correct


def build_item_bank() -> list[Item]:
    items: list[Item] = []
    difficulties: list[Difficulty] = ["easy", "medium", "hard"]

    for skill in SKILLS:
        for v, diff in enumerate(difficulties):
            stem, a, b, c, d, correct = _stem_for_skill(skill, v)
            num = skill.skill_id.replace("SK", "")
            items.append(
                Item(
                    item_id=f"Q{num}-{v + 1}",
                    skill_id=skill.skill_id,
                    stem=stem,
                    choice_a=a,
                    choice_b=b,
                    choice_c=c,
                    choice_d=d,
                    correct_choice=correct,
                    difficulty=diff,
                )
            )
    return items


ITEMS: list[Item] = build_item_bank()

assert len(ITEMS) == 72 * 3


def items_by_skill() -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = {}
    for item in ITEMS:
        out.setdefault(item.skill_id, []).append(item)
    return out


SKILL_ITEMS = items_by_skill()


def item_bank_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item_id": i.item_id,
                "skill_id": i.skill_id,
                "stem": i.stem,
                "choice_a": i.choice_a,
                "choice_b": i.choice_b,
                "choice_c": i.choice_c,
                "choice_d": i.choice_d,
                "correct_choice": i.correct_choice,
                "difficulty": i.difficulty,
            }
            for i in ITEMS
        ]
    )


def write_item_bank(path: Path) -> pd.DataFrame:
    df = item_bank_df()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
