"""Core linear-algebra pipeline for SAT skill gap prediction (synthetic data only)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from items import DIFFICULTY_OFFSET, ITEMS, SKILL_ITEMS, Item, write_item_bank
from skills import (
    CATEGORY_LATENT_LOADINGS,
    LATENT_DIMS,
    LEVEL_INTERCEPT,
    SKILLS,
    Skill,
)

N_STUDENTS = 100
N_SKILLS = 72
N_KNOWN_TARGET = 40
MASK_FRACTION = 1 - N_KNOWN_TARGET / N_SKILLS
RANDOM_SEED = 42
K_NEIGHBORS = 15
KERNEL_SIGMA = 25.0
PROPAGATION_ALPHA = 0.7
ITEMS_PER_SKILL = 3
CHOICES = ("A", "B", "C", "D")
LEVEL_INDEX = {"Basic": 0, "Intermediate": 1, "Advanced": 2}

_NAME_POOL = (
    "Aaliyah", "Aaron", "Abigail", "Adam", "Adrian", "Aiden", "Alex", "Alexis", "Alice", "Amanda",
    "Amir", "Amy", "Ana", "Andre", "Angela", "Anna", "Anthony", "Ariana", "Arthur", "Ashley",
    "Austin", "Ava", "Benjamin", "Blake", "Brandon", "Brian", "Brooke", "Caleb", "Cameron", "Carlos",
    "Carmen", "Carter", "Charlotte", "Chloe", "Chris", "Claire", "Cole", "Daniel", "David", "Diana",
    "Diego", "Dylan", "Eden", "Elena", "Eli", "Elizabeth", "Ella", "Emily", "Emma", "Ethan",
    "Eva", "Evan", "Faith", "Felix", "Fiona", "Gabriel", "Grace", "Hannah", "Harper", "Henry",
    "Ian", "Isaac", "Isabella", "Ivan", "Jack", "Jade", "James", "Jasmine", "Jason", "Jenna",
    "Jessica", "Jordan", "Jose", "Joshua", "Julia", "Julian", "Justin", "Kai", "Karen", "Kate",
    "Kevin", "Kim", "Laura", "Leah", "Leo", "Lily", "Logan", "Lucas", "Luis", "Luna",
    "Madison", "Maria", "Mark", "Mason", "Maya", "Mia", "Michael", "Michelle", "Miles", "Naomi",
    "Nathan", "Nicholas", "Nicole", "Noah", "Nora", "Oliver", "Olivia", "Oscar", "Owen", "Paige",
    "Patrick", "Paul", "Peter", "Rachel", "Rebecca", "Riley", "Robert", "Rosa", "Ryan", "Samantha",
    "Samuel", "Sara", "Sarah", "Sebastian", "Sofia", "Sophia", "Stefan", "Stella", "Stephen", "Susan",
    "Taylor", "Thomas", "Timothy", "Tyler", "Vanessa", "Victor", "Victoria", "Vincent", "Vivian", "William",
    "Zoe", "Zoey", "Zachary", "Wyatt", "Wesley", "Walter", "Valerie", "Tristan", "Travis", "Tony",
)


def assign_student_names(rng: np.random.Generator, n: int = N_STUDENTS) -> list[str]:
    """Deterministic shuffle of synthetic first names (seeded with pipeline RNG)."""
    if n > len(_NAME_POOL):
        raise ValueError(f"Need {n} names but pool has {len(_NAME_POOL)}")
    names = list(_NAME_POOL)
    rng.shuffle(names)
    return names[:n]


@dataclass
class PipelineResult:
    scores: np.ndarray
    mask: np.ndarray
    student_ids: list[str]
    skill_ids: list[str]
    skills: list[Skill]
    latent_profiles: np.ndarray
    default_student_idx: int
    responses_df: pd.DataFrame
    item_ids: list[str]
    response_matrix: np.ndarray
    skill_item_matrix: np.ndarray
    skill_affinity: np.ndarray


def skill_metadata_df() -> pd.DataFrame:
    rows = []
    for s in SKILLS:
        rows.append(
            {
                "skill_id": s.skill_id,
                "skill_name": s.skill_name,
                "category": s.category,
                "level": s.level,
                "foundational_weight": s.foundational_weight,
                "prerequisites": "|".join(s.prerequisites),
            }
        )
    return pd.DataFrame(rows)


def write_skill_metadata(path: Path) -> pd.DataFrame:
    df = skill_metadata_df()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _archetype_latent(rng: np.random.Generator) -> np.ndarray:
    archetypes = [
        {"grammar": 0.85, "reading": 0.45, "inference": 0.40, "vocabulary": 0.55, "rhetoric": 0.50},
        {"grammar": 0.50, "reading": 0.80, "inference": 0.75, "vocabulary": 0.70, "rhetoric": 0.65},
        {"grammar": 0.70, "reading": 0.72, "inference": 0.35, "vocabulary": 0.68, "rhetoric": 0.60},
        {"grammar": 0.35, "reading": 0.38, "inference": 0.32, "vocabulary": 0.40, "rhetoric": 0.35},
        {"grammar": 0.55, "reading": 0.78, "inference": 0.62, "vocabulary": 0.50, "rhetoric": 0.42},
    ]
    base = archetypes[int(rng.integers(0, len(archetypes)))]
    vec = np.array([base[d] + rng.normal(0, 0.08) for d in LATENT_DIMS], dtype=float)
    return np.clip(vec, 0.15, 0.95)


def _p_correct(item: Item, skill: Skill, latent: np.ndarray) -> float:
    loadings = CATEGORY_LATENT_LOADINGS[skill.category]
    weighted = sum(latent[LATENT_DIMS.index(k)] * w for k, w in loadings.items())
    level_adj = LEVEL_INTERCEPT[skill.level] / 100.0
    diff_adj = DIFFICULTY_OFFSET[item.difficulty]
    logit = -0.5 + 3.2 * (0.6 * weighted + 0.25 * level_adj + diff_adj)
    return float(1.0 / (1.0 + np.exp(-logit)))


def _pick_wrong_choice(correct: str, rng: np.random.Generator) -> str:
    wrong = [c for c in CHOICES if c != correct]
    return str(rng.choice(wrong))


def _assign_tested_skills(rng: np.random.Generator, n_skills: int) -> np.ndarray:
    """Which skill indices each student is tested on (~40 skills)."""
    tested = np.zeros((N_STUDENTS, n_skills), dtype=bool)
    for i in range(N_STUDENTS):
        n_test = N_KNOWN_TARGET + int(rng.integers(-2, 3))
        n_test = max(38, min(42, n_test))
        n_test = min(n_test, n_skills - 5)
        tested[i, rng.choice(n_skills, size=n_test, replace=False)] = True
    return tested


def generate_all_responses(
    rng: np.random.Generator,
    latent_profiles: np.ndarray,
    tested_skills: np.ndarray,
    student_ids: list[str],
) -> pd.DataFrame:
    skill_by_id = {s.skill_id: s for s in SKILLS}
    rows: list[dict[str, Any]] = []

    for i in range(N_STUDENTS):
        sid = student_ids[i]
        latent = latent_profiles[i]
        for j, skill in enumerate(SKILLS):
            if not tested_skills[i, j]:
                continue
            for item in SKILL_ITEMS[skill.skill_id]:
                p = _p_correct(item, skill, latent)
                is_correct = bool(rng.random() < p)
                chosen = item.correct_choice if is_correct else _pick_wrong_choice(item.correct_choice, rng)
                rows.append(
                    {
                        "student_id": sid,
                        "item_id": item.item_id,
                        "skill_id": skill.skill_id,
                        "skill_name": skill.skill_name,
                        "stem": item.stem,
                        "choice_a": item.choice_a,
                        "choice_b": item.choice_b,
                        "choice_c": item.choice_c,
                        "choice_d": item.choice_d,
                        "chosen": chosen,
                        "correct_choice": item.correct_choice,
                        "is_correct": is_correct,
                        "difficulty": item.difficulty,
                    }
                )
    return pd.DataFrame(rows)


def build_response_matrix(responses_df: pd.DataFrame, item_ids: list[str], student_ids: list[str]) -> np.ndarray:
    """R matrix: students x items, values 0/1 (NaN if not attempted)."""
    n = len(student_ids)
    q = len(item_ids)
    item_idx = {iid: k for k, iid in enumerate(item_ids)}
    student_idx = {sid: i for i, sid in enumerate(student_ids)}
    R = np.full((n, q), np.nan)
    for row in responses_df.itertuples():
        i = student_idx[row.student_id]
        j = item_idx[row.item_id]
        R[i, j] = 1.0 if row.is_correct else 0.0
    return R


def build_skill_item_matrix(item_ids: list[str]) -> np.ndarray:
    """Q matrix: items x skills — each item maps to exactly one skill."""
    q = len(item_ids)
    m = N_SKILLS
    item_to_skill = {i.item_id: i.skill_id for i in ITEMS}
    skill_idx = {s.skill_id: j for j, s in enumerate(SKILLS)}
    Q = np.zeros((q, m), dtype=float)
    for k, iid in enumerate(item_ids):
        Q[k, skill_idx[item_to_skill[iid]]] = 1.0
    return Q


def responses_to_skill_matrix(
    responses_df: pd.DataFrame,
    student_ids: list[str],
    skills: list[Skill],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate item responses to skill mastery matrix S and mask M.
    S[i,j] = 100 * (# correct / # attempted) for tested skills; 0 with mask False otherwise.
    """
    n = len(student_ids)
    m = len(skills)
    scores = np.zeros((n, m), dtype=float)
    mask = np.zeros((n, m), dtype=bool)

    if responses_df.empty:
        return scores, mask

    grouped = responses_df.groupby(["student_id", "skill_id"])["is_correct"]
    for (sid, skill_id), series in grouped:
        i = student_ids.index(sid)
        j = next(idx for idx, s in enumerate(skills) if s.skill_id == skill_id)
        attempted = len(series)
        correct = int(series.sum())
        scores[i, j] = 100.0 * correct / attempted if attempted else 0.0
        mask[i, j] = True

    return scores, mask


def build_skill_affinity_matrix(skills: list[Skill]) -> np.ndarray:
    """Skill-neighborhood weights W (row-normalized) for label propagation."""
    m = len(skills)
    W = np.zeros((m, m), dtype=float)

    for j, skill in enumerate(skills):
        for jp, other in enumerate(skills):
            if j == jp:
                continue
            w = 0.0
            if other.skill_id in skill.prerequisites or skill.skill_id in other.prerequisites:
                w = max(w, 1.0)
            if other.category == skill.category:
                w = max(w, 0.5)
            if abs(LEVEL_INDEX[skill.level] - LEVEL_INDEX[other.level]) == 1:
                w = max(w, 0.3)
            W[j, jp] = w

    for j in range(m):
        row_sum = W[j].sum()
        if row_sum > 0:
            W[j] /= row_sum
    return W


def gaussian_kernel(u: np.ndarray, v: np.ndarray, sigma: float = KERNEL_SIGMA) -> float:
    """K(u,v) with per-dimension mean squared error so σ is in mastery-% units."""
    diff = u - v
    n = len(diff)
    if n == 0:
        return 0.0
    mean_sq_dist = float(np.dot(diff, diff)) / n
    return float(np.exp(-mean_sq_dist / (2 * sigma**2)))


def kernel_similarity_observed(
    target: np.ndarray,
    peer: np.ndarray,
    shared_mask: np.ndarray,
    sigma: float = KERNEL_SIGMA,
) -> tuple[float, float, float]:
    """
    Return (kernel value, total squared distance, mean squared distance per shared skill).

    Distances are averaged over |Ω| so σ=25 means “typical per-skill gap in mastery %”
    rather than penalizing students for having many overlapping tested skills.
    """
    u = target[shared_mask]
    v = peer[shared_mask]
    n = len(u)
    if n < 5:
        return 0.0, 0.0, 0.0
    diff = u - v
    sq_dist = float(np.dot(diff, diff))
    mean_sq_dist = sq_dist / n
    k = float(np.exp(-mean_sq_dist / (2 * sigma**2)))
    return k, sq_dist, mean_sq_dist


def related_skill_prediction(
    target_scores: np.ndarray,
    target_mask: np.ndarray,
    skill_j: int,
    W: np.ndarray,
) -> float:
    """Propagate from related known skills using affinity row W[j]."""
    weights = W[skill_j] * target_mask.astype(float)
    total = weights.sum()
    if total <= 0:
        return float("nan")
    return float(np.dot(weights, target_scores) / total)


def predict_missing_scores_knn_propagate(
    scores: np.ndarray,
    mask: np.ndarray,
    target_idx: int,
    W: np.ndarray,
    top_k: int = K_NEIGHBORS,
    sigma: float = KERNEL_SIGMA,
    alpha: float = PROPAGATION_ALPHA,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, float]]]:
    """
    Kernel k-NN neighbor prediction blended with skill-neighborhood propagation.
    Returns final predictions, per-student kernel sims, and per-skill breakdown.
    """
    n, m = scores.shape
    target = scores[target_idx]
    target_obs = mask[target_idx]

    kernel_sims = np.zeros(n, dtype=float)
    for i in range(n):
        if i == target_idx:
            continue
        shared = target_obs & mask[i]
        if shared.sum() < 5:
            kernel_sims[i] = 0.0
            continue
        k, _, _ = kernel_similarity_observed(target, scores[i], shared, sigma)
        kernel_sims[i] = k

    predictions = np.full(m, np.nan)
    breakdown: dict[int, dict[str, float]] = {}

    for j in range(m):
        if target_obs[j]:
            predictions[j] = target[j]
            continue

        neighbor_pred = float("nan")
        peer_mask = mask[:, j] & (np.arange(n) != target_idx)
        peer_idxs = np.where(peer_mask)[0]
        if len(peer_idxs) > 0:
            peer_kernels = kernel_sims[peer_idxs]
            positive = peer_kernels > 0
            if positive.any():
                peer_idxs = peer_idxs[positive]
                peer_kernels = peer_kernels[positive]
                order = np.argsort(peer_kernels)[::-1][:top_k]
                chosen = peer_idxs[order]
                weights = peer_kernels[order]
                neighbor_pred = float(np.dot(weights, scores[chosen, j]) / weights.sum())

        related_pred = related_skill_prediction(target, target_obs, j, W)

        if not np.isnan(neighbor_pred) and not np.isnan(related_pred):
            final = alpha * neighbor_pred + (1 - alpha) * related_pred
        elif not np.isnan(neighbor_pred):
            final = neighbor_pred
        elif not np.isnan(related_pred):
            final = related_pred
        else:
            continue

        predictions[j] = final
        breakdown[j] = {
            "neighbor_pred": round(neighbor_pred, 1) if not np.isnan(neighbor_pred) else None,
            "related_pred": round(related_pred, 1) if not np.isnan(related_pred) else None,
            "final_pred": round(final, 1),
        }

    return predictions, kernel_sims, breakdown


def _observed_skill_stats(responses_df: pd.DataFrame, student_id: str) -> dict[str, dict[str, Any]]:
    sub = responses_df[responses_df["student_id"] == student_id]
    stats: dict[str, dict[str, Any]] = {}
    for skill_id, grp in sub.groupby("skill_id"):
        stats[skill_id] = {
            "attempted": len(grp),
            "correct": int(grp["is_correct"].sum()),
            "mastery": round(100.0 * grp["is_correct"].sum() / len(grp), 1),
            "skill_name": grp.iloc[0]["skill_name"],
        }
    return stats


def build_predicted_missing_skills_df(
    predictions: np.ndarray,
    mask: np.ndarray,
    target_idx: int,
    skills: list[Skill],
    breakdown: dict[int, dict[str, float]],
    student_id: str,
) -> pd.DataFrame:
    rows = []
    for j, skill in enumerate(skills):
        if mask[target_idx, j]:
            continue
        pred = predictions[j]
        if np.isnan(pred):
            continue
        bd = breakdown.get(j, {})
        neighbor = bd.get("neighbor_pred")
        related = bd.get("related_pred")
        priority = (100 - pred) * skill.foundational_weight
        rows.append(
            {
                "student_id": student_id,
                "skill_id": skill.skill_id,
                "skill_name": skill.skill_name,
                "category": skill.category,
                "level": skill.level,
                "neighbor_pred": neighbor,
                "related_pred": related,
                "final_pred": round(pred, 1),
                "foundational_weight": skill.foundational_weight,
                "priority": round(priority, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("priority", ascending=False).reset_index(drop=True)


def rank_recommendations(
    predictions: np.ndarray,
    mask: np.ndarray,
    target_idx: int,
    skills: list[Skill],
    observed_stats: dict[str, dict[str, Any]],
    breakdown: dict[int, dict[str, float]] | None = None,
) -> pd.DataFrame:
    rows = []
    observed_cats = {
        skills[j].category
        for j in range(len(skills))
        if mask[target_idx, j]
        for _ in [0]
    }
    weak_observed = [
        (sid, st)
        for sid, st in observed_stats.items()
        if st["mastery"] < 50
    ]
    weak_hint = ""
    if weak_observed:
        weak_hint = f" Observed weakness in {weak_observed[0][1]['skill_name']} ({weak_observed[0][1]['correct']}/{weak_observed[0][1]['attempted']} correct)."

    for j, skill in enumerate(skills):
        is_tested = bool(mask[target_idx, j])
        pred = predictions[j]
        if np.isnan(pred):
            continue
        priority = (100 - pred) * skill.foundational_weight
        bd = (breakdown or {}).get(j, {})
        neighbor = bd.get("neighbor_pred")
        related = bd.get("related_pred")
        if is_tested:
            st = observed_stats.get(skill.skill_id)
            work_note = ""
            if st:
                work_note = f" ({st['correct']}/{st['attempted']} items correct)."
            reason = (
                f"Observed {pred:.0f}% on tested {skill.level.lower()} {skill.category.lower()} skill"
                f"{work_note} Priority reflects measured weakness × foundational weight."
            )
        else:
            blend_note = ""
            if neighbor is not None and related is not None:
                blend_note = (
                    f" Blended k-NN ({neighbor:.0f}%) and related-skill propagation ({related:.0f}%)."
                )
            elif neighbor is not None:
                blend_note = f" From k-NN peers ({neighbor:.0f}%)."
            elif related is not None:
                blend_note = f" From related-skill propagation ({related:.0f}%)."
            reason = (
                f"Predicted {pred:.0f}% on untested {skill.level.lower()} {skill.category.lower()} skill;"
                f"{blend_note}{weak_hint} Similar students with overlapping "
                f"{', '.join(sorted(observed_cats)) or 'tested'} skills also struggled here."
            ).replace("  ", " ")
        rows.append(
            {
                "skill_id": skill.skill_id,
                "skill_name": skill.skill_name,
                "category": skill.category,
                "level": skill.level,
                "is_tested": is_tested,
                "predicted_mastery": round(pred, 1),
                "neighbor_pred": neighbor if not is_tested else None,
                "related_pred": related if not is_tested else None,
                "foundational_weight": skill.foundational_weight,
                "priority_score": round(priority, 1),
                "reason": reason,
            }
        )
    df = pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def build_interpretation(recs: pd.DataFrame, observed_stats: dict[str, dict[str, Any]], top_n: int = 3) -> str:
    if recs.empty:
        return "Insufficient skills to generate recommendations."
    top = recs.head(top_n)
    names = ", ".join(top["skill_name"].tolist())
    categories = ", ".join(sorted(set(top["category"].tolist())))
    tested_top = top[top["is_tested"]] if "is_tested" in top.columns else pd.DataFrame()
    untested_top = top[~top["is_tested"]] if "is_tested" in top.columns else top
    weak = sorted(observed_stats.values(), key=lambda x: x["mastery"])[:2]
    work_note = ""
    if weak:
        work_note = (
            f" Based on item responses, the student scored {weak[0]['mastery']:.0f}% on "
            f"{weak[0]['skill_name']} ({weak[0]['correct']}/{weak[0]['attempted']} correct)."
        )
    if not untested_top.empty and not tested_top.empty:
        lead = (
            f"Top priorities mix tested gaps and predicted untested skills: {names}."
        )
    elif not tested_top.empty:
        lead = f"Highest-priority tested skills to reinforce: {names}."
    else:
        lead = f"The model predicts this student is likely weak in {names}."
    return (
        f"{lead}{work_note} These foundational {categories.lower()} skills rank highest "
        f"by (100 − mastery) × foundational weight across all 72 skills."
    )


def choose_default_student(scores: np.ndarray, mask: np.ndarray, skills: list[Skill]) -> int:
    basic_idxs = [i for i, s in enumerate(skills) if s.level == "Basic"]
    best_idx = 0
    best_score = -1.0
    for i in range(scores.shape[0]):
        observed_basic = mask[i, basic_idxs]
        if observed_basic.sum() < 8:
            continue
        mean_basic = scores[i, basic_idxs][observed_basic].mean()
        if mean_basic < best_score or best_score < 0:
            best_idx = i
            best_score = mean_basic
    return best_idx


def scores_to_long_df(
    scores: np.ndarray,
    mask: np.ndarray,
    student_ids: list[str],
    skill_ids: list[str],
) -> pd.DataFrame:
    rows = []
    for i, sid in enumerate(student_ids):
        for j, kid in enumerate(skill_ids):
            rows.append(
                {
                    "student_id": sid,
                    "skill_id": kid,
                    "score": round(float(scores[i, j]), 1) if mask[i, j] else None,
                    "is_observed": bool(mask[i, j]),
                }
            )
    return pd.DataFrame(rows)


def build_observed_work(responses_df: pd.DataFrame, student_id: str) -> list[dict[str, Any]]:
    sub = responses_df[responses_df["student_id"] == student_id]
    work: list[dict[str, Any]] = []
    for skill_id, grp in sub.groupby("skill_id"):
        skill = next(s for s in SKILLS if s.skill_id == skill_id)
        attempted = len(grp)
        correct = int(grp["is_correct"].sum())
        work.append(
            {
                "skill_id": skill_id,
                "skill_name": grp.iloc[0]["skill_name"],
                "category": skill.category,
                "attempted": attempted,
                "correct": correct,
                "mastery": round(100.0 * correct / attempted, 1),
                "items": [
                    {
                        "item_id": r.item_id,
                        "stem": r.stem,
                        "chosen": r.chosen,
                        "correct_choice": r.correct_choice,
                        "is_correct": bool(r.is_correct),
                        "choice_a": r.choice_a,
                        "choice_b": r.choice_b,
                        "choice_c": r.choice_c,
                        "choice_d": r.choice_d,
                    }
                    for r in grp.itertuples()
                ],
            }
        )
    work.sort(key=lambda x: x["mastery"])
    return work


def build_methodology_demo(
    result: PipelineResult,
    student_idx: int,
    analysis: dict[str, Any],
    sigma: float = KERNEL_SIGMA,
    alpha: float = PROPAGATION_ALPHA,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Concrete numbers for the on-site linear algebra walkthrough."""
    observed_work = analysis["observed_work"]

    skill_ex: dict[str, Any] | None = None
    if observed_work:
        ex = observed_work[0]
        skill_ex = {
            "skillName": ex["skill_name"],
            "correct": ex["correct"],
            "attempted": ex["attempted"],
            "mastery": ex["mastery"],
            "formula": f"100 × {ex['correct']}/{ex['attempted']} = {ex['mastery']}%",
        }

    sim_ex: dict[str, Any] | None = None
    if analysis["peers"]:
        peer_id = analysis["peers"][0]["student_id"]
        peer_idx = result.student_ids.index(peer_id)
        shared = result.mask[student_idx] & result.mask[peer_idx]
        u = result.scores[student_idx, shared]
        v = result.scores[peer_idx, shared]
        diff = u - v
        sq_dist = float(np.dot(diff, diff))
        n_shared = int(shared.sum())
        mean_sq_dist = sq_dist / n_shared if n_shared else 0.0
        k_val, _, _ = kernel_similarity_observed(
            result.scores[student_idx],
            result.scores[peer_idx],
            shared,
            sigma,
        )
        sim_ex = {
            "peerStudentId": peer_id,
            "sharedSkills": n_shared,
            "squaredDistance": round(sq_dist, 2),
            "meanSquaredDistance": round(mean_sq_dist, 2),
            "kernelValue": round(k_val, 4),
            "sigma": sigma,
            "alpha": alpha,
        }

    pred_ex: dict[str, Any] | None = None
    if analysis["recommendations"]:
        top = analysis["recommendations"][0]
        skill_j = next(i for i, s in enumerate(result.skills) if s.skill_id == top["skill_id"])
        peers_with_skill = int(
            (result.mask[:, skill_j] & (np.arange(len(result.student_ids)) != student_idx)).sum()
        )
        pred_ex = {
            "skillName": top["skill_name"],
            "predictedMastery": top["predicted_mastery"],
            "neighborPrediction": top.get("neighbor_pred"),
            "relatedPrediction": top.get("related_pred"),
            "foundationalWeight": top["foundational_weight"],
            "priorityScore": top["priority_score"],
            "peersWithSkill": peers_with_skill,
            "priorityFormula": (
                f"(100 − {top['predicted_mastery']:.0f}) × {top['foundational_weight']:.2f} "
                f"= {top['priority_score']:.1f}"
            ),
        }

    n_students = len(result.student_ids)
    return {
        "randomSeed": random_seed,
        "matrixShapes": {
            "R": f"{n_students}×{len(result.item_ids)}",
            "Q": f"{len(result.item_ids)}×{N_SKILLS}",
            "S": f"{n_students}×{N_SKILLS}",
        },
        "dataFiles": [
            "data/item_bank.csv",
            "data/student_responses.csv",
            "data/synthetic_student_scores.csv",
        ],
        "notRandomNote": (
            "Every mastery percentage equals (# correct MCQ items) ÷ (# attempted) for that skill. "
            "Hidden latent profiles only set P(correct) when items are generated offline in Python — "
            "they are never shown as scores on this page."
        ),
        "skillAggregationExample": skill_ex,
        "studentVector": {
            "dimension": N_SKILLS,
            "observedCount": analysis["summary"]["observed_skills"],
            "untestedCount": analysis["summary"]["untested_skills"],
            "totalItemsAttempted": analysis["summary"]["total_items_attempted"],
        },
        "similarityExample": sim_ex,
        "predictionExample": pred_ex,
    }


def build_custom_observed_work(
    observed_skills: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Synthetic observed work from manual mastery inputs (no MCQ items)."""
    skill_by_id = {s.skill_id: s for s in SKILLS}
    work: list[dict[str, Any]] = []
    for entry in observed_skills:
        skill_id = entry["skill_id"]
        mastery = float(entry["mastery"])
        skill = skill_by_id[skill_id]
        attempted = 3
        correct = int(round(mastery / 100.0 * attempted))
        work.append(
            {
                "skill_id": skill_id,
                "skill_name": skill.skill_name,
                "category": skill.category,
                "attempted": attempted,
                "correct": correct,
                "mastery": round(mastery, 1),
                "is_custom_input": True,
                "items": [],
            }
        )
    work.sort(key=lambda x: x["mastery"])
    return work


def _observed_stats_from_work(work: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        w["skill_id"]: {
            "attempted": w["attempted"],
            "correct": w["correct"],
            "mastery": w["mastery"],
            "skill_name": w["skill_name"],
        }
        for w in work
    }


def inject_custom_student(
    result: PipelineResult,
    name: str,
    observed_skills: list[dict[str, Any]],
) -> tuple[PipelineResult, int, list[dict[str, Any]]]:
    """Append a custom student row; returns updated result, index, and observed work."""
    if name in result.student_ids:
        raise ValueError(f"Student name already exists: {name}")
    skill_idx = {s.skill_id: j for j, s in enumerate(result.skills)}
    new_row_scores = np.zeros(N_SKILLS, dtype=float)
    new_row_mask = np.zeros(N_SKILLS, dtype=bool)
    normalized: list[dict[str, Any]] = []
    for entry in observed_skills:
        sid = entry["skill_id"]
        if sid not in skill_idx:
            raise ValueError(f"Unknown skill: {sid}")
        j = skill_idx[sid]
        mastery = float(np.clip(entry["mastery"], 0, 100))
        new_row_scores[j] = mastery
        new_row_mask[j] = True
        normalized.append({"skill_id": sid, "mastery": mastery})

    new_scores = np.vstack([result.scores, new_row_scores])
    new_mask = np.vstack([result.mask, new_row_mask])
    new_ids = result.student_ids + [name]
    new_idx = len(new_ids) - 1
    observed_work = build_custom_observed_work(normalized)
    updated = replace(
        result,
        scores=new_scores,
        mask=new_mask,
        student_ids=new_ids,
    )
    return updated, new_idx, observed_work


def _trace_knn_neighbors(
    result: PipelineResult,
    student_idx: int,
    skill_j: int,
    kernel_sims: np.ndarray,
    top_k: int,
) -> tuple[list[dict[str, Any]], float | None]:
    """Peers tested on skill_j with kernel weights and scores (for trace table)."""
    n = len(result.student_ids)
    peer_mask = result.mask[:, skill_j] & (np.arange(n) != student_idx)
    peer_idxs = np.where(peer_mask)[0]
    if len(peer_idxs) == 0:
        return [], None
    peer_kernels = kernel_sims[peer_idxs]
    positive = peer_kernels > 0
    if not positive.any():
        return [], None
    peer_idxs = peer_idxs[positive]
    peer_kernels = peer_kernels[positive]
    order = np.argsort(peer_kernels)[::-1][:top_k]
    chosen = peer_idxs[order]
    weights = peer_kernels[order]
    weight_sum = float(weights.sum())
    rows = []
    weighted_sum = 0.0
    for idx, w in zip(chosen, weights):
        score_j = float(result.scores[idx, skill_j])
        contrib = float(w * score_j)
        weighted_sum += contrib
        rows.append(
            {
                "Peer": result.student_ids[idx],
                "K_i": round(float(w), 4),
                "Score on skill": round(score_j, 1),
                "K_i × score": round(contrib, 2),
            }
        )
    neighbor_pred = weighted_sum / weight_sum if weight_sum > 0 else None
    return rows, neighbor_pred


def _trace_propagation_sources(
    result: PipelineResult,
    student_idx: int,
    skill_j: int,
) -> list[dict[str, Any]]:
    """Related known skills that feed label propagation for skill_j."""
    W = result.skill_affinity
    target = result.scores[student_idx]
    target_obs = result.mask[student_idx]
    rows = []
    for jp, skill in enumerate(result.skills):
        w = float(W[skill_j, jp])
        if w <= 0 or not target_obs[jp]:
            continue
        score = float(target[jp])
        rows.append(
            {
                "Related skill": skill.skill_name,
                "W weight": round(w, 3),
                "Known score": round(score, 1),
                "W × score": round(w * score, 2),
            }
        )
    rows.sort(key=lambda r: -float(r["W weight"]))
    return rows[:10]


def build_computation_trace(
    result: PipelineResult,
    student_idx: int,
    analysis: dict[str, Any],
    sigma: float = KERNEL_SIGMA,
    alpha: float = PROPAGATION_ALPHA,
    top_k: int = K_NEIGHBORS,
) -> list[dict[str, Any]]:
    """Step-by-step trace for the live showcase panel with provenance for each number."""
    steps: list[dict[str, Any]] = []
    sid = result.student_ids[student_idx]
    observed = int(result.mask[student_idx].sum())
    observed_work = analysis.get("observed_work") or []
    is_custom = analysis.get("summary", {}).get("is_custom_input", False)
    step_num = 0

    step_num += 1
    skill_examples = []
    for w in observed_work[:5]:
        skill_examples.append(
            {
                "Skill": w["skill_name"],
                "Correct": w["correct"],
                "Attempted": w["attempted"],
                "Mastery %": w["mastery"],
            }
        )
    source_note = (
        "Scores entered manually in the Live showcase (no MCQ items)."
        if is_custom
        else "Each % comes from (# correct MCQs) ÷ (# attempted) for that skill in student_responses.csv."
    )
    steps.append(
        {
            "step": step_num,
            "function": "lib.responses_to_skill_matrix",
            "codeRef": "S[i,j] = 100 × (# correct) / (# attempted)",
            "title": f"Build {sid}'s partial skill vector",
            "explanation": (
                f"{sid} has {observed} tested skills and {N_SKILLS - observed} untested. "
                f"Each tested coordinate in matrix S is a real percentage, not random. {source_note}"
            ),
            "details": [
                {"label": "Student", "value": sid},
                {"label": "Known skills (|Ω|)", "value": str(observed)},
                {"label": "Missing skills", "value": str(N_SKILLS - observed)},
                {"label": "Total items attempted", "value": str(analysis["summary"].get("total_items_attempted", "—"))},
            ],
            "table": {
                "headers": ["Skill", "Correct", "Attempted", "Mastery %"],
                "rows": skill_examples,
            }
            if skill_examples
            else None,
            "inputs": {"student": sid, "knownSkills": observed, "totalSkills": N_SKILLS},
            "output": f"{observed} known coordinates, {N_SKILLS - observed} missing",
            "latex": rf"\mathbf{{s}}_{{{sid}}} \in \mathbb{{R}}^{{72}},\; |\Omega| = {observed}",
        }
    )

    kernel_sims = analysis.get("similarities")
    if kernel_sims is None:
        kernel_sims = np.zeros(len(result.student_ids))

    peers = analysis.get("peers") or []
    if len(peers) > 0:
        peer_id = peers[0]["student_id"]
        peer_idx = result.student_ids.index(peer_id)
        shared = result.mask[student_idx] & result.mask[peer_idx]
        u = result.scores[student_idx, shared]
        v = result.scores[peer_idx, shared]
        k_val, sq_dist, mean_sq_dist = kernel_similarity_observed(
            result.scores[student_idx],
            result.scores[peer_idx],
            shared,
            sigma,
        )
        shared_headers = ["Skill", sid, peer_id, "(u−v)²"]
        shared_samples = []
        shared_idxs = np.where(shared)[0]
        for j in shared_idxs[:6]:
            skill = result.skills[j]
            u_j = float(result.scores[student_idx, j])
            v_j = float(result.scores[peer_idx, j])
            shared_samples.append(
                {
                    "Skill": skill.skill_name[:24],
                    sid: round(u_j, 1),
                    peer_id: round(v_j, 1),
                    "(u−v)²": round((u_j - v_j) ** 2, 1),
                }
            )
        step_num += 1
        steps.append(
            {
                "step": step_num,
                "function": "lib.kernel_similarity_observed",
                "codeRef": "K = exp(−mean_sq / (2σ²)), mean_sq = ||u−v||² / |Ω|",
                "title": f"Compare {sid} to nearest peer {peer_id}",
                "explanation": (
                    f"Only the {int(shared.sum())} skills both students were tested on are used (set Ω). "
                    f"Total squared gap ||u−v||² = {sq_dist:.2f}; we divide by |Ω| to get mean_sq = "
                    f"{mean_sq_dist:.2f} (average per-skill gap in mastery %²). "
                    f"Then K = exp(−mean_sq / (2σ²)) with σ = {sigma} (per-skill bandwidth in %). "
                    f"Higher K means their observed profiles are more alike."
                ),
                "details": [
                    {"label": "Peer", "value": peer_id},
                    {"label": "Shared tested skills |Ω|", "value": str(int(shared.sum()))},
                    {"label": "Σ(u_k − v_k)²", "value": f"{sq_dist:.2f}"},
                    {"label": "mean_sq = Σ / |Ω|", "value": f"{mean_sq_dist:.2f}"},
                    {"label": "σ (per-skill bandwidth %)", "value": str(sigma)},
                    {"label": "Kernel K(u,v)", "value": f"{k_val:.4f}"},
                ],
                "table": {
                    "headers": shared_headers,
                    "rows": shared_samples,
                }
                if shared_samples
                else None,
                "inputs": {
                    "peer": peer_id,
                    "sharedSkills": int(shared.sum()),
                    "sigma": sigma,
                    "squaredDistance": round(sq_dist, 2),
                },
                "output": round(k_val, 4),
                "latex": (
                    rf"\text{{mean\_sq}} = \frac{{{sq_dist:.2f}}}{{{int(shared.sum())}}} = {mean_sq_dist:.2f},\;"
                    rf"K = \exp\!\left(-\frac{{{mean_sq_dist:.2f}}}{{2 \cdot {sigma}^2}}\right) = {k_val:.4f}"
                ),
            }
        )

    recs = analysis.get("recommendations") or []
    breakdown = analysis.get("breakdown") or {}
    if recs:
        top = recs[0]
        skill_j = next(i for i, s in enumerate(result.skills) if s.skill_id == top["skill_id"])
        is_tested = top.get("is_tested", False)
        bd = breakdown.get(skill_j, {})
        neighbor = bd.get("neighbor_pred")
        related = bd.get("related_pred")
        final = top["predicted_mastery"]

        if is_tested:
            step_num += 1
            st = next((w for w in observed_work if w["skill_id"] == top["skill_id"]), None)
            steps.append(
                {
                    "step": step_num,
                    "function": "lib.rank_recommendations",
                    "codeRef": "priority = (100 − observed mastery) × foundational_weight",
                    "title": f"Top priority: tested skill {top['skill_name']}",
                    "explanation": (
                        f"This skill was actually tested. The score {final:.0f}% is observed from item "
                        f"responses, not predicted. Priority still uses (100 − score) × foundational weight "
                        f"so low observed mastery on foundational skills surfaces at the top."
                    ),
                    "details": [
                        {"label": "Status", "value": "Tested (observed)"},
                        {"label": "Observed mastery", "value": f"{final:.0f}%"},
                        {
                            "label": "Item work",
                            "value": (
                                f"{st['correct']}/{st['attempted']} correct"
                                if st
                                else "from skill vector"
                            ),
                        },
                        {"label": "Foundational weight", "value": f"{top['foundational_weight']:.2f}"},
                        {"label": "Priority score", "value": f"{top['priority_score']:.1f}"},
                    ],
                    "inputs": {"skill": top["skill_name"], "observedMastery": final},
                    "output": top["priority_score"],
                    "latex": (
                        rf"\text{{priority}} = (100 - {final:.0f}) \times "
                        rf"{top['foundational_weight']:.2f} = {top['priority_score']:.1f}"
                    ),
                }
            )
        else:
            knn_rows, knn_calc = _trace_knn_neighbors(
                result, student_idx, skill_j, kernel_sims, top_k
            )
            if knn_rows:
                step_num += 1
                weight_sum = sum(float(r["K_i"]) for r in knn_rows)
                numer = sum(float(r["K_i × score"]) for r in knn_rows)
                steps.append(
                    {
                        "step": step_num,
                        "function": "lib.predict_missing_scores_knn_propagate",
                        "codeRef": "neighbor_pred = Σ(K_i × peer_score_i) / Σ K_i",
                        "title": f"k-NN neighbors for untested {top['skill_name']}",
                        "explanation": (
                            f"{len(knn_rows)} peers were tested on this skill. Each peer's score on "
                            f"'{top['skill_name']}' is weighted by kernel similarity K_i to {sid} "
                            f"(computed on shared skills from step 2). "
                            f"Weighted sum {numer:.1f} ÷ ΣK_i {weight_sum:.4f} = {neighbor or knn_calc:.1f}%."
                        ),
                        "details": [
                            {"label": "Peers with this skill tested", "value": str(len(knn_rows))},
                            {"label": "k (max neighbors used)", "value": str(top_k)},
                            {"label": "Σ K_i", "value": f"{weight_sum:.4f}"},
                            {"label": "Σ K_i × score_i", "value": f"{numer:.1f}"},
                            {"label": "Neighbor prediction", "value": f"{(neighbor or knn_calc):.1f}%"},
                        ],
                        "table": {
                            "headers": ["Peer", "K_i", "Score on skill", "K_i × score"],
                            "rows": knn_rows,
                        },
                        "inputs": {"skill": top["skill_name"], "k": top_k},
                        "output": neighbor or knn_calc,
                        "latex": (
                            rf"\hat{{y}}^{{\text{{neighbor}}}} = "
                            rf"\frac{{{numer:.1f}}}{{{weight_sum:.4f}}} = {(neighbor or knn_calc):.1f}"
                        ),
                    }
                )

            prop_rows = _trace_propagation_sources(result, student_idx, skill_j)
            if prop_rows and related is not None:
                step_num += 1
                w_sum = sum(float(r["W weight"]) for r in prop_rows)
                numer = sum(float(r["W × score"]) for r in prop_rows)
                steps.append(
                    {
                        "step": step_num,
                        "function": "lib.related_skill_prediction",
                        "codeRef": "related_pred = Σ W[j,j′] × known_score_j′ / Σ W[j,j′]",
                        "title": f"Label propagation for {top['skill_name']}",
                        "explanation": (
                            f"Skill affinity row W[{top['skill_id']}] links prerequisites, same-category, "
                            f"and adjacent-level neighbors. {sid}'s known scores on related skills are "
                            f"averaged with those weights: {numer:.1f} ÷ {w_sum:.3f} = {related:.1f}%."
                        ),
                        "details": [
                            {"label": "Related known skills used", "value": str(len(prop_rows))},
                            {"label": "Σ W[j,j′]", "value": f"{w_sum:.3f}"},
                            {"label": "Σ W × known score", "value": f"{numer:.1f}"},
                            {"label": "Related prediction", "value": f"{related:.1f}%"},
                        ],
                        "table": {
                            "headers": ["Related skill", "W weight", "Known score", "W × score"],
                            "rows": prop_rows,
                        },
                        "inputs": {"skill": top["skill_name"]},
                        "output": related,
                        "latex": (
                            rf"\hat{{y}}^{{\text{{related}}}} = "
                            rf"\frac{{{numer:.1f}}}{{{w_sum:.3f}}} = {related:.1f}"
                        ),
                    }
                )

            step_num += 1
            if neighbor is not None and related is not None:
                blend_latex = (
                    rf"\hat{{y}}^{{\text{{final}}}} = {alpha} \cdot {neighbor} + "
                    rf"(1-{alpha}) \cdot {related} = {final}"
                )
                blend_expl = (
                    f"Blend neighbor prediction ({neighbor:.1f}%) and propagation ({related:.1f}%) "
                    f"with α = {alpha}: {alpha}×{neighbor:.1f} + {1-alpha}×{related:.1f} = {final:.1f}%."
                )
            elif neighbor is not None:
                blend_latex = rf"\hat{{y}}^{{\text{{final}}}} = {neighbor} \text{{ (k-NN only)}}"
                blend_expl = f"Only k-NN available → final = {neighbor:.1f}%."
            elif related is not None:
                blend_latex = rf"\hat{{y}}^{{\text{{final}}}} = {related} \text{{ (propagation only)}}"
                blend_expl = f"Only propagation available → final = {related:.1f}%."
            else:
                blend_latex = rf"\hat{{y}}^{{\text{{final}}}} = {final}"
                blend_expl = f"Final predicted mastery = {final:.1f}%."
            steps.append(
                {
                    "step": step_num,
                    "function": "lib.predict_missing_scores_knn_propagate",
                    "codeRef": "final = α·neighbor + (1−α)·related",
                    "title": f"Final prediction for {top['skill_name']}",
                    "explanation": blend_expl,
                    "details": [
                        {"label": "α (neighbor weight)", "value": str(alpha)},
                        {"label": "Neighbor pred", "value": f"{neighbor}%" if neighbor is not None else "—"},
                        {"label": "Related pred", "value": f"{related}%" if related is not None else "—"},
                        {"label": "Final mastery", "value": f"{final:.1f}%"},
                    ],
                    "inputs": {"alpha": alpha, "neighborPred": neighbor, "relatedPred": related},
                    "output": final,
                    "latex": blend_latex,
                }
            )

            step_num += 1
            steps.append(
                {
                    "step": step_num,
                    "function": "lib.rank_recommendations",
                    "codeRef": "priority = (100 − mastery) × foundational_weight",
                    "title": "Why this skill ranks #1",
                    "explanation": (
                        f"Lower mastery on high foundational-weight skills yields higher priority. "
                        f"(100 − {final:.0f}) × {top['foundational_weight']:.2f} = {top['priority_score']:.1f} "
                        f"beats other skills in the full 72-skill ranking."
                    ),
                    "details": [
                        {"label": "Mastery used", "value": f"{final:.1f}% (predicted)"},
                        {"label": "Gap (100 − mastery)", "value": f"{100 - final:.0f}"},
                        {"label": "Foundational weight", "value": f"{top['foundational_weight']:.2f}"},
                        {"label": "Priority score", "value": f"{top['priority_score']:.1f}"},
                    ],
                    "inputs": {
                        "predictedMastery": final,
                        "foundationalWeight": top["foundational_weight"],
                    },
                    "output": top["priority_score"],
                    "latex": (
                        rf"\text{{priority}} = (100 - {final:.0f}) \times "
                        rf"{top['foundational_weight']:.2f} = {top['priority_score']:.1f}"
                    ),
                }
            )

    return steps


def export_methodology_demo(
    result: PipelineResult,
    idx: int,
    analysis: dict[str, Any],
    sigma: float = KERNEL_SIGMA,
    alpha: float = PROPAGATION_ALPHA,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    demo = build_methodology_demo(result, idx, analysis, sigma=sigma, alpha=alpha, random_seed=random_seed)
    return {
        "randomSeed": demo["randomSeed"],
        "matrixShapes": demo["matrixShapes"],
        "dataFiles": demo["dataFiles"],
        "notRandomNote": demo["notRandomNote"],
        "skillAggregationExample": demo["skillAggregationExample"],
        "studentVector": {
            "dimension": demo["studentVector"]["dimension"],
            "observedCount": demo["studentVector"]["observedCount"],
            "untestedCount": demo["studentVector"]["untestedCount"],
            "totalItemsAttempted": demo["studentVector"]["totalItemsAttempted"],
        },
        "similarityExample": demo["similarityExample"],
        "predictionExample": demo["predictionExample"],
    }


def serialize_student_dashboard(
    result: PipelineResult,
    idx: int,
    analysis: dict[str, Any],
    sigma: float = KERNEL_SIGMA,
    alpha: float = PROPAGATION_ALPHA,
    random_seed: int = RANDOM_SEED,
    computation_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "summary": {
            "observedSkills": analysis["summary"]["observed_skills"],
            "untestedSkills": analysis["summary"]["untested_skills"],
            "totalItemsAttempted": analysis["summary"]["total_items_attempted"],
            "avgItemsPerSkill": analysis["summary"]["avg_items_per_skill"],
            "topPriorityScore": analysis["summary"]["top_priority_score"],
            "interpretation": analysis["summary"]["interpretation"],
            "isCustomInput": analysis["summary"].get("is_custom_input", False),
        },
        "observedWork": [
            {
                "skillId": w["skill_id"],
                "skillName": w["skill_name"],
                "category": w["category"],
                "attempted": w["attempted"],
                "correct": w["correct"],
                "mastery": w["mastery"],
                "isCustomInput": w.get("is_custom_input", False),
                "items": [
                    {
                        "itemId": it["item_id"],
                        "stem": it["stem"],
                        "chosen": it["chosen"],
                        "correctChoice": it["correct_choice"],
                        "isCorrect": it["is_correct"],
                        "choiceA": it["choice_a"],
                        "choiceB": it["choice_b"],
                        "choiceC": it["choice_c"],
                        "choiceD": it["choice_d"],
                    }
                    for it in w.get("items", [])
                ],
            }
            for w in analysis["observed_work"]
        ],
        "recommendations": [
            {
                "rank": r["rank"],
                "skillId": r["skill_id"],
                "skillName": r["skill_name"],
                "category": r["category"],
                "level": r["level"],
                "isTested": r.get("is_tested", False),
                "predictedMastery": r["predicted_mastery"],
                "neighborPred": r.get("neighbor_pred"),
                "relatedPred": r.get("related_pred"),
                "foundationalWeight": r["foundational_weight"],
                "priorityScore": r["priority_score"],
                "reason": r["reason"],
            }
            for r in analysis["recommendations"]
        ],
        "peers": [
            {"studentId": p["student_id"], "similarity": p["similarity"]}
            for p in analysis["peers"]
        ],
        "methodologyDemo": export_methodology_demo(
            result, idx, analysis, sigma=sigma, alpha=alpha, random_seed=random_seed
        ),
    }
    if computation_trace is not None:
        payload["computationTrace"] = computation_trace
    return payload


def build_dashboard_meta(
    result: PipelineResult,
    random_seed: int = RANDOM_SEED,
    sigma: float = KERNEL_SIGMA,
    alpha: float = PROPAGATION_ALPHA,
    top_k: int = K_NEIGHBORS,
) -> dict[str, Any]:
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "isSynthetic": True,
        "algorithm": "kernel-knn-label-propagation",
        "scoreSource": "item-responses",
        "randomSeed": random_seed,
        "kernelSigma": sigma,
        "kNeighbors": top_k,
        "propagationAlpha": alpha,
        "nStudents": len(result.student_ids),
        "nSkills": N_SKILLS,
        "nItems": len(result.item_ids),
        "dataFiles": [
            "data/item_bank.csv",
            "data/student_responses.csv",
            "data/synthetic_student_scores.csv",
            "output/predicted_missing_skills.csv",
        ],
        "pipelineSteps": [
            "Generate 216 synthetic MCQs (items.py)",
            "Simulate student responses → matrix R",
            "Aggregate R·Q → skill matrix S (% correct per skill)",
            "Gaussian kernel K(u,v) on shared known skills",
            "k-NN weighted neighbor prediction for missing skills",
            "Label propagation from skill affinity graph W",
            "Blend final = α·neighbor + (1−α)·related",
            "Priority = (100 − predicted) × foundational weight",
        ],
    }


def build_students_meta(result: PipelineResult) -> list[dict[str, Any]]:
    students_meta = []
    for idx in range(len(result.student_ids)):
        sid = result.student_ids[idx]
        observed = int(result.mask[idx].sum())
        hidden = N_SKILLS - observed
        sub = result.responses_df[result.responses_df["student_id"] == sid]
        item_count = len(sub) if len(sub) else observed * 3
        students_meta.append(
            {
                "id": sid,
                "label": sid,
                "observedCount": observed,
                "hiddenCount": hidden,
                "itemCount": item_count,
            }
        )
    return students_meta


def skills_catalog() -> list[dict[str, str]]:
    return [
        {
            "id": s.skill_id,
            "name": s.skill_name,
            "category": s.category,
            "level": s.level,
        }
        for s in SKILLS
    ]


def run_pipeline(seed: int = RANDOM_SEED) -> PipelineResult:
    rng = np.random.default_rng(seed)
    write_skill_metadata(Path("data/skill_metadata.csv"))
    write_item_bank(Path("data/item_bank.csv"))

    latent_profiles = np.vstack([_archetype_latent(rng) for _ in range(N_STUDENTS)])
    student_ids = assign_student_names(rng)
    tested_skills = _assign_tested_skills(rng, N_SKILLS)
    responses_df = generate_all_responses(rng, latent_profiles, tested_skills, student_ids)
    responses_df.to_csv("data/student_responses.csv", index=False)
    skill_ids = [s.skill_id for s in SKILLS]
    item_ids = [i.item_id for i in ITEMS]

    scores, mask = responses_to_skill_matrix(responses_df, student_ids, SKILLS)
    R = build_response_matrix(responses_df, item_ids, student_ids)
    Q = build_skill_item_matrix(item_ids)
    W = build_skill_affinity_matrix(SKILLS)

    default_idx = choose_default_student(scores, mask, SKILLS)
    long_df = scores_to_long_df(scores, mask, student_ids, skill_ids)
    long_df.to_csv("data/synthetic_student_scores.csv", index=False)

    return PipelineResult(
        scores=scores,
        mask=mask,
        student_ids=student_ids,
        skill_ids=skill_ids,
        skills=SKILLS,
        latent_profiles=latent_profiles,
        default_student_idx=default_idx,
        responses_df=responses_df,
        item_ids=item_ids,
        response_matrix=R,
        skill_item_matrix=Q,
        skill_affinity=W,
    )


def analyze_student(
    result: PipelineResult,
    student_idx: int,
    top_k: int = K_NEIGHBORS,
    sigma: float = KERNEL_SIGMA,
    alpha: float = PROPAGATION_ALPHA,
    observed_work_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sid = result.student_ids[student_idx]
    is_custom = observed_work_override is not None
    if is_custom:
        observed_work = observed_work_override
        observed_stats = _observed_stats_from_work(observed_work)
    else:
        observed_stats = _observed_skill_stats(result.responses_df, sid)
        observed_work = build_observed_work(result.responses_df, sid)

    predictions, kernel_sims, breakdown = predict_missing_scores_knn_propagate(
        result.scores,
        result.mask,
        student_idx,
        result.skill_affinity,
        top_k=top_k,
        sigma=sigma,
        alpha=alpha,
    )
    recs = rank_recommendations(
        predictions, result.mask, student_idx, result.skills, observed_stats, breakdown
    )
    peer_order = np.argsort(kernel_sims)[::-1]
    peers = []
    for idx in peer_order:
        if idx == student_idx or kernel_sims[idx] <= 0:
            continue
        peers.append(
            {
                "student_id": result.student_ids[idx],
                "similarity": round(float(kernel_sims[idx]), 4),
            }
        )
        if len(peers) >= 10:
            break

    observed = int(result.mask[student_idx].sum())
    hidden = N_SKILLS - observed
    if is_custom:
        total_items = sum(w["attempted"] for w in observed_work)
    else:
        total_items = len(result.responses_df[result.responses_df["student_id"] == sid])
    avg_items = round(total_items / observed, 1) if observed else 0.0
    top_priority = float(recs.iloc[0]["priority_score"]) if not recs.empty else 0.0

    return {
        "summary": {
            "observed_skills": observed,
            "untested_skills": hidden,
            "total_items_attempted": total_items,
            "avg_items_per_skill": avg_items,
            "top_priority_score": round(top_priority, 1),
            "interpretation": build_interpretation(recs, observed_stats),
            "is_custom_input": is_custom,
        },
        "recommendations": [
            {
                "rank": int(row.rank),
                "skill_id": row.skill_id,
                "skill_name": row.skill_name,
                "category": row.category,
                "level": row.level,
                "is_tested": bool(getattr(row, "is_tested", False)),
                "predicted_mastery": float(row.predicted_mastery),
                "neighbor_pred": getattr(row, "neighbor_pred", None),
                "related_pred": getattr(row, "related_pred", None),
                "foundational_weight": float(row.foundational_weight),
                "priority_score": float(row.priority_score),
                "reason": row.reason,
            }
            for row in recs.itertuples()
        ],
        "peers": peers,
        "observed_work": observed_work,
        "predictions": predictions,
        "similarities": kernel_sims,
        "breakdown": breakdown,
    }


def save_figures(result: PipelineResult, student_idx: int, analysis: dict[str, Any]) -> None:
    out = Path("output/figures")
    out.mkdir(parents=True, exist_ok=True)

    subset_students = list(range(0, min(20, N_STUDENTS)))
    sub_scores = result.scores[subset_students]
    sub_mask = result.mask[subset_students]
    display = sub_scores.copy()
    display[~sub_mask] = np.nan

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        display,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Mastery % (from item responses)"},
        xticklabels=False,
        yticklabels=[result.student_ids[i] for i in subset_students],
    )
    plt.title("Student-Skill Matrix (NaN = untested — no items attempted)")
    plt.xlabel("Skills")
    plt.ylabel("Students")
    plt.tight_layout()
    plt.savefig(out / "heatmap.png", dpi=150)
    plt.close()

    recs = analysis["recommendations"][:10]
    if recs:
        plt.figure(figsize=(10, 6))
        names = [r["skill_name"][:28] for r in recs]
        priorities = [r["priority_score"] for r in recs]
        plt.barh(names[::-1], priorities[::-1], color="#c2a76d")
        plt.xlabel("Priority Score")
        plt.title("Top Recommended Foundational Skills")
        plt.tight_layout()
        plt.savefig(out / "top_recommendations.png", dpi=150)
        plt.close()

    peers = analysis["peers"]
    if peers:
        plt.figure(figsize=(10, 5))
        labels = [p["student_id"] for p in peers]
        values = [p["similarity"] for p in peers]
        plt.bar(labels, values, color="#3e5442")
        plt.ylim(0, 1)
        plt.ylabel("Gaussian kernel similarity")
        plt.title("Nearest k-NN Students (kernel weights)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out / "nearest_peers.png", dpi=150)
        plt.close()

    kernel_sims = analysis["similarities"]
    peer_order = np.argsort(kernel_sims)[::-1]
    neighbor_idxs = [
        idx for idx in peer_order if idx != student_idx and kernel_sims[idx] > 0
    ][:K_NEIGHBORS]
    if neighbor_idxs:
        shared = result.mask[student_idx]
        shared_skill_idxs = np.where(shared)[0]
        if len(shared_skill_idxs) > 0 and len(neighbor_idxs) > 0:
            row_labels = [result.student_ids[student_idx]] + [
                result.student_ids[i] for i in neighbor_idxs
            ]
            heat_data = np.vstack(
                [
                    result.scores[student_idx, shared_skill_idxs],
                    *[result.scores[i, shared_skill_idxs] for i in neighbor_idxs],
                ]
            )
            col_labels = [
                result.skills[j].skill_name[:18] for j in shared_skill_idxs[:25]
            ]
            display_data = heat_data[:, :25]
            plt.figure(figsize=(14, max(4, len(row_labels) * 0.5)))
            sns.heatmap(
                display_data,
                cmap="YlGnBu",
                vmin=0,
                vmax=100,
                annot=True,
                fmt=".0f",
                cbar_kws={"label": "Mastery % on shared skills"},
                xticklabels=col_labels,
                yticklabels=row_labels,
            )
            plt.title("Target + k-NN Neighbors on Shared Known Skills")
            plt.xlabel("Shared skills (subset)")
            plt.ylabel("Students")
            plt.tight_layout()
            plt.savefig(out / "kernel_neighbor_heatmap.png", dpi=150)
            plt.close()
