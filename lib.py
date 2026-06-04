"""Core linear-algebra pipeline for SAT skill gap prediction (synthetic data only)."""

from __future__ import annotations

from dataclasses import dataclass
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

N_STUDENTS = 75
N_SKILLS = 72
MASK_FRACTION = 0.40
RANDOM_SEED = 42
TOP_K_PEERS = 15
ITEMS_PER_SKILL = 3
CHOICES = ("A", "B", "C", "D")


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
    """Which skill indices each student is tested on (~60% coverage)."""
    tested = np.zeros((N_STUDENTS, n_skills), dtype=bool)
    for i in range(N_STUDENTS):
        n_test = int(round(n_skills * (1 - MASK_FRACTION)))
        n_test = max(25, min(n_skills - 5, n_test))
        tested[i, rng.choice(n_skills, size=n_test, replace=False)] = True
    return tested


def generate_all_responses(
    rng: np.random.Generator,
    latent_profiles: np.ndarray,
    tested_skills: np.ndarray,
) -> pd.DataFrame:
    skill_by_id = {s.skill_id: s for s in SKILLS}
    rows: list[dict[str, Any]] = []

    for i in range(N_STUDENTS):
        sid = f"S{i + 1:03d}"
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


def cosine_similarity_observed(
    target: np.ndarray,
    peer: np.ndarray,
    observed: np.ndarray,
) -> float:
    idx = observed
    u = target[idx]
    v = peer[idx]
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def item_cosine_similarity(
    R: np.ndarray,
    target_idx: int,
    peer_idx: int,
) -> float:
    """Cosine similarity on shared attempted items (both layers demo)."""
    u = R[target_idx]
    v = R[peer_idx]
    shared = ~np.isnan(u) & ~np.isnan(v)
    if shared.sum() < 5:
        return 0.0
    return cosine_similarity_observed(u, v, shared)


def predict_missing_scores(
    scores: np.ndarray,
    mask: np.ndarray,
    target_idx: int,
    top_k: int = TOP_K_PEERS,
) -> tuple[np.ndarray, np.ndarray]:
    n, m = scores.shape
    target = scores[target_idx]
    target_obs = mask[target_idx]

    sims = np.zeros(n, dtype=float)
    for i in range(n):
        if i == target_idx:
            continue
        peer_obs = mask[i]
        shared = target_obs & peer_obs
        if shared.sum() < 5:
            sims[i] = 0.0
            continue
        sims[i] = cosine_similarity_observed(target, scores[i], shared)

    predictions = np.full(m, np.nan)
    for j in range(m):
        if target_obs[j]:
            predictions[j] = target[j]
            continue
        peer_mask = mask[:, j] & (np.arange(n) != target_idx)
        peer_idxs = np.where(peer_mask)[0]
        if len(peer_idxs) == 0:
            continue
        peer_sims = sims[peer_idxs]
        positive = peer_sims > 0
        if not positive.any():
            continue
        peer_idxs = peer_idxs[positive]
        peer_sims = peer_sims[positive]
        order = np.argsort(peer_sims)[::-1][:top_k]
        chosen = peer_idxs[order]
        weights = peer_sims[order]
        predictions[j] = np.dot(weights, scores[chosen, j]) / weights.sum()

    return predictions, sims


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


def rank_recommendations(
    predictions: np.ndarray,
    mask: np.ndarray,
    target_idx: int,
    skills: list[Skill],
    observed_stats: dict[str, dict[str, Any]],
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
        if mask[target_idx, j]:
            continue
        pred = predictions[j]
        if np.isnan(pred):
            continue
        priority = (100 - pred) * skill.foundational_weight
        rows.append(
            {
                "skill_id": skill.skill_id,
                "skill_name": skill.skill_name,
                "category": skill.category,
                "level": skill.level,
                "predicted_mastery": round(pred, 1),
                "foundational_weight": skill.foundational_weight,
                "priority_score": round(priority, 1),
                "reason": (
                    f"Predicted {pred:.0f}% on untested {skill.level.lower()} {skill.category.lower()} skill;"
                    f"{weak_hint} similar peers with overlapping {', '.join(sorted(observed_cats)) or 'tested'} skills also struggled here."
                ).replace("  ", " "),
            }
        )
    df = pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def build_interpretation(recs: pd.DataFrame, observed_stats: dict[str, dict[str, Any]], top_n: int = 3) -> str:
    if recs.empty:
        return "Insufficient untested skills to generate recommendations."
    top = recs.head(top_n)
    names = ", ".join(top["skill_name"].tolist())
    categories = ", ".join(sorted(set(top["category"].tolist())))
    weak = sorted(observed_stats.values(), key=lambda x: x["mastery"])[:2]
    work_note = ""
    if weak:
        work_note = (
            f" Based on item responses, the student scored {weak[0]['mastery']:.0f}% on "
            f"{weak[0]['skill_name']} ({weak[0]['correct']}/{weak[0]['attempted']} correct)."
        )
    return (
        f"The model predicts this student is likely weak in {names}.{work_note} "
        f"Because these are foundational {categories.lower()} skills and similar students "
        f"with comparable response profiles struggled in them, the system recommends testing "
        f"or practicing these skills next."
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


def run_pipeline(seed: int = RANDOM_SEED) -> PipelineResult:
    rng = np.random.default_rng(seed)
    write_skill_metadata(Path("data/skill_metadata.csv"))
    write_item_bank(Path("data/item_bank.csv"))

    latent_profiles = np.vstack([_archetype_latent(rng) for _ in range(N_STUDENTS)])
    tested_skills = _assign_tested_skills(rng, N_SKILLS)
    responses_df = generate_all_responses(rng, latent_profiles, tested_skills)
    responses_df.to_csv("data/student_responses.csv", index=False)

    student_ids = [f"S{i + 1:03d}" for i in range(N_STUDENTS)]
    skill_ids = [s.skill_id for s in SKILLS]
    item_ids = [i.item_id for i in ITEMS]

    scores, mask = responses_to_skill_matrix(responses_df, student_ids, SKILLS)
    R = build_response_matrix(responses_df, item_ids, student_ids)
    Q = build_skill_item_matrix(item_ids)

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
    )


def analyze_student(
    result: PipelineResult,
    student_idx: int,
    top_k: int = TOP_K_PEERS,
) -> dict[str, Any]:
    sid = result.student_ids[student_idx]
    observed_stats = _observed_skill_stats(result.responses_df, sid)
    predictions, sims = predict_missing_scores(result.scores, result.mask, student_idx, top_k)
    recs = rank_recommendations(
        predictions, result.mask, student_idx, result.skills, observed_stats
    )
    peer_order = np.argsort(sims)[::-1]
    peers = []
    for idx in peer_order:
        if idx == student_idx or sims[idx] <= 0:
            continue
        item_sim = item_cosine_similarity(result.response_matrix, student_idx, idx)
        peers.append(
            {
                "student_id": result.student_ids[idx],
                "similarity": round(float(sims[idx]), 3),
                "item_similarity": round(item_sim, 3),
            }
        )
        if len(peers) >= 10:
            break

    observed = int(result.mask[student_idx].sum())
    hidden = N_SKILLS - observed
    total_items = len(result.responses_df[result.responses_df["student_id"] == sid])
    avg_items = round(total_items / observed, 1) if observed else 0.0
    top_priority = float(recs.iloc[0]["priority_score"]) if not recs.empty else 0.0
    observed_work = build_observed_work(result.responses_df, sid)

    return {
        "summary": {
            "observed_skills": observed,
            "untested_skills": hidden,
            "total_items_attempted": total_items,
            "avg_items_per_skill": avg_items,
            "top_priority_score": round(top_priority, 1),
            "interpretation": build_interpretation(recs, observed_stats),
        },
        "recommendations": [
            {
                "rank": int(row.rank),
                "skill_id": row.skill_id,
                "skill_name": row.skill_name,
                "category": row.category,
                "level": row.level,
                "predicted_mastery": float(row.predicted_mastery),
                "foundational_weight": float(row.foundational_weight),
                "priority_score": float(row.priority_score),
                "reason": row.reason,
            }
            for row in recs.itertuples()
        ],
        "peers": peers,
        "observed_work": observed_work,
        "predictions": predictions,
        "similarities": sims,
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
        plt.ylabel("Cosine Similarity (skill vectors)")
        plt.title("Nearest Similar Students")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out / "nearest_peers.png", dpi=150)
        plt.close()
