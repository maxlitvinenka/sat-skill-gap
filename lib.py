"""Core linear-algebra pipeline for SAT skill gap prediction (synthetic data only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from skills import (
    CATEGORY_LATENT_LOADINGS,
    LATENT_DIMS,
    LEVEL_INTERCEPT,
    SKILLS,
    Category,
    Skill,
)

N_STUDENTS = 75
N_SKILLS = 72
MASK_FRACTION = 0.40
RANDOM_SEED = 42
TOP_K_PEERS = 15


@dataclass
class PipelineResult:
    scores: np.ndarray
    mask: np.ndarray
    student_ids: list[str]
    skill_ids: list[str]
    skills: list[Skill]
    latent_profiles: np.ndarray
    default_student_idx: int


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


def _latent_score(skill: Skill, latent: np.ndarray) -> float:
    loadings = CATEGORY_LATENT_LOADINGS[skill.category]
    weighted = sum(latent[LATENT_DIMS.index(k)] * w for k, w in loadings.items())
    intercept = LEVEL_INTERCEPT[skill.level] / 100.0
    raw = 35 + 55 * (0.55 * weighted + 0.45 * intercept)
    return float(np.clip(raw, 20, 95))


def generate_scores(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = N_STUDENTS
    m = N_SKILLS
    latent_profiles = np.vstack([_archetype_latent(rng) for _ in range(n)])
    scores = np.zeros((n, m), dtype=float)
    skill_index = {s.skill_id: i for i, s in enumerate(SKILLS)}

    for j, skill in enumerate(SKILLS):
        for i in range(n):
            scores[i, j] = _latent_score(skill, latent_profiles[i])

    for _ in range(3):
        updated = scores.copy()
        for j, skill in enumerate(SKILLS):
            if not skill.prerequisites:
                continue
            prereq_idxs = [skill_index[p] for p in skill.prerequisites]
            prereq_mean = scores[:, prereq_idxs].mean(axis=1)
            for i in range(n):
                latent_part = _latent_score(skill, latent_profiles[i])
                noise = rng.normal(0, 3.5)
                updated[i, j] = np.clip(
                    0.55 * latent_part + 0.35 * prereq_mean[i] + 0.10 * scores[i, j] + noise,
                    0,
                    100,
                )
        scores = updated

    return scores, latent_profiles


def generate_mask(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    mask = np.ones((n, m), dtype=bool)
    for i in range(n):
        hide_count = int(round(m * MASK_FRACTION))
        hide_count = max(5, min(m - 25, hide_count))
        hidden = rng.choice(m, size=hide_count, replace=False)
        mask[i, hidden] = False
    return mask


def cosine_similarity_observed(
    target: np.ndarray,
    peer: np.ndarray,
    observed: np.ndarray,
) -> float:
    """Cosine similarity on shared observed skill dimensions."""
    idx = observed
    u = target[idx]
    v = peer[idx]
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def predict_missing_scores(
    scores: np.ndarray,
    mask: np.ndarray,
    target_idx: int,
    top_k: int = TOP_K_PEERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted scores for all skills and peer similarities."""
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


def rank_recommendations(
    predictions: np.ndarray,
    mask: np.ndarray,
    target_idx: int,
    skills: list[Skill],
) -> pd.DataFrame:
    rows = []
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
                    f"Predicted {pred:.0f}% mastery on untested {skill.level.lower()} "
                    f"{skill.category.lower()} skill; foundational weight {skill.foundational_weight:.1f}."
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def build_interpretation(recs: pd.DataFrame, top_n: int = 3) -> str:
    if recs.empty:
        return "Insufficient untested skills to generate recommendations."
    top = recs.head(top_n)
    names = ", ".join(top["skill_name"].tolist())
    categories = ", ".join(sorted(set(top["category"].tolist())))
    return (
        f"The model predicts this student is likely weak in {names}. "
        f"Because these are foundational {categories.lower()} skills and similar students "
        f"with the same profile struggled in them, the system recommends testing or "
        f"practicing these skills next."
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
                    "score": round(float(scores[i, j]), 1),
                    "is_observed": bool(mask[i, j]),
                }
            )
    return pd.DataFrame(rows)


def run_pipeline(seed: int = RANDOM_SEED) -> PipelineResult:
    rng = np.random.default_rng(seed)
    write_skill_metadata(Path("data/skill_metadata.csv"))
    scores, latent_profiles = generate_scores(rng)
    mask = generate_mask(rng, N_STUDENTS, N_SKILLS)
    student_ids = [f"S{i + 1:03d}" for i in range(N_STUDENTS)]
    skill_ids = [s.skill_id for s in SKILLS]
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
    )


def analyze_student(
    result: PipelineResult,
    student_idx: int,
    top_k: int = TOP_K_PEERS,
) -> dict[str, Any]:
    predictions, sims = predict_missing_scores(result.scores, result.mask, student_idx, top_k)
    recs = rank_recommendations(predictions, result.mask, student_idx, result.skills)
    peer_order = np.argsort(sims)[::-1]
    peers = []
    for idx in peer_order:
        if idx == student_idx or sims[idx] <= 0:
            continue
        peers.append({"student_id": result.student_ids[idx], "similarity": round(float(sims[idx]), 3)})
        if len(peers) >= 10:
            break
    observed = int(result.mask[student_idx].sum())
    hidden = N_SKILLS - observed
    top_priority = float(recs.iloc[0]["priority_score"]) if not recs.empty else 0.0
    return {
        "summary": {
            "observed_skills": observed,
            "untested_skills": hidden,
            "top_priority_score": round(top_priority, 1),
            "interpretation": build_interpretation(recs),
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
        cbar_kws={"label": "Mastery %"},
        xticklabels=False,
        yticklabels=[result.student_ids[i] for i in subset_students],
    )
    plt.title("Student-Skill Matrix (NaN = untested)")
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
        plt.ylabel("Cosine Similarity")
        plt.title("Nearest Similar Students")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out / "nearest_peers.png", dpi=150)
        plt.close()
