"""Export dashboard JSON for the React frontend."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lib import (
    K_NEIGHBORS,
    KERNEL_SIGMA,
    N_SKILLS,
    PROPAGATION_ALPHA,
    RANDOM_SEED,
    analyze_student,
    build_methodology_demo,
    run_pipeline,
)


def _export_methodology_demo(result, idx, analysis) -> dict:
    demo = build_methodology_demo(result, idx, analysis)
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


def export_dashboard(
    output_path: Path | None = None,
    sample_indices: list[int] | None = None,
) -> dict:
    result = run_pipeline()
    if sample_indices is None:
        sample_indices = sorted(
            {
                result.default_student_idx,
                0,
                10,
                25,
                42,
                50,
            }
        )

    by_student = {}
    students_meta = []

    for idx in range(len(result.student_ids)):
        sid = result.student_ids[idx]
        observed = int(result.mask[idx].sum())
        hidden = N_SKILLS - observed
        item_count = len(result.responses_df[result.responses_df["student_id"] == sid])
        students_meta.append(
            {
                "id": sid,
                "label": f"Student {sid}",
                "observedCount": observed,
                "hiddenCount": hidden,
                "itemCount": item_count,
            }
        )

    for idx in sample_indices:
        sid = result.student_ids[idx]
        analysis = analyze_student(result, idx)
        by_student[sid] = {
            "summary": {
                "observedSkills": analysis["summary"]["observed_skills"],
                "untestedSkills": analysis["summary"]["untested_skills"],
                "totalItemsAttempted": analysis["summary"]["total_items_attempted"],
                "avgItemsPerSkill": analysis["summary"]["avg_items_per_skill"],
                "topPriorityScore": analysis["summary"]["top_priority_score"],
                "interpretation": analysis["summary"]["interpretation"],
            },
            "observedWork": [
                {
                    "skillId": w["skill_id"],
                    "skillName": w["skill_name"],
                    "category": w["category"],
                    "attempted": w["attempted"],
                    "correct": w["correct"],
                    "mastery": w["mastery"],
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
                        for it in w["items"]
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
                {
                    "studentId": p["student_id"],
                    "similarity": p["similarity"],
                }
                for p in analysis["peers"]
            ],
            "methodologyDemo": _export_methodology_demo(result, idx, analysis),
        }

    payload = {
        "meta": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "isSynthetic": True,
            "algorithm": "kernel-knn-label-propagation",
            "scoreSource": "item-responses",
            "randomSeed": RANDOM_SEED,
            "kernelSigma": KERNEL_SIGMA,
            "kNeighbors": K_NEIGHBORS,
            "propagationAlpha": PROPAGATION_ALPHA,
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
        },
        "students": students_meta,
        "defaultStudentId": result.student_ids[result.default_student_idx],
        "byStudent": by_student,
    }

    if output_path is None:
        output_path = Path("frontend/public/data/dashboard.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    export_dashboard()
    print("Wrote frontend/public/data/dashboard.json")
