"""Export dashboard JSON for the React frontend."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lib import N_SKILLS, analyze_student, run_pipeline


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
                    "itemSimilarity": p.get("item_similarity", 0),
                }
                for p in analysis["peers"]
            ],
        }

    payload = {
        "meta": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "isSynthetic": True,
            "algorithm": "cosine-similarity",
            "scoreSource": "item-responses",
            "nStudents": len(result.student_ids),
            "nSkills": N_SKILLS,
            "nItems": len(result.item_ids),
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
