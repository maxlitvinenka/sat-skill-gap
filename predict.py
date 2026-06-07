#!/usr/bin/env python3
"""CLI: run pipeline, export artifacts, optional frontend JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from export_frontend import export_dashboard
from lib import (
    analyze_student,
    build_predicted_missing_skills_df,
    run_pipeline,
    save_figures,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAT skill gap prediction (synthetic data)")
    parser.add_argument("--student-id", default=None, help="Target student name e.g. Emma")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--export-frontend", action="store_true", help="Write dashboard.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = run_pipeline(seed=args.seed)
    idx = result.default_student_idx
    if args.student_id:
        if args.student_id not in result.student_ids:
            raise SystemExit(f"Unknown student {args.student_id}")
        idx = result.student_ids.index(args.student_id)

    analysis = analyze_student(result, idx, top_k=args.top_k)
    recs = analysis["recommendations"]
    sid = result.student_ids[idx]

    out_csv = Path("output") / f"recommendations_{sid.lower()}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(recs).to_csv(out_csv, index=False)

    missing_df = build_predicted_missing_skills_df(
        analysis["predictions"],
        result.mask,
        idx,
        result.skills,
        analysis["breakdown"],
        sid,
    )
    missing_csv = Path("output") / "predicted_missing_skills.csv"
    missing_df.to_csv(missing_csv, index=False)

    save_figures(result, idx, analysis)

    print(f"Target student: {sid}")
    print(f"Observed: {analysis['summary']['observed_skills']} skills | "
          f"{analysis['summary']['total_items_attempted']} items | "
          f"Untested: {analysis['summary']['untested_skills']}")
    print(analysis["summary"]["interpretation"])
    print(f"\nTop 5 recommendations written to {out_csv}")
    print(f"All missing-skill predictions written to {missing_csv}")
    for r in recs[:5]:
        print(
            f"  #{r['rank']} {r['skill_name']} — priority {r['priority_score']:.1f} "
            f"(predicted {r['predicted_mastery']:.0f}%)"
        )

    if args.export_frontend:
        export_dashboard()
        print("Exported frontend/public/data/dashboard.json")


if __name__ == "__main__":
    main()
