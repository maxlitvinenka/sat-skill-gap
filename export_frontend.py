"""Export dashboard JSON for the React frontend."""

from __future__ import annotations

import json
from pathlib import Path

from lib import (
    K_NEIGHBORS,
    KERNEL_SIGMA,
    PROPAGATION_ALPHA,
    RANDOM_SEED,
    analyze_student,
    build_computation_trace,
    build_dashboard_meta,
    build_students_meta,
    run_pipeline,
    serialize_student_dashboard,
)


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
    students_meta = build_students_meta(result)

    for idx in sample_indices:
        sid = result.student_ids[idx]
        analysis = analyze_student(result, idx)
        trace = build_computation_trace(result, idx, analysis)
        by_student[sid] = serialize_student_dashboard(
            result, idx, analysis, computation_trace=trace
        )

    payload = {
        "meta": build_dashboard_meta(result),
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
