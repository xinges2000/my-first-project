from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scripts import build_velocity, qc_velocity


SUMMARY_FILENAME = "velocity_qc_summary.json"
CONTRACT_REPORT_FILENAME = "qc_contract_velocity.json"
ARTIFACT_REPORT_FILENAME = "qc_artifact_velocity.json"
ARTIFACT_QUERY_CSV_FILENAME = "qc_velocity_artifact_query.csv"
LAYERIZE_REPORT_FILENAME = "qc_layerize_1d.json"
SUMMARY_REQUIRED_FIELDS = {
    "module",
    "overall_status",
    "contract_qc",
    "artifact_qc",
    "layerize_qc",
    "overview_qc",
    "checked_assets",
    "warnings",
    "errors",
    "results",
    "reports",
}
CORE_QC_KEYS = ("contract_qc", "artifact_qc", "layerize_qc")
NON_BLOCKING_STATUSES = {"passed", "warning"}
OVERVIEW_NON_BLOCKING_STATUSES = {"not_run", "passed", "warning"}


def _load_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"expected JSON file to exist: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _run_velocity_normal_build_chain(test_run_root: Path, velocity_normal_inputs: Any) -> None:
    """Build the minimum normal velocity outputs consumed by Step 7 QC."""
    log_csv = velocity_normal_inputs.ref_engineering.engineering_log_csv

    rc = build_velocity.main(
        [
            "--task",
            "engineering",
            "--out_root",
            str(test_run_root),
            "--module_name",
            "velocity",
            "--log_csv",
            str(log_csv),
            "--ref_depth_axis",
            "md",
        ]
    )
    assert rc == 0

    rc = build_velocity.main(
        [
            "--task",
            "layerize",
            "--out_root",
            str(test_run_root),
            "--module_name",
            "velocity",
            "--ref_depth_axis",
            "md",
            "--quiet",
        ]
    )
    assert rc == 0


def _run_velocity_qc_all(test_run_root: Path) -> Path:
    """Run the public Step 7 single-module QC entry and return the summary path."""
    rc = qc_velocity.main(
        [
            "--task",
            "all",
            "--out_root",
            str(test_run_root),
            "--module_name",
            "velocity",
            "--ref_depth_axis",
            "md",
            # Keep artifact query depths inside the normal sample range so the
            # minimum Step 7.5 test focuses on aggregation rather than warning cases.
            "--artifact_z",
            "0",
            "100",
            "200",
        ]
    )
    assert rc == 0
    return test_run_root / "qc" / "velocity" / "reports" / SUMMARY_FILENAME


def _assert_status_is_expressed(record: Mapping[str, Any], *, allowed: set[str]) -> None:
    status = str(record.get("status", ""))
    assert status, f"QC record should expose a non-empty status: {record}"
    assert status in allowed, f"unexpected QC status {status!r}; record={record}"


def test_velocity_qc_summary_is_generated_and_aggregates_core_qc_reports(
    test_run_root: Path,
    velocity_normal_inputs: Any,
) -> None:
    """
    Step 7.5 minimum regression:
    - normal velocity outputs are generated in an isolated test root;
    - scripts.qc_velocity --task all writes velocity_qc_summary.json;
    - contract/artifact/layerize QC records are aggregated;
    - overview remains weak/non-blocking when it is not requested.
    """
    _run_velocity_normal_build_chain(
        test_run_root=test_run_root,
        velocity_normal_inputs=velocity_normal_inputs,
    )

    summary_path = _run_velocity_qc_all(test_run_root)
    reports_dir = test_run_root / "qc" / "velocity" / "reports"
    figures_dir = test_run_root / "qc" / "velocity" / "figures"

    assert reports_dir.is_dir()
    assert summary_path == reports_dir / SUMMARY_FILENAME
    assert summary_path.is_file()

    summary = _load_json(summary_path)

    assert SUMMARY_REQUIRED_FIELDS.issubset(summary.keys())
    assert summary["module"] == "velocity"
    assert str(summary["overall_status"]) in NON_BLOCKING_STATUSES

    for qc_key in CORE_QC_KEYS:
        record = summary[qc_key]
        assert isinstance(record, dict), f"{qc_key} should be a structured object"
        _assert_status_is_expressed(record, allowed=NON_BLOCKING_STATUSES)
        assert record.get("report_path"), f"{qc_key} should expose report_path"

    overview_qc = summary["overview_qc"]
    assert isinstance(overview_qc, dict)
    _assert_status_is_expressed(
        overview_qc,
        allowed=OVERVIEW_NON_BLOCKING_STATUSES,
    )
    assert overview_qc["status"] != "failed"
    assert overview_qc.get("required") is False

    # Required Step 7 reports are produced under qc/velocity/reports/.
    for filename in (
        CONTRACT_REPORT_FILENAME,
        ARTIFACT_REPORT_FILENAME,
        ARTIFACT_QUERY_CSV_FILENAME,
        LAYERIZE_REPORT_FILENAME,
        SUMMARY_FILENAME,
    ):
        assert (reports_dir / filename).is_file(), f"expected report output: {filename}"

    reports = summary["reports"]
    assert isinstance(reports, dict)
    for key in ("contract_qc", "artifact_qc", "layerize_qc", "velocity_summary"):
        assert key in reports, f"summary.reports missing {key}"
        assert "qc/velocity/reports" in Path(str(reports[key])).as_posix()

    assert isinstance(summary["checked_assets"], list)
    assert isinstance(summary["warnings"], list)
    assert isinstance(summary["errors"], list)
    assert isinstance(summary["results"], dict)

    # Figures are weak assertions: if layerize produced figures, only their paths
    # and existence are checked; no image-content or pixel-level regression.
    figures = summary.get("figures", [])
    assert isinstance(figures, list)
    if figures:
        assert figures_dir.is_dir()
        for figure in figures:
            figure_path = Path(str(figure))
            if not figure_path.is_absolute():
                figure_path = test_run_root / figure_path
            assert figure_path.exists(), f"weak figure path should exist if reported: {figure}"

    # Step 7.5 remains single-module velocity QC. It must not route through Step 8
    # cross-module QC paths, scripts, or test locations.
    serialized_summary = json.dumps(summary, ensure_ascii=False)
    assert "velocity_traveltime" not in serialized_summary
    assert "scripts/qc_velocity_traveltime.py" not in serialized_summary
    assert "tests/cross_module" not in serialized_summary
