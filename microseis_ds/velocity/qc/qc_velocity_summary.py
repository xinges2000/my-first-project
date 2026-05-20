# microseis_ds/velocity/qc/qc_velocity_summary.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

QC_NAME = "velocity_summary"
SUMMARY_FILENAME = "velocity_qc_summary.json"
CONTRACT_REPORT_FILENAME = "qc_contract_velocity.json"
ARTIFACT_REPORT_FILENAME = "qc_artifact_velocity.json"
LAYERIZE_REPORT_FILENAME = "qc_layerize_1d.json"
LEGACY_LAYERIZE_REPORT_FILENAME = "report_layerize_1d.json"
LEGACY_OVERVIEW_REPORT_FILENAME = "report_velocity_overview_4panel.json"

REQUIRED_QC_NAMES = ("contract_qc", "artifact_qc", "layerize_qc")
VALID_STATUS_ORDER = {"passed": 0, "not_run": 1, "warning": 1, "failed": 2}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_posix(path: str | Path | None) -> str:
    if path is None:
        return ""
    return Path(path).as_posix()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return _to_posix(value)
    return str(value)


def _issue(code: str, message: str, *, path: str = "", source: str = "") -> Dict[str, str]:
    item: Dict[str, str] = {"code": code, "message": message}
    if path:
        item["path"] = path
    if source:
        item["source"] = source
    return item


def _normalize_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status in ("passed", "pass", "ok", "success"):
        return "passed"
    if status in ("warning", "warn", "warnings"):
        return "warning"
    if status in ("failed", "fail", "error", "errors"):
        return "failed"
    if status in ("not_run", "not-run", "skipped", "skip", "missing"):
        return "not_run"
    return "warning" if status else "not_run"


def _read_json_report(path: Path) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - filesystem/json-parser dependent
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(obj, dict):
        return None, f"QC report root must be an object, got {type(obj).__name__}"
    return obj, None


def _listify_issues(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                out.append(dict(item))
            else:
                out.append({"code": "UNSTRUCTURED_ISSUE", "message": str(item)})
        return out
    return [{"code": "UNSTRUCTURED_ISSUE", "message": str(value)}]


def _coerce_checked_assets(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(dict(item))
        else:
            out.append({"role": "unknown", "path": str(item), "status": "unknown"})
    return out


def _extract_report_paths(report: Mapping[str, Any], fallback_path: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {"report": _to_posix(fallback_path)}
    for key in ("report_path", "query_csv_path"):
        value = report.get(key)
        if value:
            paths[key] = str(value)
    outputs = report.get("outputs")
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if isinstance(value, str) and value:
                paths[f"output.{key}"] = value
    return paths


def _extract_figure_paths(report: Mapping[str, Any]) -> List[str]:
    figures: List[str] = []
    outputs = report.get("outputs")
    if isinstance(outputs, dict):
        for value in outputs.values():
            if isinstance(value, str) and value.lower().endswith((".png", ".jpg", ".jpeg", ".html")):
                figures.append(value)
    figures_dir = report.get("figures_dir")
    if isinstance(figures_dir, str) and figures_dir:
        figures.append(figures_dir)
    # De-duplicate while preserving order.
    seen = set()
    out: List[str] = []
    for item in figures:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _minimal_qc_record(
    *,
    qc_key: str,
    path: Path,
    required: bool,
    fallback_paths: Iterable[Path] = (),
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str], List[str]]:
    """Load one child QC report and normalize it for summary aggregation."""
    read_path = path
    fallback_used = False
    if not read_path.is_file():
        for fallback in fallback_paths:
            if fallback.is_file():
                read_path = fallback
                fallback_used = True
                break

    summary_warnings: List[Dict[str, Any]] = []
    summary_errors: List[Dict[str, Any]] = []
    report_paths: Dict[str, str] = {qc_key: _to_posix(read_path)}
    figures: List[str] = []

    if not read_path.is_file():
        status = "failed" if required else "not_run"
        issue = _issue(
            "QC_REPORT_MISSING" if required else "OPTIONAL_QC_REPORT_NOT_RUN",
            f"{'Required' if required else 'Optional'} QC report not found: {read_path}",
            path=_to_posix(read_path),
            source=qc_key,
        )
        if required:
            summary_errors.append(issue)
        else:
            summary_warnings.append(issue)
        record = {
            "status": status,
            "report_path": _to_posix(read_path),
            "required": bool(required),
            "checked_assets": [],
            "warnings": [] if required else [issue],
            "errors": [issue] if required else [],
            "results": {},
        }
        return record, [], summary_warnings, summary_errors, report_paths, figures

    raw, read_error = _read_json_report(read_path)
    if read_error is not None or raw is None:
        issue = _issue(
            "QC_REPORT_UNREADABLE",
            f"Failed to read QC report: {read_error}",
            path=_to_posix(read_path),
            source=qc_key,
        )
        if required:
            summary_errors.append(issue)
            status = "failed"
        else:
            summary_warnings.append(issue)
            status = "warning"
        record = {
            "status": status,
            "report_path": _to_posix(read_path),
            "required": bool(required),
            "checked_assets": [],
            "warnings": [] if required else [issue],
            "errors": [issue] if required else [],
            "results": {},
        }
        return record, [], summary_warnings, summary_errors, report_paths, figures

    child_status = _normalize_status(raw.get("status"))
    child_warnings = _listify_issues(raw.get("warnings"))
    child_errors = _listify_issues(raw.get("errors"))
    checked_assets = _coerce_checked_assets(raw.get("checked_assets"))
    results = raw.get("results") if isinstance(raw.get("results"), dict) else {}

    if fallback_used:
        child_warnings = child_warnings + [
            _issue(
                "QC_REPORT_LEGACY_PATH_USED",
                "Primary Step 7 report path was missing; summary used legacy layerize report path.",
                path=_to_posix(read_path),
                source=qc_key,
            )
        ]
        if child_status == "passed":
            child_status = "warning"

    if required and child_status == "not_run":
        issue = _issue(
            "REQUIRED_QC_NOT_RUN",
            f"Required QC report has status not_run: {qc_key}",
            path=_to_posix(read_path),
            source=qc_key,
        )
        child_errors.append(issue)
        child_status = "failed"

    record = {
        "status": child_status,
        "report_path": _to_posix(read_path),
        "required": bool(required),
        "checked_assets_count": len(checked_assets),
        "warning_count": len(child_warnings),
        "error_count": len(child_errors),
        "warnings": child_warnings,
        "errors": child_errors,
        "results": results,
    }

    report_paths.update(_extract_report_paths(raw, read_path))
    figures.extend(_extract_figure_paths(raw))
    return record, checked_assets, child_warnings, child_errors, report_paths, figures


def _derive_overall_status(
    child_records: Mapping[str, Mapping[str, Any]],
    summary_warnings: List[Dict[str, Any]],
    summary_errors: List[Dict[str, Any]],
    *,
    require_overview: bool,
) -> str:
    required_statuses = [_normalize_status(child_records[name].get("status")) for name in REQUIRED_QC_NAMES]
    if any(status == "failed" for status in required_statuses):
        return "failed"
    if summary_errors:
        return "failed"

    overview_status = _normalize_status(child_records.get("overview_qc", {}).get("status"))
    if require_overview and overview_status in ("failed", "not_run"):
        return "failed"

    if any(status == "warning" for status in required_statuses):
        return "warning"
    if overview_status in ("failed", "warning"):
        return "warning"
    if summary_warnings:
        return "warning"
    return "passed"


def _default_qc_root(out_root: str | Path, module_name: str) -> Path:
    return Path(out_root) / "qc" / module_name


def run_velocity_summary_qc(
    *,
    module_name: str = "velocity",
    out_root: str | Path = "prepared_project",
    qc_root: str | Path | None = None,
    reports_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    contract_report_path: str | Path | None = None,
    artifact_report_path: str | Path | None = None,
    layerize_report_path: str | Path | None = None,
    layerize_legacy_report_path: str | Path | None = None,
    overview_report_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    require_overview: bool = False,
    write_report: bool = True,
) -> Dict[str, Any]:
    """Aggregate Step 7 velocity single-module QC reports into one summary.

    This function only aggregates velocity-owned QC reports. It does not check
    velocity -> traveltime cross-module compatibility; that remains Step 8 scope.
    """
    qroot = Path(qc_root) if qc_root is not None else _default_qc_root(out_root, module_name)
    rdir = Path(reports_dir) if reports_dir is not None else (qroot / "reports")
    fdir = Path(figures_dir) if figures_dir is not None else (qroot / "figures")

    paths = {
        "contract_qc": Path(contract_report_path) if contract_report_path is not None else (rdir / CONTRACT_REPORT_FILENAME),
        "artifact_qc": Path(artifact_report_path) if artifact_report_path is not None else (rdir / ARTIFACT_REPORT_FILENAME),
        "layerize_qc": Path(layerize_report_path) if layerize_report_path is not None else (rdir / LAYERIZE_REPORT_FILENAME),
        "overview_qc": Path(overview_report_path) if overview_report_path is not None else (qroot / "qc_velocity_overview_4panel" / LEGACY_OVERVIEW_REPORT_FILENAME),
    }
    layerize_fallbacks = []
    if layerize_legacy_report_path is not None:
        layerize_fallbacks.append(Path(layerize_legacy_report_path))
    else:
        layerize_fallbacks.append(qroot / "qc_layerize_1d" / LEGACY_LAYERIZE_REPORT_FILENAME)

    child_records: Dict[str, Dict[str, Any]] = {}
    checked_assets: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    reports: Dict[str, str] = {}
    figures: List[str] = []

    for qc_key in ("contract_qc", "artifact_qc", "layerize_qc"):
        fallback_paths = layerize_fallbacks if qc_key == "layerize_qc" else ()
        record, assets, child_warnings, child_errors, child_reports, child_figures = _minimal_qc_record(
            qc_key=qc_key,
            path=paths[qc_key],
            required=True,
            fallback_paths=fallback_paths,
        )
        child_records[qc_key] = record
        checked_assets.extend(assets)
        warnings.extend(child_warnings)
        errors.extend(child_errors)
        reports.update(child_reports)
        figures.extend(child_figures)

    overview_record, overview_assets, overview_warnings, overview_errors, overview_reports, overview_figures = _minimal_qc_record(
        qc_key="overview_qc",
        path=paths["overview_qc"],
        required=bool(require_overview),
        fallback_paths=(),
    )
    if not require_overview and overview_record.get("status") == "not_run":
        # Keep explicit weak status without polluting top-level warnings.
        overview_record["weak_assertion"] = True
        overview_warnings = []
    else:
        overview_record["weak_assertion"] = not require_overview
    child_records["overview_qc"] = overview_record
    checked_assets.extend(overview_assets)
    warnings.extend(overview_warnings)
    errors.extend(overview_errors)
    reports.update(overview_reports)
    figures.extend(overview_figures)

    spath = Path(summary_path) if summary_path is not None else (rdir / SUMMARY_FILENAME)
    reports["velocity_summary"] = _to_posix(spath)

    # De-duplicate checked assets and figures while preserving order.
    seen_assets = set()
    dedup_assets: List[Dict[str, Any]] = []
    for asset in checked_assets:
        key = (asset.get("role"), asset.get("path"))
        if key not in seen_assets:
            seen_assets.add(key)
            dedup_assets.append(asset)

    seen_figures = set()
    dedup_figures: List[str] = []
    for figure in figures:
        if figure not in seen_figures:
            seen_figures.add(figure)
            dedup_figures.append(figure)

    overall_status = _derive_overall_status(
        child_records,
        warnings,
        errors,
        require_overview=require_overview,
    )

    summary: Dict[str, Any] = {
        "schema_version": "velocity_qc_summary_v1",
        "module": module_name,
        "qc_name": QC_NAME,
        "overall_status": overall_status,
        "created_at": _now_utc_iso(),
        "qc_root": _to_posix(qroot),
        "reports_dir": _to_posix(rdir),
        "figures_dir": _to_posix(fdir),
        "contract_qc": child_records["contract_qc"],
        "artifact_qc": child_records["artifact_qc"],
        "layerize_qc": child_records["layerize_qc"],
        "overview_qc": child_records["overview_qc"],
        "checked_assets": dedup_assets,
        "warnings": warnings,
        "errors": errors,
        "results": {
            "required_qc": list(REQUIRED_QC_NAMES),
            "required_statuses": {name: child_records[name]["status"] for name in REQUIRED_QC_NAMES},
            "overview_required": bool(require_overview),
            "overview_status": child_records["overview_qc"]["status"],
            "checked_asset_count": len(dedup_assets),
            "warning_count": len(warnings),
            "error_count": len(errors),
            "report_count": len(reports),
            "figure_count": len(dedup_figures),
        },
        "reports": reports,
        "figures": dedup_figures,
        "report_path": _to_posix(spath),
    }

    if write_report:
        write_velocity_qc_summary(summary, spath)
    return summary


def write_velocity_qc_summary(summary: Mapping[str, Any], summary_path: str | Path) -> Path:
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qc_velocity_summary",
        description="Aggregate velocity single-module QC reports into velocity_qc_summary.json.",
    )
    p.add_argument("--module_name", default="velocity", help="Module name. Default: velocity")
    p.add_argument("--out_root", default="prepared_project", help="Root output directory. Default: prepared_project")
    p.add_argument("--qc_root", default=None, help="Override QC root. Default: <out_root>/qc/<module_name>")
    p.add_argument("--reports_dir", default=None, help="Override reports dir. Default: <qc_root>/reports")
    p.add_argument("--figures_dir", default=None, help="Override figures dir. Default: <qc_root>/figures")
    p.add_argument("--contract_report_path", default=None, help="Override contract QC report path.")
    p.add_argument("--artifact_report_path", default=None, help="Override artifact QC report path.")
    p.add_argument("--layerize_report_path", default=None, help="Override layerize QC report path.")
    p.add_argument("--layerize_legacy_report_path", default=None, help="Optional legacy layerize report fallback path.")
    p.add_argument("--overview_report_path", default=None, help="Optional overview report path.")
    p.add_argument("--summary_path", default=None, help="Override output velocity_qc_summary.json path.")
    p.add_argument("--require_overview", action="store_true", help="If set, overview missing/failed can fail the summary.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    summary = run_velocity_summary_qc(
        module_name=args.module_name,
        out_root=args.out_root,
        qc_root=args.qc_root,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        contract_report_path=args.contract_report_path,
        artifact_report_path=args.artifact_report_path,
        layerize_report_path=args.layerize_report_path,
        layerize_legacy_report_path=args.layerize_legacy_report_path,
        overview_report_path=args.overview_report_path,
        summary_path=args.summary_path,
        require_overview=args.require_overview,
        write_report=True,
    )
    print("[OK] qc_velocity_summary finished")
    print(f"  overall_status: {summary['overall_status']}")
    print(f"  report_path   : {Path(str(summary['report_path'])).resolve()}")
    print(f"  errors        : {len(summary.get('errors', []))}")
    print(f"  warnings      : {len(summary.get('warnings', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
