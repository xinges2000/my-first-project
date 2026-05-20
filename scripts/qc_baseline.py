#!/usr/bin/env python
# scripts/qc_baseline.py
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional


FORMAL_QC_ITEMS = ["contract", "overlay"]
AUXILIARY_QC_ITEMS = ["coor"]
KEEP_MODULE_ERROR_CODES = {"contract_module_error", "overlay_module_error", "coor_module_error"}


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _json_dump(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _json_load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _call_module_main(module, passed_argv: List[str]) -> int:
    if not hasattr(module, "main"):
        raise RuntimeError(f"{module.__name__} has no main().")
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + passed_argv
        ret = module.main()
        return int(ret) if ret is not None else 0
    finally:
        sys.argv = old_argv


def _run_module_check(
    *,
    check_name: str,
    module,
    argv: List[str],
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    started_at = _now_ts()
    result: Dict[str, Any] = {
        "check_name": check_name,
        "started_at": started_at,
        "status": "failed",
        "exit_code": None,
        "output_json": output_json,
        "error": None,
        "traceback": None,
    }
    try:
        ret = _call_module_main(module, argv)
        result["exit_code"] = int(ret)

        payload: Optional[Dict[str, Any]] = None
        if output_json and os.path.exists(output_json):
            payload = _json_load(output_json)
            result["payload"] = payload
            result["status"] = _normalize_check_status(payload.get("status", "failed" if ret != 0 else "passed"))
        elif ret == 0:
            result["status"] = _normalize_check_status("passed")

        if ret != 0:
            result["error"] = f"module returned non-zero exit code: {ret}"
            if payload is None:
                result["status"] = "failed"
    except Exception as exc:
        result["exit_code"] = 1
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    result["finished_at"] = _now_ts()
    return result


def _find_result(results: List[Dict[str, Any]], check_name: str) -> Optional[Dict[str, Any]]:
    for item in results:
        if item.get("check_name") == check_name:
            return item
    return None

def _normalize_check_status(value: Any) -> str:
    status = str(value or "unknown").strip().lower()
    if status in ("pass", "passed"):
        return "passed"
    if status in ("fail", "failed"):
        return "failed"
    if status == "not_requested":
        return "not_requested"
    if status == "skipped":
        return "skipped"
    return status


def _dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _normalize_issue_entry(
    issue: Any,
    *,
    default_severity: str,
    default_code: str,
    source_check: str,
) -> Dict[str, Any]:
    if isinstance(issue, dict):
        normalized = dict(issue)
        normalized.setdefault("severity", default_severity)
        normalized.setdefault("code", default_code)
        normalized.setdefault("message", str(issue.get("message", issue.get("code", default_code))))
    else:
        normalized = {
            "severity": default_severity,
            "code": default_code,
            "message": str(issue),
        }

    normalized["severity"] = str(normalized.get("severity", default_severity)).lower()
    normalized["code"] = str(normalized.get("code", default_code))
    normalized["message"] = str(normalized.get("message", normalized["code"]))
    normalized["source_check"] = source_check
    return normalized


def _normalize_issue_list(
    issues: Any,
    *,
    default_severity: str,
    default_code_prefix: str,
    source_check: str,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, issue in enumerate(list(issues or [])):
        normalized.append(
            _normalize_issue_entry(
                issue,
                default_severity=default_severity,
                default_code=f"{default_code_prefix}_{idx}",
                source_check=source_check,
            )
        )
    return normalized


def _extract_issue_dict_list(
    node: Any,
    *,
    source_check: str,
    default_severity: str,
    default_code_prefix: str,
) -> List[Dict[str, Any]]:
    if isinstance(node, list):
        return _normalize_issue_list(
            node,
            default_severity=default_severity,
            default_code_prefix=default_code_prefix,
            source_check=source_check,
        )
    return []


def _extract_contract_payload_issues(payload: Any, *, source_check: str) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    collected: List[Dict[str, Any]] = []

    collected.extend(
        _extract_issue_dict_list(
            payload.get("issues"),
            source_check=source_check,
            default_severity="error",
            default_code_prefix=f"{source_check}_issue",
        )
    )
    collected.extend(
        _extract_issue_dict_list(
            payload.get("errors"),
            source_check=source_check,
            default_severity="error",
            default_code_prefix=f"{source_check}_error",
        )
    )
    collected.extend(
        _extract_issue_dict_list(
            payload.get("warnings"),
            source_check=source_check,
            default_severity="warning",
            default_code_prefix=f"{source_check}_warning",
        )
    )

    counts = payload.get("counts", {}) if isinstance(payload.get("counts"), dict) else {}
    stations_qc = payload.get("stations_qc", {}) if isinstance(payload.get("stations_qc"), dict) else {}
    manifest_qc = payload.get("manifest_qc", {}) if isinstance(payload.get("manifest_qc"), dict) else {}
    wellpaths_qc = payload.get("wellpaths_qc", {}) if isinstance(payload.get("wellpaths_qc"), dict) else {}
    baseline_build_summary_qc = (
        payload.get("baseline_build_summary_qc", {})
        if isinstance(payload.get("baseline_build_summary_qc"), dict)
        else {}
    )
    wellpath_build_summary_qc = (
        payload.get("wellpath_build_summary_qc", {})
        if isinstance(payload.get("wellpath_build_summary_qc"), dict)
        else {}
    )

    nested_sections = [
        ("stations_qc", stations_qc),
        ("manifest_qc", manifest_qc),
        ("wellpaths_qc", wellpaths_qc),
        ("baseline_build_summary_qc", baseline_build_summary_qc),
        ("wellpath_build_summary_qc", wellpath_build_summary_qc),
        ("counts", counts),
    ]

    for section_name, section_payload in nested_sections:
        if not isinstance(section_payload, dict):
            continue

        collected.extend(
            _extract_issue_dict_list(
                section_payload.get("issues"),
                source_check=source_check,
                default_severity="error",
                default_code_prefix=f"{source_check}_{section_name}_issue",
            )
        )
        collected.extend(
            _extract_issue_dict_list(
                section_payload.get("errors"),
                source_check=source_check,
                default_severity="error",
                default_code_prefix=f"{source_check}_{section_name}_error",
            )
        )
        collected.extend(
            _extract_issue_dict_list(
                section_payload.get("warnings"),
                source_check=source_check,
                default_severity="warning",
                default_code_prefix=f"{source_check}_{section_name}_warning",
            )
        )

    for idx, item in enumerate(_safe_list(wellpaths_qc.get("items"))):
        if not isinstance(item, dict):
            continue
        per_item_prefix = f"{source_check}_wellpath_item_{idx}"
        collected.extend(
            _extract_issue_dict_list(
                item.get("issues"),
                source_check=source_check,
                default_severity="error",
                default_code_prefix=f"{per_item_prefix}_issue",
            )
        )
        collected.extend(
            _extract_issue_dict_list(
                item.get("errors"),
                source_check=source_check,
                default_severity="error",
                default_code_prefix=f"{per_item_prefix}_error",
            )
        )
        collected.extend(
            _extract_issue_dict_list(
                item.get("warnings"),
                source_check=source_check,
                default_severity="warning",
                default_code_prefix=f"{per_item_prefix}_warning",
            )
        )

    return collected


def _extract_payload_issues(payload: Any, *, source_check: str) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    if source_check == "contract":
        return _extract_contract_payload_issues(payload, source_check=source_check)

    issues = _normalize_issue_list(
        payload.get("issues", []),
        default_severity="error",
        default_code_prefix=f"{source_check}_issue",
        source_check=source_check,
    )
    warnings = _normalize_issue_list(
        payload.get("warnings", []),
        default_severity="warning",
        default_code_prefix=f"{source_check}_warning",
        source_check=source_check,
    )
    errors = _normalize_issue_list(
        payload.get("errors", []),
        default_severity="error",
        default_code_prefix=f"{source_check}_error",
        source_check=source_check,
    )
    return issues + errors + warnings


def _extract_result_issues(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_check = str(result.get("check_name", "unknown"))
    issues: List[Dict[str, Any]] = []

    issues.extend(_extract_payload_issues(result.get("payload"), source_check=source_check))

    if result.get("error"):
        issues.append(
            {
                "severity": "error",
                "code": f"{source_check}_module_error",
                "message": str(result["error"]),
                "source_check": source_check,
                "aggregated_from": "module_runner",
            }
        )

    return issues


def _collect_issue_codes(issues: List[Dict[str, Any]]) -> List[str]:
    codes: List[str] = []
    for issue in issues:
        code = issue.get("code")
        if code is not None:
            codes.append(str(code))
    return _dedupe_keep_order(codes)


def _dedupe_issues_keep_order(issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for issue in issues:
        severity = str(issue.get("severity", "")).lower()
        code = str(issue.get("code", ""))
        message = str(issue.get("message", ""))
        source_check = str(issue.get("source_check", ""))
        key = (severity, code, message, source_check)
        if key in seen:
            continue
        seen.add(key)
        out.append(issue)
    return out


def _infer_issue_counts(issues: List[Dict[str, Any]]) -> Dict[str, int]:
    error_count = 0
    warning_count = 0
    for issue in issues:
        if str(issue.get("severity", "")).lower() == "warning":
            warning_count += 1
        else:
            error_count += 1
    return {
        "error_count": error_count,
        "warning_count": warning_count,
    }


def _pick_first_int(*values: Any, default: int = 0) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return default


def _extract_checked_wellpath_count(payload: Dict[str, Any]) -> int:
    counts = payload.get("counts", {}) if isinstance(payload.get("counts"), dict) else {}
    wellpaths_qc = payload.get("wellpaths_qc", {}) if isinstance(payload.get("wellpaths_qc"), dict) else {}

    items = _safe_list(wellpaths_qc.get("items"))
    if items:
        return len(items)

    return _pick_first_int(
        wellpaths_qc.get("checked_wellpath_count"),
        wellpaths_qc.get("count"),
        counts.get("checked_wellpath_count"),
        counts.get("wellpath_count"),
        counts.get("checked_count"),
        default=0,
    )


def _build_contract_qc_summary(contract_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not contract_result:
        return {
            "status": "not_requested",
            "error_count": 0,
            "warning_count": 0,
            "checked_wellpath_count": 0,
            "issues": [],
            "issue_codes": [],
            "output_json": None,
        }

    payload = contract_result.get("payload", {}) if isinstance(contract_result.get("payload"), dict) else {}
    counts = payload.get("counts", {}) if isinstance(payload.get("counts"), dict) else {}

    normalized_issues = _dedupe_issues_keep_order(
        _extract_payload_issues(payload, source_check="contract")
    )
    inferred_counts = _infer_issue_counts(normalized_issues)
    checked_wellpath_count = _extract_checked_wellpath_count(payload)

    return {
        "status": _normalize_check_status(contract_result.get("status", "unknown")),
        "error_count": _pick_first_int(
            counts.get("error_count"),
            payload.get("error_count"),
            inferred_counts["error_count"],
            default=0,
        ),
        "warning_count": _pick_first_int(
            counts.get("warning_count"),
            payload.get("warning_count"),
            inferred_counts["warning_count"],
            default=0,
        ),
        "checked_wellpath_count": checked_wellpath_count,
        "issues": normalized_issues,
        "issue_codes": _collect_issue_codes(normalized_issues),
        "output_json": contract_result.get("output_json"),
    }


def _build_overlay_qc_summary(overlay_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overlay_result:
        return {
            "status": "not_requested",
            "plot_png": None,
            "station_count": 0,
            "wellhead_count": 0,
            "wellpath_count": 0,
            "issues": [],
            "output_json": None,
        }

    payload = overlay_result.get("payload", {}) if isinstance(overlay_result.get("payload"), dict) else {}
    counts = payload.get("counts", {}) if isinstance(payload.get("counts"), dict) else {}
    plot = payload.get("plot", {}) if isinstance(payload.get("plot"), dict) else {}
    overlay_issues = _dedupe_issues_keep_order(
        _extract_payload_issues(payload, source_check="overlay")
    )

    return {
        "status": _normalize_check_status(overlay_result.get("status", "unknown")),
        "plot_png": plot.get("out_png"),
        "station_count": int(counts.get("station_count", 0) or 0),
        "wellhead_count": int(counts.get("wellhead_count", 0) or 0),
        "wellpath_count": int(counts.get("wellpath_count", 0) or 0),
        "issues": overlay_issues,
        "issue_codes": _collect_issue_codes(overlay_issues),
        "output_json": overlay_result.get("output_json"),
    }


def _aggregate_global_issues(
    results: List[Dict[str, Any]],
    contract_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for result in results:
        source_check = str(result.get("check_name", "unknown"))

        if source_check == "contract" and isinstance(contract_summary, dict):
            result_issues = list(contract_summary.get("issues", []) or [])
            if result.get("error"):
                result_issues.append(
                    {
                        "severity": "error",
                        "code": f"{source_check}_module_error",
                        "message": str(result["error"]),
                        "source_check": source_check,
                        "aggregated_from": "module_runner",
                    }
                )
        else:
            result_issues = _extract_result_issues(result)

        for issue in result_issues:
            severity = str(issue.get("severity", "")).lower()
            if severity == "warning":
                warnings.append(issue)
            else:
                errors.append(issue)

    return {
        "warnings": _dedupe_issues_keep_order(warnings),
        "errors": _dedupe_issues_keep_order(errors),
    }


def _build_checked_assets(
    contract_result: Optional[Dict[str, Any]],
    overlay_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    contract_payload = (
        contract_result.get("payload", {})
        if contract_result and isinstance(contract_result.get("payload"), dict)
        else {}
    )
    overlay_payload = (
        overlay_result.get("payload", {})
        if overlay_result and isinstance(overlay_result.get("payload"), dict)
        else {}
    )

    contract_paths = contract_payload.get("paths", {}) if isinstance(contract_payload.get("paths"), dict) else {}
    overlay_inputs = overlay_payload.get("inputs", {}) if isinstance(overlay_payload.get("inputs"), dict) else {}
    wellpaths_qc = contract_payload.get("wellpaths_qc", {}) if isinstance(contract_payload.get("wellpaths_qc"), dict) else {}

    wellpath_csvs: List[str] = []
    for p in list(overlay_inputs.get("wellpath_csvs", []) or []):
        if p:
            wellpath_csvs.append(str(p))
    for item in list(wellpaths_qc.get("items", []) or []):
        if isinstance(item, dict):
            path = item.get("path") or item.get("wellpath_csv")
            if path:
                wellpath_csvs.append(str(path))

    return {
        "stations_csv": contract_paths.get("stations_csv") or overlay_inputs.get("stations_csv"),
        "manifest_json": contract_paths.get("manifest_json") or overlay_inputs.get("manifest_json"),
        "baseline_build_summary_json": contract_paths.get("baseline_build_summary_json"),
        "wellpaths_index_json": contract_paths.get("wellpaths_index_json") or overlay_inputs.get("wellpaths_index_json"),
        "wellpath_build_summary_json": contract_paths.get("wellpath_build_summary_json"),
        "wellpath_csvs": _dedupe_keep_order(wellpath_csvs),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "QC baseline unified entry. Formal QC chain = contract + overlay. "
            "Auxiliary QC = qc_coor (optional, not in default core chain)."
        )
    )
    ap.add_argument("--module_name", default="baseline", help="Module name (default: baseline)")
    ap.add_argument("--out_root", default="prepared_project", help="prepared_project root directory")
    ap.add_argument(
        "--qc",
        choices=["core", "contract", "overlay", "coor", "all"],
        default="core",
        help=(
            "QC suite: core=contract+overlay (default), contract=contract QC only, "
            "overlay=overlay QC only, coor=auxiliary coordinate QC only, all=core+coor."
        ),
    )

    ap.add_argument("--stations_csv", default=None, help="Default: <out_root>/<module_name>/stations.csv")
    ap.add_argument("--manifest_json", default=None, help="Default: <out_root>/<module_name>/manifest.json")
    ap.add_argument(
        "--baseline_build_summary_json",
        default=None,
        help="Default: <out_root>/<module_name>/meta/baseline_build_summary.json",
    )
    ap.add_argument(
        "--wellpaths_index_json",
        default=None,
        help="Default: <out_root>/<module_name>/meta/wellpaths_index.json",
    )
    ap.add_argument(
        "--wellpath_build_summary_json",
        default=None,
        help="Default: <out_root>/<module_name>/meta/wellpath_build_summary.json",
    )
    ap.add_argument(
        "--wellpath_csv",
        action="append",
        default=None,
        help="Optional explicit wellpath csv(s), repeatable. If omitted, infer from formal assets.",
    )
    ap.add_argument("--wellheads_csv", default=None, help="Compatibility-only optional wellheads csv for overlay QC")
    ap.add_argument("--title", default=None, help="Optional overlay title override")

    ap.add_argument("--x", type=float, default=None, help="Raw X for qc_coor")
    ap.add_argument("--y", type=float, default=None, help="Raw Y for qc_coor")
    ap.add_argument("--lon0", type=float, default=105.0, help="Target GK lon0 for qc_coor (default 105)")
    ap.add_argument("--validate_china", action="store_true", help="Enable China-range validation for qc_coor")
    ap.add_argument("--out_txt", default=None, help="Optional explicit txt path for qc_coor")

    ap.add_argument("--out_json", default=None, help="Default: <out_root>/qc/<module_name>/reports/qc_summary.json")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    qc_root = os.path.join(args.out_root, "qc", args.module_name)
    figures_dir = os.path.join(qc_root, "figures")
    reports_dir = os.path.join(qc_root, "reports")
    logs_dir = os.path.join(qc_root, "logs")
    for p in (qc_root, figures_dir, reports_dir, logs_dir):
        _ensure_dir(p)

    out_json = args.out_json or os.path.join(reports_dir, "qc_summary.json")
    contract_json = os.path.join(reports_dir, "qc_contract_baseline.json")
    overlay_json = os.path.join(reports_dir, "qc_spatial_overlay.json")

    requested_checks: List[str]
    if args.qc == "core":
        requested_checks = ["contract", "overlay"]
    elif args.qc == "all":
        requested_checks = ["contract", "overlay", "coor"]
    else:
        requested_checks = [args.qc]

    results: List[Dict[str, Any]] = []

    if "contract" in requested_checks:
        from microseis_ds.baseline.qc import qc_contract_baseline as contract_mod

        contract_argv = [
            "--out_root", args.out_root,
            "--module_name", args.module_name,
            "--out_json", contract_json,
        ]
        if args.stations_csv:
            contract_argv += ["--stations_csv", args.stations_csv]
        if args.manifest_json:
            contract_argv += ["--manifest_json", args.manifest_json]
        if args.baseline_build_summary_json:
            contract_argv += ["--baseline_build_summary_json", args.baseline_build_summary_json]
        if args.wellpaths_index_json:
            contract_argv += ["--wellpaths_index_json", args.wellpaths_index_json]
        if args.wellpath_build_summary_json:
            contract_argv += ["--wellpath_build_summary_json", args.wellpath_build_summary_json]
        if args.wellpath_csv:
            for p in args.wellpath_csv:
                contract_argv += ["--wellpath_csv", p]

        results.append(
            _run_module_check(
                check_name="contract",
                module=contract_mod,
                argv=contract_argv,
                output_json=contract_json,
            )
        )

    if "overlay" in requested_checks:
        from microseis_ds.baseline.qc import qc_plot_spatial_overlay as overlay_mod

        overlay_argv = [
            "--out_root", args.out_root,
            "--module_name", args.module_name,
            "--out_json", overlay_json,
        ]
        if args.stations_csv:
            overlay_argv += ["--stations_csv", args.stations_csv]
        if args.manifest_json:
            overlay_argv += ["--manifest_json", args.manifest_json]
        if args.wellpaths_index_json:
            overlay_argv += ["--wellpaths_index_json", args.wellpaths_index_json]
        if args.wellpath_csv:
            for p in args.wellpath_csv:
                overlay_argv += ["--wellpath_csv", p]
        if args.wellheads_csv:
            overlay_argv += ["--wellheads_csv", args.wellheads_csv]
        if args.title:
            overlay_argv += ["--title", args.title]

        results.append(
            _run_module_check(
                check_name="overlay",
                module=overlay_mod,
                argv=overlay_argv,
                output_json=overlay_json,
            )
        )

    if "coor" in requested_checks:
        if args.x is None or args.y is None:
            results.append(
                {
                    "check_name": "coor",
                    "status": "skipped",
                    "exit_code": None,
                    "error": "qc_coor requires --x and --y",
                    "auxiliary": True,
                }
            )
        else:
            from microseis_ds.baseline.qc import qc_coor as coor_mod

            coor_argv = [
                "--out_root", args.out_root,
                "--module_name", args.module_name,
                "--x", str(float(args.x)),
                "--y", str(float(args.y)),
                "--lon0", str(float(args.lon0)),
            ]
            if args.validate_china:
                coor_argv += ["--validate_china"]
            if args.out_txt:
                coor_argv += ["--out_txt", args.out_txt]

            coor_result = _run_module_check(
                check_name="coor",
                module=coor_mod,
                argv=coor_argv,
                output_json=None,
            )
            coor_result["auxiliary"] = True
            results.append(coor_result)

    formal_results = [r for r in results if r.get("check_name") in FORMAL_QC_ITEMS]
    auxiliary_results = [r for r in results if r.get("check_name") in AUXILIARY_QC_ITEMS]

    formal_failed = [r for r in formal_results if r.get("status") not in ("passed", "pass")]
    auxiliary_failed = [r for r in auxiliary_results if r.get("status") not in ("passed", "pass", "skipped")]

    contract_result = _find_result(results, "contract")
    overlay_result = _find_result(results, "overlay")

    contract_summary = _build_contract_qc_summary(contract_result)
    overlay_summary = _build_overlay_qc_summary(overlay_result)
    global_issues = _aggregate_global_issues(results, contract_summary=contract_summary)

    overall_status = "failed" if formal_failed else "passed"
    normalized_warnings = _dedupe_issues_keep_order(global_issues["warnings"])
    normalized_errors = _dedupe_issues_keep_order(global_issues["errors"])

    summary: Dict[str, Any] = {
        "summary_version": "v1",
        "module_name": args.module_name,
        "qc_entry": "scripts/qc_baseline.py",
        "generated_at": _now_ts(),
        "status": overall_status,
        "overall_status": overall_status,
        "requested_qc": args.qc,
        "formal_qc_items": list(FORMAL_QC_ITEMS),
        "auxiliary_qc_items": list(AUXILIARY_QC_ITEMS),
        "contract_qc": contract_summary,
        "overlay_qc": overlay_summary,
        "checked_assets": _build_checked_assets(contract_result, overlay_result),
        "warnings": normalized_warnings,
        "errors": normalized_errors,
        "warning_codes": _collect_issue_codes(normalized_warnings),
        "error_codes": _collect_issue_codes(normalized_errors),
        "paths": {
            "qc_root": qc_root,
            "figures_dir": figures_dir,
            "reports_dir": reports_dir,
            "logs_dir": logs_dir,
            "summary_json": out_json,
            "contract_json": contract_json if "contract" in requested_checks else None,
            "overlay_json": overlay_json if "overlay" in requested_checks else None,
        },
        "results": results,
        "counts": {
            "formal_requested": len([c for c in requested_checks if c in FORMAL_QC_ITEMS]),
            "formal_failed": len(formal_failed),
            "auxiliary_requested": len([c for c in requested_checks if c in AUXILIARY_QC_ITEMS]),
            "auxiliary_failed": len(auxiliary_failed),
        },
        "notes": {
            "formal_qc_chain": "contract + overlay",
            "auxiliary_qc_chain": "qc_coor is retained as optional auxiliary check and is not part of Step 5 default core chain.",
            "summary_interface_note": (
                "overall_status / contract_qc / overlay_qc / checked_assets / warnings / errors "
                "are the normalized top-level summary fields; results remains the raw detail payload for compatibility."
            ),
            "module_error_policy": (
                "If a sub-check returns non-zero or raises, the top-level summary keeps a wrapper *_module_error "
                "in addition to normalized payload-derived issues."
            ),
        },
    }

    _json_dump(out_json, summary)

    print("[qc_baseline] DONE")
    print("  status:", summary["status"])
    print("  summary_json:", out_json)
    return 0 if not formal_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
