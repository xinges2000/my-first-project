# microseis_ds/velocity/qc/qc_contract_velocity.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from microseis_ds.common.schema import check_velocity_equiv_layered_contract
from microseis_ds.common.types import (
    TRAVELTIME_READY_REQUIRED_FILES,
    VELOCITY_EQUIV_MODEL_TYPE,
    VELOCITY_REQUIRED_PHASES,
)

QC_NAME = "contract_velocity"
REPORT_FILENAME = "qc_contract_velocity.json"
MODEL_JSON_NAME = "velocity_model_equiv_layered.json"
REQUIRED_ARRAY_FILES: Tuple[str, ...] = ("depth.npy", "vp.npy", "vs.npy")
REQUIRED_LAYER_KEYS: Tuple[str, ...] = ("layer_id", "z_top_m", "z_bot_m", "v_mps")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_posix(p: Path) -> str:
    return p.as_posix()


def _as_path(value: str | Path | None) -> Optional[Path]:
    if value is None:
        return None
    return Path(value)


def _default_report_path(velocity_root: Path, module_name: str) -> Path:
    # velocity_root is normally prepared_project/<module_name>.
    out_root = velocity_root.parent
    return out_root / "qc" / module_name / "reports" / REPORT_FILENAME


def _issue(code: str, message: str, *, path: str = "", source: str = "") -> Dict[str, str]:
    item: Dict[str, str] = {"code": code, "message": message}
    if path:
        item["path"] = path
    if source:
        item["source"] = source
    return item


def _schema_error_to_dict(err: Any) -> Dict[str, str]:
    one_line = getattr(err, "one_line", None)
    message = one_line() if callable(one_line) else str(err)
    path = str(getattr(err, "json_path", "") or "")
    return _issue("SCHEMA_CONTRACT_ERROR", message, path=path, source="schema")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return _to_posix(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact exception depends on filesystem/json parser
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(obj, dict):
        return None, f"JSON root must be an object, got {type(obj).__name__}"
    return obj, None


def _asset_record(role: str, path: Path, *, required: bool) -> Dict[str, Any]:
    exists = path.is_file()
    return {
        "role": role,
        "path": _to_posix(path),
        "required": bool(required),
        "exists": bool(exists),
        "status": "present" if exists else ("missing" if required else "not_found"),
    }


def _check_model_json(
    model_json_path: Path,
    checked_assets: List[Dict[str, Any]],
    errors: List[Dict[str, str]],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    checked_assets.append(_asset_record("equiv_layered_model_json", model_json_path, required=True))
    result: Dict[str, Any] = {
        "path": _to_posix(model_json_path),
        "status": "failed",
        "schema_valid": False,
        "model_type": None,
        "phase_counts": {},
        "error_count": 0,
    }

    if not model_json_path.is_file():
        errors.append(
            _issue(
                "MODEL_JSON_MISSING",
                f"Required model JSON not found: {model_json_path}",
                path=_to_posix(model_json_path),
                source="model_json",
            )
        )
        result["error_count"] = 1
        return None, result

    obj, read_error = _read_json(model_json_path)
    if read_error is not None or obj is None:
        errors.append(
            _issue(
                "MODEL_JSON_UNREADABLE",
                f"Failed to read model JSON: {read_error}",
                path=_to_posix(model_json_path),
                source="model_json",
            )
        )
        result["error_count"] = 1
        return None, result

    result["model_type"] = obj.get("model_type")
    layers = obj.get("layers")
    if isinstance(layers, Mapping):
        for phase in VELOCITY_REQUIRED_PHASES:
            phase_layers = layers.get(phase)
            result["phase_counts"][phase] = len(phase_layers) if isinstance(phase_layers, list) else 0

    schema_report = check_velocity_equiv_layered_contract(obj)
    schema_errors = list(schema_report.get("errors", []))
    schema_warnings = list(schema_report.get("warnings", []))
    if schema_errors:
        errors.extend(_schema_error_to_dict(err) for err in schema_errors)

    result.update(
        {
            "status": "passed" if not schema_errors else "failed",
            "schema_valid": not schema_errors,
            "required_model_type": VELOCITY_EQUIV_MODEL_TYPE,
            "required_phases": list(VELOCITY_REQUIRED_PHASES),
            "required_layer_keys": list(REQUIRED_LAYER_KEYS),
            "schema_error_count": len(schema_errors),
            "schema_warning_count": len(schema_warnings),
            "error_count": len(schema_errors),
        }
    )
    return obj, result


def _check_array_file(path: Path, role: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    issues: List[Dict[str, str]] = []
    result: Dict[str, Any] = {
        "path": _to_posix(path),
        "role": role,
        "status": "failed",
        "exists": path.is_file(),
        "shape": None,
        "dtype": None,
        "size": None,
        "finite": False,
    }

    if not path.is_file():
        issues.append(
            _issue(
                "ARRAY_FILE_MISSING",
                f"Required array file not found: {path.name}",
                path=_to_posix(path),
                source="arrays",
            )
        )
        return result, issues

    try:
        arr = np.load(path)
    except Exception as exc:  # pragma: no cover
        issues.append(
            _issue(
                "ARRAY_FILE_UNREADABLE",
                f"Failed to load {path.name}: {type(exc).__name__}: {exc}",
                path=_to_posix(path),
                source="arrays",
            )
        )
        return result, issues

    result["shape"] = list(arr.shape)
    result["dtype"] = str(arr.dtype)
    result["size"] = int(arr.size)

    if arr.ndim != 1:
        issues.append(
            _issue(
                "ARRAY_NOT_1D",
                f"{path.name} must be a 1D array, got shape {arr.shape}",
                path=_to_posix(path),
                source="arrays",
            )
        )
    if arr.size == 0:
        issues.append(
            _issue(
                "ARRAY_EMPTY",
                f"{path.name} must not be empty",
                path=_to_posix(path),
                source="arrays",
            )
        )

    finite = bool(np.all(np.isfinite(arr))) if arr.size else False
    result["finite"] = finite
    if not finite:
        issues.append(
            _issue(
                "ARRAY_NONFINITE",
                f"{path.name} contains non-finite values",
                path=_to_posix(path),
                source="arrays",
            )
        )

    if role in ("vp", "vs") and arr.size and bool(np.any(arr <= 0)):
        issues.append(
            _issue(
                "VELOCITY_ARRAY_NONPOSITIVE",
                f"{path.name} must contain positive velocities only",
                path=_to_posix(path),
                source="arrays",
            )
        )

    if role == "depth" and arr.size > 1 and not bool(np.all(np.diff(arr.astype(float)) > 0)):
        issues.append(
            _issue(
                "DEPTH_ARRAY_NOT_STRICTLY_INCREASING",
                "depth.npy must be strictly increasing",
                path=_to_posix(path),
                source="arrays",
            )
        )

    result["status"] = "passed" if not issues else "failed"
    return result, issues


def _check_arrays(
    equiv_layered_dir: Path,
    checked_assets: List[Dict[str, Any]],
    errors: List[Dict[str, str]],
) -> Dict[str, Any]:
    role_by_name = {"depth.npy": "depth", "vp.npy": "vp", "vs.npy": "vs"}
    array_results: Dict[str, Dict[str, Any]] = {}
    lengths: Dict[str, Optional[int]] = {}

    for filename in REQUIRED_ARRAY_FILES:
        path = equiv_layered_dir / filename
        checked_assets.append(_asset_record(filename, path, required=True))
        role = role_by_name[filename]
        one_result, one_issues = _check_array_file(path, role)
        array_results[filename] = one_result
        lengths[filename] = one_result.get("size") if one_result.get("status") != "failed" else one_result.get("size")
        errors.extend(one_issues)

    present_lengths = [v for v in lengths.values() if isinstance(v, int)]
    shape_match = len(set(present_lengths)) == 1 if len(present_lengths) == len(REQUIRED_ARRAY_FILES) else False
    if len(present_lengths) == len(REQUIRED_ARRAY_FILES) and not shape_match:
        errors.append(
            _issue(
                "ARRAY_LENGTH_MISMATCH",
                f"depth/vp/vs lengths must match, got {lengths}",
                path=_to_posix(equiv_layered_dir),
                source="arrays",
            )
        )

    status = "passed"
    if any(item["status"] == "failed" for item in array_results.values()) or not shape_match:
        status = "failed"

    return {
        "status": status,
        "required_files": list(REQUIRED_ARRAY_FILES),
        "files": array_results,
        "lengths": lengths,
        "shape_match": bool(shape_match),
    }


def _check_optional_json_metadata(
    path: Path,
    role: str,
    checked_assets: List[Dict[str, Any]],
    warnings: List[Dict[str, str]],
) -> Dict[str, Any]:
    checked_assets.append(_asset_record(role, path, required=False))
    result: Dict[str, Any] = {
        "path": _to_posix(path),
        "status": "not_found",
        "exists": path.is_file(),
        "readable": False,
        "summary": {},
    }

    if not path.is_file():
        warnings.append(
            _issue(
                f"{role.upper()}_MISSING",
                f"Optional metadata file not found: {path}",
                path=_to_posix(path),
                source=role,
            )
        )
        return result

    obj, read_error = _read_json(path)
    if read_error is not None or obj is None:
        warnings.append(
            _issue(
                f"{role.upper()}_UNREADABLE",
                f"Optional metadata file exists but is not readable JSON: {read_error}",
                path=_to_posix(path),
                source=role,
            )
        )
        result["status"] = "warning"
        return result

    result["readable"] = True
    result["status"] = "passed"

    summary: Dict[str, Any] = {}
    for key in (
        "status",
        "overall_status",
        "asset_count",
        "asset_hashes_count",
        "warnings_count",
        "errors_count",
    ):
        if key in obj:
            summary[key] = obj.get(key)

    required_asset_status = obj.get("required_asset_status")
    if isinstance(required_asset_status, Mapping):
        summary["required_asset_status"] = dict(required_asset_status)

    asset_hashes = obj.get("asset_hashes")
    if isinstance(asset_hashes, Mapping):
        summary["asset_hashes_count"] = len(asset_hashes)
    elif isinstance(asset_hashes, list):
        summary["asset_hashes_count"] = len(asset_hashes)

    if not summary:
        warnings.append(
            _issue(
                f"{role.upper()}_STATUS_NOT_EXPRESSED",
                f"Optional metadata file is readable but does not expose a recognizable status/count field: {path.name}",
                path=_to_posix(path),
                source=role,
            )
        )
        result["status"] = "warning"

    result["summary"] = summary
    return result


def run_contract_qc(
    *,
    velocity_root: str | Path = "prepared_project/velocity",
    module_name: str = "velocity",
    equiv_layered_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    build_summary_path: str | Path | None = None,
    report_path: str | Path | None = None,
    write_report: bool = True,
) -> Dict[str, Any]:
    """
    Run velocity single-module contract QC and optionally write a JSON report.

    This function checks only velocity-owned build artifacts. It does not check
    cross-module compatibility with traveltime, locator, or any downstream
    consumer.
    """
    vroot = Path(velocity_root)
    equiv_dir = _as_path(equiv_layered_dir) or (vroot / "equiv_layered")
    model_json_path = equiv_dir / MODEL_JSON_NAME
    manifest = _as_path(manifest_path) or (vroot / "velocity_manifest.json")
    build_summary = _as_path(build_summary_path) or (vroot / "meta" / "velocity_build_summary.json")
    rpt_path = _as_path(report_path) or _default_report_path(vroot, module_name)

    warnings: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    checked_assets: List[Dict[str, Any]] = []

    model_obj, model_result = _check_model_json(model_json_path, checked_assets, errors)
    arrays_result = _check_arrays(equiv_dir, checked_assets, errors)
    manifest_result = _check_optional_json_metadata(manifest, "manifest", checked_assets, warnings)
    build_summary_result = _check_optional_json_metadata(build_summary, "build_summary", checked_assets, warnings)

    hard_error_count = len(errors)
    status = "failed" if hard_error_count else ("warning" if warnings else "passed")

    report: Dict[str, Any] = {
        "schema_version": "velocity_qc_contract_v1",
        "module": module_name,
        "qc_name": QC_NAME,
        "status": status,
        "checked_at": _now_utc_iso(),
        "inputs": {
            "velocity_root": _to_posix(vroot),
            "equiv_layered_dir": _to_posix(equiv_dir),
            "model_json": _to_posix(model_json_path),
            "manifest_path": _to_posix(manifest),
            "build_summary_path": _to_posix(build_summary),
        },
        "checked_assets": checked_assets,
        "warnings": warnings,
        "errors": errors,
        "results": {
            "contract_status": "failed" if hard_error_count else "passed",
            "model_json_status": model_result["status"],
            "layers_status": model_result["status"],
            "arrays_status": arrays_result["status"],
            "manifest_status": manifest_result["status"],
            "build_summary_status": build_summary_result["status"],
            "required_model_type": VELOCITY_EQUIV_MODEL_TYPE,
            "required_phases": list(VELOCITY_REQUIRED_PHASES),
            "required_layer_keys": list(REQUIRED_LAYER_KEYS),
            "required_array_files": list(REQUIRED_ARRAY_FILES),
            "traveltime_ready_required_files": list(TRAVELTIME_READY_REQUIRED_FILES),
        },
        "contract_checks": {
            "model_json": model_result,
            "arrays": arrays_result,
        },
        "manifest_check": manifest_result,
        "build_summary_check": build_summary_result,
        "report_path": _to_posix(rpt_path),
    }

    if write_report:
        write_contract_qc_report(report, rpt_path)

    return report


def write_contract_qc_report(report: Mapping[str, Any], report_path: str | Path) -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qc_contract_velocity",
        description="Velocity single-module contract QC for equiv_layered build artifacts.",
    )
    p.add_argument(
        "--velocity_root",
        default="prepared_project/velocity",
        help="Velocity module build output root. Default: prepared_project/velocity",
    )
    p.add_argument("--module_name", default="velocity", help="Module name. Default: velocity")
    p.add_argument(
        "--equiv_layered_dir",
        default=None,
        help="Override equiv_layered directory. Default: <velocity_root>/equiv_layered",
    )
    p.add_argument(
        "--manifest_path",
        default=None,
        help="Override velocity_manifest.json path. Default: <velocity_root>/velocity_manifest.json",
    )
    p.add_argument(
        "--build_summary_path",
        default=None,
        help="Override velocity_build_summary.json path. Default: <velocity_root>/meta/velocity_build_summary.json",
    )
    p.add_argument(
        "--report_path",
        default=None,
        help="Override output report path. Default: <out_root>/qc/<module_name>/reports/qc_contract_velocity.json",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    report = run_contract_qc(
        velocity_root=args.velocity_root,
        module_name=args.module_name,
        equiv_layered_dir=args.equiv_layered_dir,
        manifest_path=args.manifest_path,
        build_summary_path=args.build_summary_path,
        report_path=args.report_path,
        write_report=True,
    )

    print("[OK] qc_contract_velocity finished")
    print(f"  status     : {report['status']}")
    print(f"  report_path: {Path(str(report['report_path'])).resolve()}")
    print(f"  errors     : {len(report.get('errors', []))}")
    print(f"  warnings   : {len(report.get('warnings', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
