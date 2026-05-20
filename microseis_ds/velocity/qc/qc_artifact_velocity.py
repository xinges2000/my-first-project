# microseis_ds/velocity/qc/qc_artifact_velocity.py
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from microseis_ds.velocity.runtime.velocity_model import VelocityModel

QC_NAME = "artifact_velocity"
REPORT_FILENAME = "qc_artifact_velocity.json"
QUERY_CSV_FILENAME = "qc_velocity_artifact_query.csv"
DEFAULT_QUERY_DEPTHS = (0.0, 100.0, 500.0, 1000.0)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_posix(path: Path) -> str:
    return path.as_posix()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return _to_posix(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _issue(code: str, message: str, *, path: str = "", source: str = "") -> Dict[str, str]:
    item: Dict[str, str] = {"code": code, "message": message}
    if path:
        item["path"] = path
    if source:
        item["source"] = source
    return item


def _asset_record(role: str, path: Path, *, required: bool) -> Dict[str, Any]:
    exists = path.exists()
    return {
        "role": role,
        "path": _to_posix(path),
        "required": bool(required),
        "exists": bool(exists),
        "status": "present" if exists else ("missing" if required else "not_found"),
    }


def _default_reports_dir(velocity_root: Path, module_name: str) -> Path:
    out_root = velocity_root.parent
    return out_root / "qc" / module_name / "reports"


def _default_report_path(velocity_root: Path, module_name: str) -> Path:
    return _default_reports_dir(velocity_root, module_name) / REPORT_FILENAME


def _default_query_csv_path(velocity_root: Path, module_name: str) -> Path:
    return _default_reports_dir(velocity_root, module_name) / QUERY_CSV_FILENAME


def _model_summary_safe(vm: Any) -> Dict[str, Any]:
    try:
        summary = vm.summary()
    except Exception as exc:  # pragma: no cover - defensive only
        return {"summary_error": f"{type(exc).__name__}: {exc}"}
    if isinstance(summary, Mapping):
        return dict(summary)
    return {"summary": summary}


def _write_query_csv(
    *,
    path: Path,
    requested_depths: np.ndarray,
    query_depths: np.ndarray,
    vp: np.ndarray,
    vs: np.ndarray,
    include_requested_depth: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    if include_requested_depth:
        lines.append("requested_depth_m,query_depth_m,vp_mps,vs_mps,clipped\n")
        for requested, query, vpi, vsi in zip(
            requested_depths.tolist(),
            query_depths.tolist(),
            vp.tolist(),
            vs.tolist(),
        ):
            clipped = bool(float(requested) != float(query))
            lines.append(f"{float(requested):.6f},{float(query):.6f},{float(vpi):.6f},{float(vsi):.6f},{str(clipped).lower()}\n")
    else:
        lines.append("depth_m,vp_mps,vs_mps\n")
        for query, vpi, vsi in zip(query_depths.tolist(), vp.tolist(), vs.tolist()):
            lines.append(f"{float(query):.6f},{float(vpi):.6f},{float(vsi):.6f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def run_artifact_qc(
    *,
    velocity_root: str | Path = "prepared_project/velocity",
    module_name: str = "velocity",
    ref_dir: str | Path | None = None,
    query_depths: List[float] | tuple[float, ...] = DEFAULT_QUERY_DEPTHS,
    no_extrapolate: bool = False,
    report_path: str | Path | None = None,
    query_csv_path: str | Path | None = None,
    write_report: bool = True,
    write_csv: bool = True,
) -> Dict[str, Any]:
    """Run velocity single-module artifact QC and optionally write JSON/CSV outputs.

    This module checks only velocity-owned ref_engineering artifacts. It does not
    check cross-module compatibility with traveltime, locator, or any downstream
    consumer.
    """
    vroot = Path(velocity_root)
    ref_path = Path(ref_dir) if ref_dir is not None else (vroot / "ref_engineering")
    rpt_path = Path(report_path) if report_path is not None else _default_report_path(vroot, module_name)
    csv_path = Path(query_csv_path) if query_csv_path is not None else _default_query_csv_path(vroot, module_name)

    warnings: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    checked_assets: List[Dict[str, Any]] = [_asset_record("ref_engineering", ref_path, required=True)]

    requested_z = np.asarray([float(z) for z in query_depths], dtype=float)
    query_z = requested_z.copy()
    vp = np.asarray([], dtype=float)
    vs = np.asarray([], dtype=float)
    model_summary: Dict[str, Any] = {}
    load_status = "failed"
    query_status = "not_run"
    csv_status = "not_written"

    if requested_z.size == 0:
        errors.append(_issue("QUERY_DEPTHS_EMPTY", "At least one query depth is required.", source="artifact"))

    try:
        vm = VelocityModel.load(str(ref_path))
        load_status = "passed"
        model_summary = _model_summary_safe(vm)

        depth_min = float(np.nanmin(vm.depth_m))
        depth_max = float(np.nanmax(vm.depth_m))
        if requested_z.size and (float(np.nanmin(requested_z)) < depth_min or float(np.nanmax(requested_z)) > depth_max):
            warnings.append(
                _issue(
                    "QUERY_DEPTH_OUT_OF_MODEL_RANGE",
                    "One or more requested query depths are outside the artifact depth range.",
                    path=_to_posix(ref_path),
                    source="artifact",
                )
            )
        if no_extrapolate and requested_z.size:
            query_z = np.clip(requested_z, depth_min, depth_max)

        if requested_z.size:
            vp_raw, vs_raw = vm.query(query_z.tolist())
            vp = np.asarray(vp_raw, dtype=float)
            vs = np.asarray(vs_raw, dtype=float)
            query_status = "passed"

            if vp.shape != query_z.shape or vs.shape != query_z.shape:
                errors.append(
                    _issue(
                        "QUERY_RESULT_SHAPE_MISMATCH",
                        f"Query result shapes must match query depths; got vp={vp.shape}, vs={vs.shape}, z={query_z.shape}.",
                        path=_to_posix(ref_path),
                        source="artifact",
                    )
                )
                query_status = "failed"
            if vp.size and not bool(np.all(np.isfinite(vp))):
                errors.append(_issue("QUERY_VP_NONFINITE", "Queried Vp contains non-finite values.", source="artifact"))
                query_status = "failed"
            if vs.size and not bool(np.all(np.isfinite(vs))):
                errors.append(_issue("QUERY_VS_NONFINITE", "Queried Vs contains non-finite values.", source="artifact"))
                query_status = "failed"
            if vp.size and bool(np.any(vp <= 0)):
                errors.append(_issue("QUERY_VP_NONPOSITIVE", "Queried Vp must be positive.", source="artifact"))
                query_status = "failed"
            if vs.size and bool(np.any(vs <= 0)):
                errors.append(_issue("QUERY_VS_NONPOSITIVE", "Queried Vs must be positive.", source="artifact"))
                query_status = "failed"

            if write_csv:
                try:
                    _write_query_csv(
                        path=csv_path,
                        requested_depths=requested_z,
                        query_depths=query_z,
                        vp=vp,
                        vs=vs,
                        include_requested_depth=bool(no_extrapolate),
                    )
                    csv_status = "written"
                except Exception as exc:  # pragma: no cover - filesystem dependent
                    csv_status = "warning"
                    warnings.append(
                        _issue(
                            "QUERY_CSV_WRITE_FAILED",
                            f"Failed to write artifact query CSV: {type(exc).__name__}: {exc}",
                            path=_to_posix(csv_path),
                            source="artifact",
                        )
                    )
    except Exception as exc:
        errors.append(
            _issue(
                "ARTIFACT_LOAD_OR_QUERY_FAILED",
                f"Artifact load/query failed: {type(exc).__name__}: {exc}",
                path=_to_posix(ref_path),
                source="artifact",
            )
        )
        load_status = "failed"
        if query_status == "not_run":
            query_status = "failed"

    hard_error_count = len(errors)
    status = "failed" if hard_error_count else ("warning" if warnings else "passed")

    result_stats: Dict[str, Any] = {
        "load_status": load_status,
        "query_status": query_status,
        "csv_status": csv_status,
        "query_count": int(query_z.size),
        "requested_depths_m": requested_z.tolist(),
        "query_depths_m": query_z.tolist(),
        "no_extrapolate": bool(no_extrapolate),
        "model_summary": model_summary,
    }
    if query_z.size:
        result_stats["query_depth_min_m"] = float(np.nanmin(query_z))
        result_stats["query_depth_max_m"] = float(np.nanmax(query_z))
    if vp.size:
        result_stats["vp_min_mps"] = float(np.nanmin(vp))
        result_stats["vp_max_mps"] = float(np.nanmax(vp))
    if vs.size:
        result_stats["vs_min_mps"] = float(np.nanmin(vs))
        result_stats["vs_max_mps"] = float(np.nanmax(vs))

    report: Dict[str, Any] = {
        "schema_version": "velocity_qc_artifact_v1",
        "module": module_name,
        "qc_name": QC_NAME,
        "status": status,
        "checked_at": _now_utc_iso(),
        "inputs": {
            "velocity_root": _to_posix(vroot),
            "ref_dir": _to_posix(ref_path),
            "query_depths_m": requested_z.tolist(),
            "no_extrapolate": bool(no_extrapolate),
        },
        "checked_assets": checked_assets,
        "warnings": warnings,
        "errors": errors,
        "results": result_stats,
        "report_path": _to_posix(rpt_path),
        "query_csv_path": _to_posix(csv_path),
    }

    if write_report:
        write_artifact_qc_report(report, rpt_path)

    return report


def write_artifact_qc_report(report: Mapping[str, Any], report_path: str | Path) -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qc_artifact_velocity",
        description="Velocity single-module artifact QC wrapper for ref_engineering build artifacts.",
    )
    p.add_argument(
        "--velocity_root",
        default="prepared_project/velocity",
        help="Velocity module build output root. Default: prepared_project/velocity",
    )
    p.add_argument("--module_name", default="velocity", help="Module name. Default: velocity")
    p.add_argument(
        "--ref_dir",
        default=None,
        help="Override ref_engineering directory. Default: <velocity_root>/ref_engineering",
    )
    p.add_argument(
        "--query_z",
        nargs="+",
        type=float,
        default=list(DEFAULT_QUERY_DEPTHS),
        help="Depths (m) used in artifact query. Default: 0 100 500 1000",
    )
    p.add_argument(
        "--no_extrapolate",
        action="store_true",
        help="If set, clip query depths into model range before querying.",
    )
    p.add_argument(
        "--report_path",
        default=None,
        help="Override output JSON report path. Default: <out_root>/qc/<module_name>/reports/qc_artifact_velocity.json",
    )
    p.add_argument(
        "--query_csv_path",
        default=None,
        help="Override artifact query CSV path. Default: <out_root>/qc/<module_name>/reports/qc_velocity_artifact_query.csv",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    report = run_artifact_qc(
        velocity_root=args.velocity_root,
        module_name=args.module_name,
        ref_dir=args.ref_dir,
        query_depths=args.query_z,
        no_extrapolate=args.no_extrapolate,
        report_path=args.report_path,
        query_csv_path=args.query_csv_path,
        write_report=True,
        write_csv=True,
    )

    print("[OK] qc_artifact_velocity finished")
    print(f"  status        : {report['status']}")
    print(f"  report_path   : {Path(str(report['report_path'])).resolve()}")
    print(f"  query_csv_path: {Path(str(report['query_csv_path'])).resolve()}")
    print(f"  errors        : {len(report.get('errors', []))}")
    print(f"  warnings      : {len(report.get('warnings', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
