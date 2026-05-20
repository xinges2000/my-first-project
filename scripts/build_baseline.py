#!/usr/bin/env python
# scripts/build_baseline.py
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from microseis_ds.common.schema import check_csv_contract

CORE_BUILD_STAGE = "baseline_core_project_assets"
ENTRY_BUILD_STAGE = "baseline_entry_build"
LOCKED_LON0 = 105.0
HISTORY_DIRNAME = "history"
ENTRY_FINALIZER = "scripts/build_baseline.py"
REQUIRED_FINAL_ENTRY_ASSETS = [
    "meta/wellpaths_index.json",
    "meta/baseline_build_summary.json",
    "manifest.json",
]
CURRENT_VIEW_BUILD_SCOPE = "baseline_entry_registry_current_view"
CURRENT_VIEW_RECORD_SET_SCOPE = "current_entry_build_only"


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_dump(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _json_load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _asset_relpath(module_out_dir: str, path: str) -> str:
    try:
        return os.path.relpath(os.path.abspath(path), os.path.abspath(module_out_dir)).replace("\\", "/")
    except Exception:
        return os.path.basename(path)


def _assert_lon0_locked(lon0: float) -> None:
    try:
        value = float(lon0)
    except Exception as exc:
        raise SystemExit(f"[build_baseline] ERROR: --lon0 must be numeric, got {lon0!r}.") from exc

    if abs(value - LOCKED_LON0) > 1e-9:
        raise SystemExit(
            f"[build_baseline] ERROR: baseline build entry is locked to lon0={LOCKED_LON0:.1f} (GK105), "
            f"but received --lon0={value:.6f}. Please correct the CLI input instead of relying on implicit override."
        )


def _build_sequence_payload(wellpath_count: int) -> List[Dict[str, Any]]:
    return [
        {
            "order": 1,
            "stage": "per_well_wellpath_build_or_verified_reuse",
            "builder": "microseis_ds.baseline.build.build_wellpath_gk105",
            "count": int(wellpath_count),
        },
        {
            "order": 2,
            "stage": "core_project_assets_build",
            "builder": "microseis_ds.baseline.build.build_project_from_metadata_gk105",
            "outputs": [
                "project.json",
                "manifest.json",
                "stations.csv",
                "links/traceid_station.csv",
                "meta/stations_contract_summary.json",
                "meta/traceid_station_summary.json",
                "meta/baseline_build_summary.json",
            ],
        },
        {
            "order": 3,
            "stage": "write_wellpaths_index",
            "output": "meta/wellpaths_index.json",
            "note": "Registry wellhead coordinates must come from per-well canonical builder outputs, not raw wellheads_csv input.",
        },
        {
            "order": 4,
            "stage": "patch_per_well_metadata",
            "effect": "set entry-finalized registry state and add wellpaths_index_json",
        },
        {
            "order": 5,
            "stage": "patch_wellpath_build_summary",
            "output": "meta/wellpath_build_summary.json",
        },
        {
            "order": 6,
            "stage": "update_baseline_build_summary",
            "output": "meta/baseline_build_summary.json",
        },
        {
            "order": 7,
            "stage": "finalize_manifest",
            "output": "manifest.json",
        },
    ]


def _group_rows_by_key(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped

def _raise_duplicate_plan_error(kind: str, grouped_rows: Dict[str, List[Dict[str, Any]]]) -> None:
    examples = []
    for dup_key, rows in grouped_rows.items():
        if len(rows) < 2:
            continue
        examples.append(
            {
                "kind": kind,
                "value": dup_key,
                "build_orders": [int(r["build_order"]) for r in rows],
                "survey_csvs": [r["survey_csv_arg"] for r in rows],
                "well_ids": [r["well_id"] for r in rows],
                "out_csvs": [r["out_csv"] for r in rows],
            }
        )

    if examples:
        raise ValueError(
            f"[build_baseline] duplicate {kind} detected in baseline build plan.\n"
            f"{json.dumps(examples[:10], ensure_ascii=False, indent=2)}"
        )

def _build_wellpath_plan(
    module_out_dir: str,
    survey_paths: List[str],
    survey_ids: List[str],
    survey_id_sources: List[str],
) -> List[Dict[str, Any]]:
    plan_rows: List[Dict[str, Any]] = []

    for idx, (survey_csv, well_id, well_id_source) in enumerate(
        zip(survey_paths, survey_ids, survey_id_sources),
        start=1,
    ):
        wid = str(well_id).strip()
        survey_csv_arg = str(survey_csv)
        survey_csv_abs = os.path.abspath(survey_csv_arg)
        out_csv = os.path.join(module_out_dir, f"wellpath_{wid}_gk.csv")
        out_csv_abs = os.path.abspath(out_csv)

        plan_rows.append(
            {
                "build_order": int(idx),
                "survey_csv_arg": survey_csv_arg,
                "survey_csv_abs": survey_csv_abs,
                "survey_csv_basename": os.path.basename(survey_csv_arg),
                "well_id": wid,
                "well_id_source": str(well_id_source),
                "out_csv": out_csv,
                "out_csv_abs": out_csv_abs,
            }
        )

    survey_groups = _group_rows_by_key(plan_rows, "survey_csv_abs")
    survey_dups = {k: v for k, v in survey_groups.items() if len(v) > 1}
    if survey_dups:
        _raise_duplicate_plan_error("survey_csv", survey_dups)

    well_id_groups = _group_rows_by_key(plan_rows, "well_id")
    well_id_dups = {k: v for k, v in well_id_groups.items() if len(v) > 1}
    if well_id_dups:
        _raise_duplicate_plan_error("well_id", well_id_dups)

    out_csv_groups = _group_rows_by_key(plan_rows, "out_csv_abs")
    out_csv_dups = {k: v for k, v in out_csv_groups.items() if len(v) > 1}
    if out_csv_dups:
        _raise_duplicate_plan_error("out_csv", out_csv_dups)

    return plan_rows


def _infer_well_id_from_survey_path(p: str) -> str:
    base = os.path.basename(p)
    name, _ = os.path.splitext(base)
    name = name.replace("well_", "").replace("survey_", "")
    for suf in ["_survey", "-survey", "_mdincaz", "_MDIncAz"]:
        if name.endswith(suf):
            name = name[: -len(suf)]
    return name.strip()


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


def _ensure_xy_m(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]

    if "X_m" in d.columns and "Y_m" in d.columns:
        d["X_m"] = pd.to_numeric(d["X_m"], errors="coerce")
        d["Y_m"] = pd.to_numeric(d["Y_m"], errors="coerce")
        return d

    if "E_gk_m" in d.columns and "N_gk_m" in d.columns:
        d["E_gk_m"] = pd.to_numeric(d["E_gk_m"], errors="coerce")
        d["N_gk_m"] = pd.to_numeric(d["N_gk_m"], errors="coerce")
        d["X_m"] = d["N_gk_m"]
        d["Y_m"] = d["E_gk_m"]
        return d

    if "X" in d.columns and "Y" in d.columns:
        d["X_m"] = pd.to_numeric(d["X"], errors="coerce")
        d["Y_m"] = pd.to_numeric(d["Y"], errors="coerce")
        return d

    raise ValueError(
        f"[build_baseline] Cannot infer {kind} coordinates. Need (X_m,Y_m) or (E_gk_m,N_gk_m) or legacy (X,Y). "
        f"Columns={d.columns.tolist()}"
    )


def _read_wellheads_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[build_baseline] wellheads_csv not found: {path}")

    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]

    if "well_id" not in d.columns:
        raise ValueError(f"[build_baseline] wellheads_csv missing required column 'well_id': {path}")

    d["well_id"] = d["well_id"].astype(str).str.strip()
    if (d["well_id"] == "").any():
        bad = d.loc[d["well_id"] == "", ["well_id"]].head(10).to_dict(orient="records")
        raise ValueError(
            f"[build_baseline] wellheads_csv contains empty well_id, which is not allowed. Examples={bad}"
        )

    dup_mask = d["well_id"].duplicated(keep=False)
    if dup_mask.any():
        bad = d.loc[dup_mask, ["well_id"]].head(10).to_dict(orient="records")
        raise ValueError(
            "[build_baseline] wellheads_csv contains duplicate well_id values. "
            "This would cause silent wrong mapping if .iloc[0] were used. "
            f"Please fix the input file. Examples={bad}"
        )

    d = _ensure_xy_m(d, "wellheads")

    if "Z_m" not in d.columns:
        d["Z_m"] = 0.0
    d["Z_m"] = pd.to_numeric(d["Z_m"], errors="coerce").fillna(0.0)

    if d["X_m"].isna().any() or d["Y_m"].isna().any():
        bad = d[d["X_m"].isna() | d["Y_m"].isna()][["well_id", "X_m", "Y_m"]].head(10)
        raise ValueError(f"[build_baseline] wellheads_csv has NaN in X_m/Y_m. Examples:\n{bad}")

    return d


def _build_wellheads_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    lookup: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        wid = str(row["well_id"])
        lookup[wid] = {
            "X_m": float(row["X_m"]),
            "Y_m": float(row["Y_m"]),
            "Z_m": float(row["Z_m"]) if pd.notna(row["Z_m"]) else 0.0,
        }
    return lookup


def _build_meta_dir(module_out_dir: str) -> str:
    meta_dir = os.path.join(module_out_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    return meta_dir


def _metadata_path_for_wellpath(out_csv: str) -> str:
    stem = Path(out_csv).stem
    return os.path.join(os.path.dirname(os.path.abspath(out_csv)), "meta", f"{stem}_metadata.json")


def _wellpath_summary_path(module_out_dir: str) -> str:
    return os.path.join(module_out_dir, "meta", "wellpath_build_summary.json")


def _registry_state_payload(index_included: bool) -> Dict[str, Any]:
    if index_included:
        return {
            "registry_scope": "baseline_entry_finalized",
            "per_well_asset_complete": True,
            "full_baseline_registry_complete": True,
            "index_included": True,
            "baseline_entry_required": False,
            "baseline_entry_finalizer": ENTRY_FINALIZER,
            "required_entry_assets": [],
            "status_note": "Per-well asset has been registered into baseline entry-level registry.",
        }
    return {
        "registry_scope": "partial_single_well_state",
        "per_well_asset_complete": True,
        "full_baseline_registry_complete": False,
        "index_included": False,
        "baseline_entry_required": True,
        "baseline_entry_finalizer": ENTRY_FINALIZER,
        "required_entry_assets": list(REQUIRED_FINAL_ENTRY_ASSETS),
        "status_note": (
            "Single-well builder completed per-well asset generation only. "
            "Baseline global registry is NOT complete until scripts/build_baseline.py writes/patches entry-level assets."
        ),
    }


def _float_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def _compare_xyz(expected: Dict[str, float], actual: Dict[str, float], label: str, tol: float = 1e-6) -> Optional[str]:
    for axis in ("X_m", "Y_m", "Z_m"):
        if axis not in actual:
            return f"{label} missing axis {axis}"
        if not _float_close(float(expected[axis]), float(actual[axis]), tol=tol):
            return (
                f"{label} mismatch at {axis}: expected={float(expected[axis]):.6f}, "
                f"actual={float(actual[axis]):.6f}"
            )
    return None


def _extract_raw_input_wellhead_xyz_from_metadata(metadata: Dict[str, Any]) -> Dict[str, float]:
    raw = metadata.get("raw_input_wellhead_xyz_m")
    if isinstance(raw, dict):
        return {
            "X_m": float(raw["X_m"]),
            "Y_m": float(raw["Y_m"]),
            "Z_m": float(raw["Z_m"]),
        }

    wellhead = metadata.get("wellhead", {})
    if isinstance(wellhead, dict):
        if all(k in wellhead for k in ("wellhead_x", "wellhead_y", "wellhead_z")):
            return {
                "X_m": float(wellhead["wellhead_x"]),
                "Y_m": float(wellhead["wellhead_y"]),
                "Z_m": float(wellhead["wellhead_z"]),
            }
    raise ValueError("per-well metadata missing raw input wellhead fields")


def _extract_canonical_wellhead_xyz_from_metadata(metadata: Dict[str, Any]) -> Dict[str, float]:
    canonical = metadata.get("canonical_wellhead_xyz_m")
    if isinstance(canonical, dict):
        return {
            "X_m": float(canonical["X_m"]),
            "Y_m": float(canonical["Y_m"]),
            "Z_m": float(canonical["Z_m"]),
        }

    wellhead = metadata.get("wellhead", {})
    canonical_gk = wellhead.get("canonical_gk", {}) if isinstance(wellhead, dict) else {}
    if isinstance(canonical_gk, dict) and all(k in canonical_gk for k in ("X_m_northing", "Y_m_easting", "Z_m")):
        return {
            "X_m": float(canonical_gk["X_m_northing"]),
            "Y_m": float(canonical_gk["Y_m_easting"]),
            "Z_m": float(canonical_gk["Z_m"]),
        }
    raise ValueError("per-well metadata missing canonical wellhead fields")


def _extract_raw_input_wellhead_xyz_from_summary_record(record: Dict[str, Any]) -> Optional[Dict[str, float]]:
    raw = record.get("raw_input_wellhead_xyz_m")
    if isinstance(raw, dict) and all(k in raw for k in ("X_m", "Y_m", "Z_m")):
        return {
            "X_m": float(raw["X_m"]),
            "Y_m": float(raw["Y_m"]),
            "Z_m": float(raw["Z_m"]),
        }

    wellhead = record.get("wellhead", {})
    if isinstance(wellhead, dict) and all(k in wellhead for k in ("wellhead_x", "wellhead_y", "wellhead_z")):
        return {
            "X_m": float(wellhead["wellhead_x"]),
            "Y_m": float(wellhead["wellhead_y"]),
            "Z_m": float(wellhead["wellhead_z"]),
        }
    return None


def _extract_canonical_wellhead_xyz_from_summary_record(record: Dict[str, Any]) -> Optional[Dict[str, float]]:
    canonical = record.get("canonical_wellhead_xyz_m")
    if isinstance(canonical, dict) and all(k in canonical for k in ("X_m", "Y_m", "Z_m")):
        return {
            "X_m": float(canonical["X_m"]),
            "Y_m": float(canonical["Y_m"]),
            "Z_m": float(canonical["Z_m"]),
        }
    return None


def _find_summary_record_for_wellpath(module_out_dir: str, summary_data: Dict[str, Any], out_csv_abs: str) -> Optional[Dict[str, Any]]:
    target_rel = _asset_relpath(module_out_dir, out_csv_abs)
    for rec in list(summary_data.get("records", [])):
        outputs = rec.get("outputs", {}) if isinstance(rec.get("outputs", {}), dict) else {}
        rel = str(outputs.get("wellpath_csv") or rec.get("wellpath_csv") or "")
        if rel == target_rel:
            return rec
    return None


def _raise_existing_wellpath_csv_error(out_csv: str, detail: str) -> None:
    raise ValueError(f"existing wellpath CSV validation failed: file={out_csv}. {detail}")


def _require_wellpath_integrity_payload(well_id: str, payload: Any, source_name: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"well_id={well_id!r} missing usable wellpath_integrity in {source_name}"
        )

    required_top_keys = ("row_count", "first_row", "last_row", "md_range_m", "tvd_range_m")
    missing_top = [k for k in required_top_keys if k not in payload]
    if missing_top:
        raise ValueError(
            f"well_id={well_id!r} wellpath_integrity in {source_name} missing keys {missing_top}"
        )

    for point_name in ("first_row", "last_row"):
        point = payload.get(point_name)
        if not isinstance(point, dict):
            raise ValueError(
                f"well_id={well_id!r} wellpath_integrity.{point_name} in {source_name} must be an object"
            )
        missing_axes = [axis for axis in ("MD", "TVD_m", "X_m", "Y_m", "Z_m") if axis not in point]
        if missing_axes:
            raise ValueError(
                f"well_id={well_id!r} wellpath_integrity.{point_name} in {source_name} missing keys {missing_axes}"
            )

    for range_name in ("md_range_m", "tvd_range_m"):
        range_obj = payload.get(range_name)
        if not isinstance(range_obj, dict):
            raise ValueError(
                f"well_id={well_id!r} wellpath_integrity.{range_name} in {source_name} must be an object"
            )
        missing_range_keys = [k for k in ("min", "max") if k not in range_obj]
        if missing_range_keys:
            raise ValueError(
                f"well_id={well_id!r} wellpath_integrity.{range_name} in {source_name} missing keys {missing_range_keys}"
            )

    normalized = copy.deepcopy(payload)
    normalized["row_count"] = int(normalized["row_count"])
    for point_name in ("first_row", "last_row"):
        normalized[point_name] = {
            axis: float(normalized[point_name][axis])
            for axis in ("MD", "TVD_m", "X_m", "Y_m", "Z_m")
        }
    for range_name in ("md_range_m", "tvd_range_m"):
        normalized[range_name] = {
            "min": float(normalized[range_name]["min"]),
            "max": float(normalized[range_name]["max"]),
        }
    return normalized


def _validate_existing_wellpath_csv(
    plan: Dict[str, Any],
    canonical_wellhead_xyz_m: Dict[str, float],
    expected_wellpath_integrity: Dict[str, Any],
) -> Dict[str, Any]:
    out_csv = plan["out_csv"]
    well_id = str(plan["well_id"])

    if not os.path.exists(out_csv):
        _raise_existing_wellpath_csv_error(out_csv, "file does not exist.")

    try:
        df = pd.read_csv(out_csv)
    except pd.errors.EmptyDataError as exc:
        _raise_existing_wellpath_csv_error(out_csv, "CSV is empty and cannot be reused.")
    except Exception as exc:
        _raise_existing_wellpath_csv_error(out_csv, f"unable to read CSV: {exc}")

    if df.empty:
        _raise_existing_wellpath_csv_error(out_csv, "CSV has zero data rows and cannot be reused.")

    contract = check_csv_contract(df, "wellpath")
    missing_required = list(contract.get("missing_required", []))
    null_in_required = list(contract.get("null_in_required", []))
    extra_columns = list(contract.get("extra_columns", []))
    if missing_required or null_in_required or extra_columns:
        details = []
        if missing_required:
            details.append(f"missing_required={missing_required}")
        if null_in_required:
            details.append(f"null_in_required={null_in_required}")
        if extra_columns:
            details.append(f"extra_columns={extra_columns}")
        _raise_existing_wellpath_csv_error(
            out_csv,
            "contract check did not pass for kind='wellpath': " + "; ".join(details),
        )

    numeric_cols = ["MD", "TVD_m", "X_m", "Y_m", "Z_m"]
    num = df.copy()
    for col in numeric_cols:
        if col not in num.columns:
            _raise_existing_wellpath_csv_error(out_csv, f"required column missing after contract check: {col}")
        num[col] = pd.to_numeric(num[col], errors="coerce")

    if num[numeric_cols].isna().any().any():
        bad_cols = [col for col in numeric_cols if num[col].isna().any()]
        _raise_existing_wellpath_csv_error(
            out_csv,
            f"key content check failed: non-numeric or NaN values found in {bad_cols}",
        )

    row_count = int(len(num))
    if row_count <= 0:
        _raise_existing_wellpath_csv_error(out_csv, "key content check failed: row_count must be > 0")

    md_diff = num["MD"].diff().iloc[1:]
    if (md_diff <= 0).any():
        bad_idx = md_diff.index[md_diff <= 0].tolist()[:10]
        _raise_existing_wellpath_csv_error(
            out_csv,
            f"key content check failed: MD must be strictly increasing, bad_row_indices={bad_idx}",
        )

    tvd_min = float(num["TVD_m"].min())
    tvd_max = float(num["TVD_m"].max())
    if tvd_max < tvd_min:
        _raise_existing_wellpath_csv_error(
            out_csv,
            f"key content check failed: invalid TVD_m range, min={tvd_min:.6f}, max={tvd_max:.6f}",
        )

    first_row = num.iloc[0]
    csv_first_xyz = {
        "X_m": float(first_row["X_m"]),
        "Y_m": float(first_row["Y_m"]),
        "Z_m": float(first_row["Z_m"]),
    }
    canonical_diff = _compare_xyz(
        canonical_wellhead_xyz_m,
        csv_first_xyz,
        "existing CSV first-row canonical wellhead",
        tol=1e-4,
    )
    if canonical_diff:
        _raise_existing_wellpath_csv_error(out_csv, f"canonical wellhead consistency check failed: {canonical_diff}")

    expected = _require_wellpath_integrity_payload(
        well_id=well_id,
        payload=expected_wellpath_integrity,
        source_name="per-well metadata / wellpath_build_summary",
    )

    actual = {
        "row_count": row_count,
        "first_row": {axis: float(num.iloc[0][axis]) for axis in ("MD", "TVD_m", "X_m", "Y_m", "Z_m")},
        "last_row": {axis: float(num.iloc[-1][axis]) for axis in ("MD", "TVD_m", "X_m", "Y_m", "Z_m")},
        "md_range_m": {"min": float(num["MD"].min()), "max": float(num["MD"].max())},
        "tvd_range_m": {"min": float(num["TVD_m"].min()), "max": float(num["TVD_m"].max())},
    }

    if actual["row_count"] != expected["row_count"]:
        _raise_existing_wellpath_csv_error(
            out_csv,
            f"integrity check failed: row_count mismatch, expected={expected['row_count']}, actual={actual['row_count']}",
        )

    for point_name in ("first_row", "last_row"):
        for axis in ("MD", "TVD_m", "X_m", "Y_m", "Z_m"):
            if not _float_close(actual[point_name][axis], expected[point_name][axis], tol=1e-4):
                _raise_existing_wellpath_csv_error(
                    out_csv,
                    (
                        f"integrity check failed: {point_name}.{axis} mismatch, "
                        f"expected={expected[point_name][axis]:.6f}, actual={actual[point_name][axis]:.6f}"
                    ),
                )

    for range_name in ("md_range_m", "tvd_range_m"):
        for bound in ("min", "max"):
            if not _float_close(actual[range_name][bound], expected[range_name][bound], tol=1e-4):
                _raise_existing_wellpath_csv_error(
                    out_csv,
                    (
                        f"integrity check failed: {range_name}.{bound} mismatch, "
                        f"expected={expected[range_name][bound]:.6f}, actual={actual[range_name][bound]:.6f}"
                    ),
                )

    return actual


def _load_verified_per_well_assets(
    module_out_dir: str,
    plan: Dict[str, Any],
    input_wellhead_xyz_m: Dict[str, float],
) -> Dict[str, Any]:
    metadata_path = _metadata_path_for_wellpath(plan["out_csv"])
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"per-well metadata missing: {metadata_path}. "
            "Step 4 final closure does not allow registry/index/summary to proceed without per-well metadata."
        )

    metadata = _json_load(metadata_path)
    if str(metadata.get("build_status", "")) != "success":
        raise ValueError(
            f"per-well metadata build_status is not success for well_id={plan['well_id']!r}: {metadata_path}"
        )

    meta_well_id = str(metadata.get("well_id", "")).strip()
    if meta_well_id != str(plan["well_id"]):
        raise ValueError(
            f"per-well metadata well_id mismatch for out_csv={plan['out_csv']}: "
            f"expected={plan['well_id']!r}, actual={meta_well_id!r}"
        )

    meta_survey = str(metadata.get("survey_csv", "")).strip()
    if not meta_survey:
        raise ValueError(
            f"per-well metadata missing survey identity for well_id={plan['well_id']!r}: {metadata_path}"
        )
    if meta_survey != str(plan["survey_csv_basename"]):
        raise ValueError(
            f"per-well metadata survey identity mismatch for well_id={plan['well_id']!r}: "
            f"expected survey basename={plan['survey_csv_basename']!r}, actual={meta_survey!r}"
        )

    metadata_raw_input = _extract_raw_input_wellhead_xyz_from_metadata(metadata)
    raw_diff = _compare_xyz(input_wellhead_xyz_m, metadata_raw_input, "metadata raw_input_wellhead_xyz_m")
    if raw_diff:
        raise ValueError(
            f"per-well metadata input-wellhead mismatch for well_id={plan['well_id']!r}: {raw_diff}"
        )

    canonical_wellhead_xyz_m = _extract_canonical_wellhead_xyz_from_metadata(metadata)

    summary_path = _wellpath_summary_path(module_out_dir)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"wellpath_build_summary.json missing: {summary_path}. "
            "existing_reused branch must not silently reuse old wellpath without summary validation."
        )

    summary_data = _json_load(summary_path)
    summary_record = _find_summary_record_for_wellpath(module_out_dir, summary_data, plan["out_csv_abs"])
    if summary_record is None:
        raise ValueError(
            f"wellpath_build_summary.json does not contain a record for out_csv={plan['out_csv']}"
        )

    summary_well_id = str(summary_record.get("well_id", "")).strip()
    if summary_well_id != str(plan["well_id"]):
        raise ValueError(
            f"wellpath_build_summary.json well_id mismatch for out_csv={plan['out_csv']}: "
            f"expected={plan['well_id']!r}, actual={summary_well_id!r}"
        )

    summary_survey = str(
        (summary_record.get("input_files", {}) if isinstance(summary_record.get("input_files", {}), dict) else {}).get("survey_csv")
        or summary_record.get("survey_csv")
        or (summary_record.get("survey_summary", {}) if isinstance(summary_record.get("survey_summary", {}), dict) else {}).get("survey_csv_basename")
        or ""
    ).strip()
    if not summary_survey:
        raise ValueError(
            f"wellpath_build_summary.json missing survey identity for well_id={plan['well_id']!r}: {summary_path}"
        )
    if summary_survey != str(plan["survey_csv_basename"]):
        raise ValueError(
            f"wellpath_build_summary.json survey identity mismatch for well_id={plan['well_id']!r}: "
            f"expected survey basename={plan['survey_csv_basename']!r}, actual={summary_survey!r}"
        )

    summary_raw_input = _extract_raw_input_wellhead_xyz_from_summary_record(summary_record)
    summary_canonical = _extract_canonical_wellhead_xyz_from_summary_record(summary_record)
    if summary_raw_input is None and summary_canonical is None:
        raise ValueError(
            f"wellpath_build_summary.json missing wellhead identity fields for well_id={plan['well_id']!r}: {summary_path}"
        )
    if summary_raw_input is not None:
        raw_summary_diff = _compare_xyz(
            input_wellhead_xyz_m,
            summary_raw_input,
            "summary raw_input_wellhead_xyz_m",
        )
        if raw_summary_diff:
            raise ValueError(
                f"wellpath_build_summary.json input-wellhead mismatch for well_id={plan['well_id']!r}: {raw_summary_diff}"
            )

    if summary_canonical is not None:
        canonical_diff = _compare_xyz(
            canonical_wellhead_xyz_m,
            summary_canonical,
            "summary canonical_wellhead_xyz_m",
        )
        if canonical_diff:
            raise ValueError(
                f"wellpath_build_summary.json canonical wellhead mismatch for well_id={plan['well_id']!r}: {canonical_diff}"
            )

    expected_wellpath_integrity = copy.deepcopy(
        metadata.get("wellpath_integrity") or summary_record.get("wellpath_integrity") or {}
    )
    validated_wellpath_integrity = _validate_existing_wellpath_csv(
        plan=plan,
        canonical_wellhead_xyz_m=canonical_wellhead_xyz_m,
        expected_wellpath_integrity=expected_wellpath_integrity,
    )

    return {
        "metadata": metadata,
        "metadata_path_abs": metadata_path,
        "metadata_rel": _asset_relpath(module_out_dir, metadata_path),
        "summary_path_abs": summary_path,
        "summary_rel": _asset_relpath(module_out_dir, summary_path),
        "summary_record": summary_record,
        "canonical_wellhead_xyz_m": canonical_wellhead_xyz_m,
        "raw_input_wellhead_xyz_m": metadata_raw_input,
        "well_id_resolution": copy.deepcopy(metadata.get("well_id_resolution") or summary_record.get("well_id_resolution") or {}),
        "wellpath_integrity": validated_wellpath_integrity,
    }


def _build_record_from_verified_assets(
    module_out_dir: str,
    plan: Dict[str, Any],
    verified_assets: Dict[str, Any],
    build_status: str,
) -> Dict[str, Any]:
    return {
        "build_order": int(plan["build_order"]),
        "survey_csv_arg": plan["survey_csv_arg"],
        "survey_csv_abs": plan["survey_csv_abs"],
        "survey_csv_basename": plan["survey_csv_basename"],
        "well_id": plan["well_id"],
        "well_id_source": plan["well_id_source"],
        "well_id_resolution": copy.deepcopy(verified_assets.get("well_id_resolution") or {}),
        "raw_input_wellhead_xyz_m": dict(verified_assets["raw_input_wellhead_xyz_m"]),
        "canonical_wellhead_xyz_m": dict(verified_assets["canonical_wellhead_xyz_m"]),
        "wellpath_integrity": copy.deepcopy(verified_assets.get("wellpath_integrity") or {}),
        "wellpath_csv_abs": plan["out_csv_abs"],
        "wellpath_metadata_json_abs": verified_assets["metadata_path_abs"],
        "wellpath_metadata_json_rel": verified_assets["metadata_rel"],
        "wellpath_build_summary_json_abs": verified_assets["summary_path_abs"],
        "wellpath_build_summary_json_rel": verified_assets["summary_rel"],
        "build_status": build_status,
    }


def _build_reuse_repair_reason(exc: Exception) -> str:
    return f"existing wellpath reuse denied: {exc}"


def _survey_well_id_policy_payload(args, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    mode_used = "explicit_arg" if args.survey_well_id else "filename_inference"
    is_fallback = mode_used != "explicit_arg"
    record_count = len(records)
    return {
        "preferred_mode": "explicit_arg",
        "fallback_mode": "filename_inference",
        "mode_used": mode_used,
        "mode_used_is_fallback": bool(is_fallback),
        "fallback_record_count": int(record_count if is_fallback else 0),
        "policy_note": (
            "Prefer explicit --survey_well_id; filename inference is retained only as compatibility fallback "
            "and must not be treated as equivalent to the preferred governance path."
        ),
        "warning": (
            "Current entry build used filename inference fallback for survey-to-well association. "
            "This is compatibility-only and should be replaced by explicit --survey_well_id in formal runs."
            if is_fallback
            else "Current entry build used explicit --survey_well_id mapping for survey-to-well association."
        ),
    }


def _write_wellpaths_index(module_out_dir: str, records: List[Dict[str, Any]]) -> str:
    meta_dir = _build_meta_dir(module_out_dir)
    out_path = os.path.join(meta_dir, "wellpaths_index.json")

    items: List[Dict[str, Any]] = []
    for rec in records:
        item = {
            "build_order": int(rec["build_order"]),
            "survey_csv_arg": rec["survey_csv_arg"],
            "survey_csv_basename": rec["survey_csv_basename"],
            "survey_csv_relpath_from_module": _asset_relpath(module_out_dir, rec["survey_csv_abs"]),
            "well_id": rec["well_id"],
            "well_id_source": rec["well_id_source"],
            "well_id_resolution": copy.deepcopy(rec.get("well_id_resolution") or {}),
            "wellhead_xyz_m": dict(rec["canonical_wellhead_xyz_m"]),
            "wellpath_csv": _asset_relpath(module_out_dir, rec["wellpath_csv_abs"]),
            "wellpath_metadata_json": rec["wellpath_metadata_json_rel"],
            "wellpath_build_summary_json": rec["wellpath_build_summary_json_rel"],
            "wellpath_integrity": copy.deepcopy(rec.get("wellpath_integrity") or {}),
            "build_status": rec["build_status"],
            "full_baseline_registry_complete": True,
        }
        items.append(item)

    payload = {
        "build_stage": ENTRY_BUILD_STAGE,
        "generated_at": _now_ts(),
        "row_count": int(len(items)),
        "mapping_policy": "Prefer explicit --survey_well_id; filename inference is retained only as compatibility fallback.",
        "wellhead_registry_policy": (
            "Entry-level registry writes canonical wellhead coordinates only. "
            "Raw wellheads_csv input is retained in per-well metadata and wellpath_build_summary for audit traceability, not in entry-level registry items."
        ),
        "full_baseline_registry_complete": True,
        "items": items,
    }
    _json_dump(out_path, payload)
    return out_path


def _patch_per_well_metadata_index_flag(module_out_dir: str, records: List[Dict[str, Any]]) -> None:
    finalized_at = _now_ts()
    registry_state = _registry_state_payload(index_included=True)

    for rec in records:
        abs_path = rec["wellpath_metadata_json_abs"]
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"per-well metadata missing during final entry patch: {abs_path}. "
                "Step 4 final closure must stop here."
            )
        data = _json_load(abs_path)
        data["index_included"] = True
        data["wellpaths_index_json"] = "meta/wellpaths_index.json"
        data["build_order"] = int(rec["build_order"])
        data["survey_csv_relpath_from_module"] = _asset_relpath(module_out_dir, rec["survey_csv_abs"])
        data["full_baseline_registry_complete"] = True
        data["registry_state"] = dict(registry_state)
        data["baseline_registry_note"] = registry_state["status_note"]
        data["baseline_entry_build_stage"] = ENTRY_BUILD_STAGE
        data["baseline_entry_finalized_at"] = finalized_at
        data.setdefault("outputs", {})
        if isinstance(data["outputs"], dict):
            data["outputs"]["wellpaths_index_json"] = "meta/wellpaths_index.json"
        _json_dump(abs_path, data)


def _summary_record_relpath(module_out_dir: str, rec: Dict[str, Any]) -> str:
    outputs = rec.get("outputs", {}) if isinstance(rec.get("outputs", {}), dict) else {}
    rel = str(outputs.get("wellpath_csv") or rec.get("wellpath_csv") or "").strip()
    if rel:
        return rel
    abs_path = rec.get("wellpath_csv_abs")
    if abs_path:
        return _asset_relpath(module_out_dir, str(abs_path))
    return ""


def _normalize_current_view_well_ids(records: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for rec in records:
        wid = str(rec.get("well_id", "")).strip()
        if wid and wid not in seen:
            seen.add(wid)
            ordered.append(wid)
    return ordered


def _derive_entry_registry_current_view_available(
    build_scope: str,
    current_records: List[Dict[str, Any]],
    current_view_record_count: int,
    current_view_well_ids: List[str],
) -> bool:
    if str(build_scope).strip() != CURRENT_VIEW_BUILD_SCOPE:
        return False
    if int(current_view_record_count) != len(current_records):
        return False
    record_well_ids = _normalize_current_view_well_ids(current_records)
    if record_well_ids != list(current_view_well_ids):
        return False
    return len(current_records) > 0


def _build_history_record_from_prior_summary_record(module_out_dir: str, rec: Dict[str, Any], finalized_at: str) -> Dict[str, Any]:
    history_rec = copy.deepcopy(rec)
    history_rec["entry_registry_scope"] = "history_records"
    history_rec["excluded_reason"] = "not_in_current_entry_build"
    history_rec["excluded_at"] = finalized_at
    history_rec["full_baseline_registry_complete"] = True
    rel = _summary_record_relpath(module_out_dir, history_rec)
    if rel:
        history_rec["history_record_key"] = rel
    return history_rec


def _merge_history_records(module_out_dir: str, prior_history: List[Dict[str, Any]], derived_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    ordered: List[str] = []

    for rec in list(prior_history) + list(derived_history):
        key = _summary_record_relpath(module_out_dir, rec) or str(rec.get("well_id") or json.dumps(rec, ensure_ascii=False, sort_keys=True))
        if key not in merged:
            ordered.append(key)
        merged[key] = rec

    return [merged[k] for k in ordered]


def _patch_wellpath_build_summary_index_flag(module_out_dir: str, records: List[Dict[str, Any]]) -> None:
    summary_path = _wellpath_summary_path(module_out_dir)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"wellpath_build_summary.json missing during final entry patch: {summary_path}"
        )

    data = _json_load(summary_path)
    by_csv = {_asset_relpath(module_out_dir, rec["wellpath_csv_abs"]): rec for rec in records}
    patched_relpaths = set()
    current_records: List[Dict[str, Any]] = []
    derived_history_records: List[Dict[str, Any]] = []
    finalized_at = _now_ts()
    registry_state = _registry_state_payload(index_included=True)

    for rec in list(data.get("records", [])):
        outputs = rec.get("outputs", {}) if isinstance(rec.get("outputs", {}), dict) else {}
        rel = str(outputs.get("wellpath_csv") or rec.get("wellpath_csv") or "").strip()
        matched = by_csv.get(rel)
        if matched is not None:
            patched_relpaths.add(rel)
            patched = copy.deepcopy(rec)
            patched["index_included"] = True
            patched["build_order"] = int(matched["build_order"])
            patched["survey_csv_relpath_from_module"] = _asset_relpath(module_out_dir, matched["survey_csv_abs"])
            patched["raw_input_wellhead_xyz_m"] = dict(matched["raw_input_wellhead_xyz_m"])
            patched["canonical_wellhead_xyz_m"] = dict(matched["canonical_wellhead_xyz_m"])
            patched["well_id_resolution"] = copy.deepcopy(matched.get("well_id_resolution") or patched.get("well_id_resolution") or {})
            patched["wellpath_integrity"] = copy.deepcopy(matched.get("wellpath_integrity") or patched.get("wellpath_integrity") or {})
            patched["full_baseline_registry_complete"] = True
            patched["registry_state"] = dict(registry_state)
            patched["baseline_registry_note"] = registry_state["status_note"]
            patched["entry_registry_scope"] = "current_view_records"
            patched["outputs"] = dict(outputs)
            patched["outputs"]["wellpaths_index_json"] = "meta/wellpaths_index.json"
            current_records.append(patched)
        else:
            derived_history_records.append(_build_history_record_from_prior_summary_record(module_out_dir, rec, finalized_at))

    missing = [rel for rel in by_csv.keys() if rel not in patched_relpaths]
    if missing:
        raise ValueError(
            "wellpath_build_summary.json is missing records for one or more wellpaths required by the current baseline build: "
            f"{missing}"
        )

    prior_history_records = list(data.get("history_records", []))
    history_records = _merge_history_records(module_out_dir, prior_history_records, derived_history_records)
    current_view_well_ids = _normalize_current_view_well_ids(current_records)
    current_view_record_count = int(len(current_records))
    entry_registry_current_view_available = _derive_entry_registry_current_view_available(
        build_scope=CURRENT_VIEW_BUILD_SCOPE,
        current_records=current_records,
        current_view_record_count=current_view_record_count,
        current_view_well_ids=current_view_well_ids,
    )

    success_count = sum(1 for r in current_records if r.get("build_status") == "success")
    failed_count = sum(1 for r in current_records if r.get("build_status") == "failed")

    data["build_stage"] = ENTRY_BUILD_STAGE
    data["build_scope"] = CURRENT_VIEW_BUILD_SCOPE
    data["generated_at"] = _now_ts()
    data["record_count"] = int(len(current_records))
    data["success_count"] = int(success_count)
    data["failed_count"] = int(failed_count)
    data["record_set_semantics"] = (
        "records stores the current entry-build current view only; history_records stores prior registry records excluded from the current entry build."
    )
    data["current_view_record_count"] = current_view_record_count
    data["current_view_well_ids"] = current_view_well_ids
    data["entry_registry_current_view_available"] = bool(entry_registry_current_view_available)
    data["history_records"] = history_records
    data["history_record_count"] = int(len(history_records))
    data["full_baseline_registry_complete"] = True
    data["registry_state"] = dict(registry_state)
    data["baseline_registry_note"] = registry_state["status_note"]
    data["baseline_entry_finalized_at"] = finalized_at
    data["records"] = current_records
    _json_dump(summary_path, data)


def _update_baseline_build_summary(module_out_dir: str, args, records: List[Dict[str, Any]]) -> None:
    path = os.path.join(module_out_dir, "meta", "baseline_build_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"baseline_build_summary.json missing: {path}. Entry-level final closure cannot continue."
        )

    data = _json_load(path)
    survey_well_id_policy = _survey_well_id_policy_payload(args, records)

    data["build_stage"] = ENTRY_BUILD_STAGE
    data["generated_at"] = _now_ts()
    data["full_baseline_registry_complete"] = True

    data["inputs"] = {
        "gps_txt": os.path.basename(args.gps_txt),
        "wellheads_csv": os.path.basename(args.wellheads_csv),
        "survey_count": int(len(records)),
        "survey_csvs": [rec["survey_csv_arg"] for rec in records],
        "survey_well_id_mode": survey_well_id_policy["mode_used"],
        "survey_well_id_policy": survey_well_id_policy,
        "mag_decl_deg": float(args.mag_decl_deg),
        "lon0": float(LOCKED_LON0),
    }

    outputs = dict(data.get("outputs", {}))
    outputs.update(
        {
            "project_json": "project.json",
            "manifest_json": "manifest.json",
            "stations_csv": "stations.csv",
            "traceid_station_csv": "links/traceid_station.csv",
            "stations_contract_summary_json": "meta/stations_contract_summary.json",
            "traceid_station_summary_json": "meta/traceid_station_summary.json",
            "history_dir": HISTORY_DIRNAME,
            "baseline_build_summary_json": "meta/baseline_build_summary.json",
            "wellpath_build_summary_json": "meta/wellpath_build_summary.json",
            "wellpaths_index_json": "meta/wellpaths_index.json",
            "wellpath_csvs": [_asset_relpath(module_out_dir, rec["wellpath_csv_abs"]) for rec in records],
            "wellpath_metadata_jsons": [rec["wellpath_metadata_json_rel"] for rec in records],
        }
    )
    data["outputs"] = outputs

    data["build_sequence"] = _build_sequence_payload(len(records))

    success_count = sum(1 for rec in records if rec.get("build_status") == "success")
    reused_count = sum(1 for rec in records if rec.get("build_status") == "existing_reused")

    items = []
    for rec in records:
        items.append(
            {
                "build_order": int(rec["build_order"]),
                "well_id": rec["well_id"],
                "well_id_source": rec["well_id_source"],
                "well_id_resolution": copy.deepcopy(rec.get("well_id_resolution") or {}),
                "survey_csv_arg": rec["survey_csv_arg"],
                "survey_csv_basename": rec["survey_csv_basename"],
                "wellhead_xyz_m": dict(rec["canonical_wellhead_xyz_m"]),
                "wellpath_csv": _asset_relpath(module_out_dir, rec["wellpath_csv_abs"]),
                "wellpath_metadata_json": rec["wellpath_metadata_json_rel"],
                "wellpath_integrity": copy.deepcopy(rec.get("wellpath_integrity") or {}),
                "build_status": rec["build_status"],
            }
        )

    data["wellpaths"] = {
        "row_count": int(len(records)),
        "success_count": int(success_count),
        "existing_reused_count": int(reused_count),
        "survey_well_mapping_policy": survey_well_id_policy["policy_note"],
        "survey_well_id_policy": survey_well_id_policy,
        "wellhead_registry_policy": (
            "Entry-level registry writes canonical wellhead coordinates only. "
            "Raw wellheads_csv input is retained in per-well metadata and wellpath_build_summary for audit traceability, not in entry-level registry items."
        ),
        "items": items,
    }

    data["registry_state"] = {
        "full_baseline_registry_complete": True,
        "baseline_entry_finalized_at": _now_ts(),
        "baseline_entry_finalizer": ENTRY_FINALIZER,
        "record_set_scope": CURRENT_VIEW_RECORD_SET_SCOPE,
    }

    _json_dump(path, data)


def _finalize_manifest(module_out_dir: str, args, records: List[Dict[str, Any]]) -> None:
    path = os.path.join(module_out_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest.json missing: {path}. Entry-level final closure cannot continue.")

    data = _json_load(path)
    survey_well_id_policy = _survey_well_id_policy_payload(args, records)
    wellpath_summary_path = _wellpath_summary_path(module_out_dir)
    wellpath_summary = _json_load(wellpath_summary_path) if os.path.exists(wellpath_summary_path) else {}

    assets = dict(data.get("assets", {}))
    assets.update(
        {
            "project_json": "project.json",
            "stations_csv": "stations.csv",
            "traceid_station_link": "links/traceid_station.csv",
            "stations_contract_summary_json": "meta/stations_contract_summary.json",
            "traceid_station_summary_json": "meta/traceid_station_summary.json",
            "baseline_build_summary_json": "meta/baseline_build_summary.json",
            "wellpath_build_summary_json": "meta/wellpath_build_summary.json",
            "wellpaths_index_json": "meta/wellpaths_index.json",
            "wellpath_csvs": [_asset_relpath(module_out_dir, rec["wellpath_csv_abs"]) for rec in records],
            "wellpath_metadata_jsons": [rec["wellpath_metadata_json_rel"] for rec in records],
        }
    )
    data["assets"] = assets

    state = dict(data.get("state", {}))
    state.update(
        {
            "baseline_build_stage": ENTRY_BUILD_STAGE,
            "baseline_entry_finalized_at": _now_ts(),
            "baseline_source": "metadata_and_survey",
            "history_dir": HISTORY_DIRNAME,
            "wellpath_count": int(len(records)),
            "full_baseline_registry_complete": True,
            "survey_well_mapping_policy": survey_well_id_policy["policy_note"],
            "survey_well_id_policy": survey_well_id_policy,
            "wellhead_registry_policy": "Registry wellhead coordinates are taken from per-well canonical builder outputs. Raw wellheads_csv input is audit-only.",
            "build_sequence": _build_sequence_payload(len(records)),
            "wellpath_summary_current_view_record_count": int(wellpath_summary.get("current_view_record_count", len(records))),
            "wellpath_summary_record_set_scope": str(wellpath_summary.get("build_scope", CURRENT_VIEW_RECORD_SET_SCOPE)),
        }
    )
    data["state"] = state

    _json_dump(path, data)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build baseline module assets from GPS metadata + per-well survey CSVs.")
    ap.add_argument("--gps_txt", required=True, help="GPS txt file: inline_3D lon lat z E N")
    ap.add_argument("--wellheads_csv", required=True, help="Wellheads csv (REQUIRED)")
    ap.add_argument(
        "--survey_csv",
        action="append",
        required=True,
        help="Survey CSV (MD/Inc/Az). REQUIRED. Repeatable for multiple wells.",
    )
    ap.add_argument(
        "--survey_well_id",
        action="append",
        default=None,
        help="Recommended: explicit well_id for each --survey_csv (repeatable, same count). If omitted, infer from filename as compatibility fallback.",
    )
    ap.add_argument("--out_root", default="prepared_project", help="Output root directory")
    ap.add_argument("--module_name", default="baseline", help="Output module directory name under --out_root")
    ap.add_argument("--project_name", default="microseis_project_gk105", help="Project name recorded in project.json")
    ap.add_argument(
        "--lon0",
        type=float,
        default=105.0,
        help="GK central meridian. Current baseline build entry is locked to 105.0 and will reject other values.",
    )
    ap.add_argument("--mag_decl_deg", type=float, default=0.0, help="Magnetic declination in degrees (default 0.0)")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite baseline build files if already exist (build_project_from_metadata_gk105 handles archiving).",
    )
    ap.add_argument(
        "--overwrite_wellpath",
        action="store_true",
        help="Overwrite generated wellpath CSV if exists.",
    )
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    _assert_lon0_locked(args.lon0)

    module_out_dir = os.path.join(args.out_root, args.module_name)
    _ensure_dir(module_out_dir)
    _build_meta_dir(module_out_dir)

    wellheads = _read_wellheads_csv(args.wellheads_csv)
    wellheads_lookup = _build_wellheads_lookup(wellheads)

    from microseis_ds.baseline.build import build_wellpath_gk105 as wp_mod

    survey_paths = list(args.survey_csv)

    if args.survey_well_id:
        if len(args.survey_well_id) != len(survey_paths):
            raise SystemExit(
                f"[build_baseline] ERROR: --survey_well_id count ({len(args.survey_well_id)}) "
                f"must match --survey_csv count ({len(survey_paths)})."
            )
        survey_ids = [str(x).strip() for x in args.survey_well_id]
        survey_id_sources = ["explicit_arg"] * len(survey_ids)
    else:
        survey_ids = [_infer_well_id_from_survey_path(p) for p in survey_paths]
        survey_id_sources = ["filename_inference"] * len(survey_ids)

    build_plan = _build_wellpath_plan(
        module_out_dir=module_out_dir,
        survey_paths=survey_paths,
        survey_ids=survey_ids,
        survey_id_sources=survey_id_sources,
    )

    available = set(wellheads_lookup.keys())
    generated_wellpaths: List[str] = []
    wellpath_records: List[Dict[str, Any]] = []

    for plan in build_plan:
        survey_csv = plan["survey_csv_arg"]
        wid = plan["well_id"]
        well_id_source = plan["well_id_source"]

        if wid not in available:
            raise SystemExit(
                f"[build_baseline] ERROR: well_id={wid!r} not found in wellheads_csv.\n"
                f"  survey_csv={survey_csv}\n"
                f"  Tip: pass --survey_well_id explicitly, or rename survey filename to include well_id.\n"
                f"  Available examples: {sorted(list(available))[:10]}"
            )

        input_wellhead_xyz_m = dict(wellheads_lookup[wid])
        out_csv = plan["out_csv"]

        should_rebuild = True
        reuse_reason = None

        if (not args.overwrite_wellpath) and os.path.exists(out_csv):
            try:
                verified_assets = _load_verified_per_well_assets(
                    module_out_dir=module_out_dir,
                    plan=plan,
                    input_wellhead_xyz_m=input_wellhead_xyz_m,
                )
                wellpath_records.append(
                    _build_record_from_verified_assets(
                        module_out_dir=module_out_dir,
                        plan=plan,
                        verified_assets=verified_assets,
                        build_status="existing_reused",
                    )
                )
                generated_wellpaths.append(out_csv)
                should_rebuild = False
            except Exception as exc:
                reuse_reason = _build_reuse_repair_reason(exc)
                raise SystemExit(
                    f"[build_baseline] ERROR: reuse validation failed for well_id={wid!r}; {reuse_reason}. "
                    "Existing wellpath CSV assets must be repaired or rebuilt explicitly; automatic fallback rebuild is disabled."
                ) from exc

        if not should_rebuild:
            continue

        wp_argv = [
            "--survey_csv",
            survey_csv,
            "--wellhead_x",
            str(float(input_wellhead_xyz_m["X_m"])),
            "--wellhead_y",
            str(float(input_wellhead_xyz_m["Y_m"])),
            "--wellhead_z",
            str(float(input_wellhead_xyz_m["Z_m"])),
            "--wellhead_is_gk_m",
            "--mag_decl_deg",
            str(float(args.mag_decl_deg)),
            "--out_csv",
            out_csv,
            "--well_id",
            wid,
            "--well_id_source",
            well_id_source,
        ]
        ret_wp = _call_module_main(wp_mod, wp_argv)
        if ret_wp != 0:
            raise SystemExit(f"[build_baseline] ERROR: build_wellpath_gk105 failed for well_id={wid}, exit={ret_wp}")

        try:
            verified_assets = _load_verified_per_well_assets(
                module_out_dir=module_out_dir,
                plan=plan,
                input_wellhead_xyz_m=input_wellhead_xyz_m,
            )
        except Exception as exc:
            raise SystemExit(
                "[build_baseline] ERROR: per-well assets exist but did not pass post-build validation. "
                f"well_id={wid!r}, reason={exc}"
            ) from exc

        generated_wellpaths.append(out_csv)
        wellpath_records.append(
            _build_record_from_verified_assets(
                module_out_dir=module_out_dir,
                plan=plan,
                verified_assets=verified_assets,
                build_status="success",
            )
        )

    from microseis_ds.baseline.build import build_project_from_metadata_gk105 as proj_mod

    proj_argv = [
        "--gps_txt",
        args.gps_txt,
        "--out_dir",
        module_out_dir,
        "--project_name",
        args.project_name,
    ]
    if args.force:
        proj_argv += ["--force"]

    ret_proj = _call_module_main(proj_mod, proj_argv)
    if ret_proj != 0:
        raise SystemExit(f"[build_baseline] ERROR: build_project_from_metadata_gk105 failed, exit={ret_proj}")

    wellpaths_index_path = _write_wellpaths_index(module_out_dir, wellpath_records)
    _patch_per_well_metadata_index_flag(module_out_dir, wellpath_records)
    _patch_wellpath_build_summary_index_flag(module_out_dir, wellpath_records)
    _update_baseline_build_summary(module_out_dir, args, wellpath_records)
    _finalize_manifest(module_out_dir, args, wellpath_records)

    print("[build_baseline] DONE")
    print("  module_out_dir:", module_out_dir)
    print("  build_outputs:")
    print("   -", os.path.join(module_out_dir, "project.json"))
    print("   -", os.path.join(module_out_dir, "manifest.json"))
    print("   -", os.path.join(module_out_dir, "stations.csv"))
    print("   -", os.path.join(module_out_dir, "links", "traceid_station.csv"))
    print("   -", os.path.join(module_out_dir, "meta", "stations_contract_summary.json"))
    print("   -", os.path.join(module_out_dir, "meta", "traceid_station_summary.json"))
    print("   -", os.path.join(module_out_dir, "meta", "baseline_build_summary.json"))
    print("   -", os.path.join(module_out_dir, "meta", "wellpath_build_summary.json"))
    print("   -", wellpaths_index_path)
    print("   -", os.path.join(module_out_dir, HISTORY_DIRNAME))
    print("  wellpaths:")
    for p in generated_wellpaths:
        print("   -", p)
    print("  note: Recommended mapping policy is explicit --survey_well_id; filename inference is retained only as compatibility fallback.")
    print(f"  note: lon0 is locked to {LOCKED_LON0:.1f} in this baseline build entry.")
    print("  note: wellpaths_index / baseline_build_summary / manifest now register canonical per-well wellhead coordinates only.")
    print("  note: raw wellhead input is retained in per-well metadata / wellpath_build_summary for audit traceability, not in entry-level registry items.")
    print("  note: QC is NOT generated here. Run: python scripts/qc_baseline.py ...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
