#!/usr/bin/env python
# -*- coding: utf-8 -*-
# microseis_ds/baseline/qc/qc_contract_baseline.py
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from microseis_ds.common.schema import check_csv_contract


CHECK_NAME = "qc_contract_baseline"
WELLPATH_CSV_MISSING_WELL_ID_CODE = "wellpath_csv_missing_well_id_column"
METADATA_INTEGRITY_FIRST_ROW_MISMATCH_CODE = "metadata_integrity_first_row_mismatch"
STATIONS_REQUIRED_COORD_COLUMNS = ("X_m", "Y_m", "Z_m")


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


def _rel(base: str, p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    try:
        return os.path.relpath(os.path.abspath(p), os.path.abspath(base)).replace("\\", "/")
    except Exception:
        return p


def _read_csv_nonempty(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return df


def _add_issue(bucket: List[Dict[str, Any]], severity: str, code: str, message: str, **extra: Any) -> None:
    item = {"severity": severity, "code": code, "message": message}
    item.update(extra)
    bucket.append(item)


def _is_finite_series(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    return vals.notna() & vals.map(math.isfinite)


def _canonical_xyz_from_index_item(item: Dict[str, Any]) -> Optional[Dict[str, float]]:
    xyz = item.get("wellhead_xyz_m")
    if isinstance(xyz, dict) and all(k in xyz for k in ("X_m", "Y_m", "Z_m")):
        return {k: float(xyz[k]) for k in ("X_m", "Y_m", "Z_m")}
    return None


def _canonical_xyz_from_metadata(md: Dict[str, Any]) -> Optional[Dict[str, float]]:
    xyz = md.get("canonical_wellhead_xyz_m")
    if isinstance(xyz, dict) and all(k in xyz for k in ("X_m", "Y_m", "Z_m")):
        return {k: float(xyz[k]) for k in ("X_m", "Y_m", "Z_m")}
    return None


def _safe_float_dict(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Dict[str, float]]:
    if not isinstance(d, dict):
        return None
    out: Dict[str, float] = {}
    for k in keys:
        if k not in d:
            return None
        out[k] = float(d[k])
    return out


def _compare_float_dict(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]], tol: float = 1e-4) -> bool:
    if a is None or b is None:
        return False
    for k in a.keys():
        if k not in b:
            return False
        if abs(float(a[k]) - float(b[k])) > tol:
            return False
    return True


def _default_paths(out_root: str, module_name: str) -> Dict[str, str]:
    module_dir = os.path.join(out_root, module_name)
    return {
        "module_dir": module_dir,
        "project_json": os.path.join(module_dir, "project.json"),
        "manifest_json": os.path.join(module_dir, "manifest.json"),
        "stations_csv": os.path.join(module_dir, "stations.csv"),
        "baseline_build_summary_json": os.path.join(module_dir, "meta", "baseline_build_summary.json"),
        "wellpaths_index_json": os.path.join(module_dir, "meta", "wellpaths_index.json"),
        "wellpath_build_summary_json": os.path.join(module_dir, "meta", "wellpath_build_summary.json"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Baseline contract / registry / metadata QC")
    ap.add_argument("--out_root", default="prepared_project", help="prepared_project root")
    ap.add_argument("--module_name", default="baseline", help="module name (default: baseline)")
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
        help="Optional explicit wellpath csv(s), repeatable. If omitted, infer from manifest/index or file glob.",
    )
    ap.add_argument(
        "--out_json",
        default=None,
        help="Default: <out_root>/qc/<module_name>/reports/qc_contract_baseline.json",
    )
    return ap


def _collect_wellpath_relpaths(
    *,
    module_dir: str,
    manifest: Optional[Dict[str, Any]],
    wellpaths_index: Optional[Dict[str, Any]],
    explicit_wellpath_csvs: Optional[List[str]],
) -> List[str]:
    rels: List[str] = []
    if explicit_wellpath_csvs:
        for p in explicit_wellpath_csvs:
            rels.append(_rel(module_dir, p) or p)
    if manifest:
        assets = manifest.get("assets", {}) if isinstance(manifest.get("assets", {}), dict) else {}
        for p in list(assets.get("wellpath_csvs", []) or []):
            rels.append(str(p))
    if wellpaths_index:
        for item in list(wellpaths_index.get("items", []) or []):
            rel = str(item.get("wellpath_csv") or "").strip()
            if rel:
                rels.append(rel)
    if not rels:
        for p in sorted(glob.glob(os.path.join(module_dir, "wellpath_*_gk.csv"))):
            rels.append(_rel(module_dir, p) or p)
    out: List[str] = []
    seen = set()
    for x in rels:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main() -> int:
    args = build_arg_parser().parse_args()

    defaults = _default_paths(args.out_root, args.module_name)
    module_dir = defaults["module_dir"]
    qc_root = os.path.join(args.out_root, "qc", args.module_name)
    reports_dir = os.path.join(qc_root, "reports")
    _ensure_dir(reports_dir)

    out_json = args.out_json or os.path.join(reports_dir, "qc_contract_baseline.json")
    stations_csv = args.stations_csv or defaults["stations_csv"]
    manifest_json = args.manifest_json or defaults["manifest_json"]
    baseline_build_summary_json = args.baseline_build_summary_json or defaults["baseline_build_summary_json"]
    wellpaths_index_json = args.wellpaths_index_json or defaults["wellpaths_index_json"]
    wellpath_build_summary_json = args.wellpath_build_summary_json or defaults["wellpath_build_summary_json"]

    top_issues: List[Dict[str, Any]] = []

    asset_paths = {
        "module_dir": module_dir,
        "stations_csv": stations_csv,
        "manifest_json": manifest_json,
        "baseline_build_summary_json": baseline_build_summary_json,
        "wellpaths_index_json": wellpaths_index_json,
        "wellpath_build_summary_json": wellpath_build_summary_json,
    }

    existence_items: List[Dict[str, Any]] = []
    for name, path in asset_paths.items():
        if name == "module_dir":
            continue
        exists = os.path.exists(path)
        existence_items.append({"name": name, "path": path, "exists": bool(exists)})
        if not exists:
            _add_issue(top_issues, "error", "missing_required_asset", f"Required asset missing: {name}", path=path)

    manifest = _json_load(manifest_json) if os.path.exists(manifest_json) else None
    baseline_build_summary = _json_load(baseline_build_summary_json) if os.path.exists(baseline_build_summary_json) else None
    wellpaths_index = _json_load(wellpaths_index_json) if os.path.exists(wellpaths_index_json) else None
    wellpath_build_summary = _json_load(wellpath_build_summary_json) if os.path.exists(wellpath_build_summary_json) else None

    stations_issues: List[Dict[str, Any]] = []
    stations_result: Dict[str, Any] = {
        "status": "failed",
        "path": stations_csv,
        "row_count": None,
        "contract": None,
        "duplicates": {},
        "issues": stations_issues,
    }
    if os.path.exists(stations_csv):
        df_sta = _read_csv_nonempty(stations_csv)
        stations_result["row_count"] = int(len(df_sta))
        stations_result["column_count"] = int(len(df_sta.columns))
        if df_sta.empty:
            _add_issue(stations_issues, "error", "stations_empty", "stations.csv is empty", path=stations_csv)
        else:
            contract = check_csv_contract(df_sta, "stations")
            stations_result["contract"] = contract
            missing_required = list(contract.get("missing_required", []))
            null_in_required = list(contract.get("null_in_required", []))
            if missing_required:
                _add_issue(
                    stations_issues,
                    "error",
                    "stations_missing_required",
                    "stations.csv missing required columns",
                    columns=missing_required,
                )
                for col in STATIONS_REQUIRED_COORD_COLUMNS:
                    if col in missing_required:
                        _add_issue(
                            stations_issues,
                            "error",
                            f"stations_missing_required_{col}",
                            f"stations.csv missing required coordinate column {col}",
                            column=col,
                        )
            if null_in_required:
                _add_issue(
                    stations_issues,
                    "error",
                    "stations_null_in_required",
                    "stations.csv has null in required columns",
                    columns=null_in_required,
                )
                for col in STATIONS_REQUIRED_COORD_COLUMNS:
                    if col in null_in_required:
                        _add_issue(
                            stations_issues,
                            "error",
                            f"stations_null_required_{col}",
                            f"stations.csv has null in required coordinate column {col}",
                            column=col,
                        )
            if contract.get("extra_columns"):
                _add_issue(stations_issues, "warning", "stations_extra_columns", "stations.csv has extra columns", columns=list(contract.get("extra_columns", [])))

            for col in ("station_id", "inline_3D"):
                if col in df_sta.columns:
                    s = df_sta[col].astype(str).str.strip()
                    dup = s[s.duplicated(keep=False)]
                    if len(dup) > 0:
                        values = sorted(set(dup.tolist()))[:20]
                        stations_result["duplicates"][col] = values
                        sev = "error" if col == "station_id" else "warning"
                        _add_issue(stations_issues, sev, f"duplicate_{col}", f"Duplicate {col} detected", values=values)
                    empty_mask = s == ""
                    if empty_mask.any():
                        _add_issue(stations_issues, "error", f"empty_{col}", f"Empty {col} detected", row_indices=df_sta.index[empty_mask].tolist()[:20])

            bad_numeric: Dict[str, List[int]] = {}
            for col in STATIONS_REQUIRED_COORD_COLUMNS:
                if col in df_sta.columns:
                    mask = ~_is_finite_series(df_sta[col])
                    if mask.any():
                        bad_numeric[col] = df_sta.index[mask].tolist()[:20]
                        _add_issue(
                            stations_issues,
                            "error",
                            f"stations_invalid_required_coord_{col}",
                            f"Non-finite or invalid numeric values in required coordinate column {col}",
                            row_indices=bad_numeric[col],
                            column=col,
                        )
            stations_result["invalid_numeric_rows"] = bad_numeric
            stations_result["status"] = "passed" if not any(i["severity"] == "error" for i in stations_issues) else "failed"

    registry_issues: List[Dict[str, Any]] = []
    registry_result: Dict[str, Any] = {
        "status": "failed",
        "issues": registry_issues,
        "consistency_checks": [],
    }

    if manifest and baseline_build_summary and wellpaths_index and wellpath_build_summary:
        assets = manifest.get("assets", {}) if isinstance(manifest.get("assets", {}), dict) else {}
        b_outputs = baseline_build_summary.get("outputs", {}) if isinstance(baseline_build_summary.get("outputs", {}), dict) else {}
        wp_records = list(wellpath_build_summary.get("records", []) or [])
        idx_items = list(wellpaths_index.get("items", []) or [])

        checks: List[Tuple[str, bool, Dict[str, Any]]] = []
        checks.append((
            "manifest_asset_baseline_build_summary",
            str(assets.get("baseline_build_summary_json")) == "meta/baseline_build_summary.json",
            {"actual": assets.get("baseline_build_summary_json")},
        ))
        checks.append((
            "manifest_asset_wellpath_build_summary",
            str(assets.get("wellpath_build_summary_json")) == "meta/wellpath_build_summary.json",
            {"actual": assets.get("wellpath_build_summary_json")},
        ))
        checks.append((
            "manifest_asset_wellpaths_index",
            str(assets.get("wellpaths_index_json")) == "meta/wellpaths_index.json",
            {"actual": assets.get("wellpaths_index_json")},
        ))
        checks.append((
            "baseline_outputs_manifest_json",
            str(b_outputs.get("manifest_json")) == "manifest.json",
            {"actual": b_outputs.get("manifest_json")},
        ))
        checks.append((
            "baseline_outputs_wellpaths_index",
            str(b_outputs.get("wellpaths_index_json")) == "meta/wellpaths_index.json",
            {"actual": b_outputs.get("wellpaths_index_json")},
        ))
        checks.append((
            "registry_complete_flag_manifest",
            bool(manifest.get("state", {}).get("full_baseline_registry_complete")) is True,
            {"actual": manifest.get("state", {}).get("full_baseline_registry_complete")},
        ))
        checks.append((
            "registry_complete_flag_baseline_summary",
            bool(baseline_build_summary.get("full_baseline_registry_complete")) is True,
            {"actual": baseline_build_summary.get("full_baseline_registry_complete")},
        ))
        checks.append((
            "registry_complete_flag_wellpaths_index",
            bool(wellpaths_index.get("full_baseline_registry_complete")) is True,
            {"actual": wellpaths_index.get("full_baseline_registry_complete")},
        ))
        checks.append((
            "registry_complete_flag_wellpath_summary",
            bool(wellpath_build_summary.get("full_baseline_registry_complete")) is True,
            {"actual": wellpath_build_summary.get("full_baseline_registry_complete")},
        ))
        checks.append((
            "wellpath_count_manifest_vs_index",
            int(manifest.get("state", {}).get("wellpath_count", -1)) == int(wellpaths_index.get("row_count", -2)),
            {
                "manifest": manifest.get("state", {}).get("wellpath_count"),
                "index": wellpaths_index.get("row_count"),
            },
        ))
        checks.append((
            "wellpath_count_baseline_summary_vs_index",
            int(baseline_build_summary.get("wellpaths", {}).get("row_count", -1)) == int(wellpaths_index.get("row_count", -2)),
            {
                "baseline_build_summary": baseline_build_summary.get("wellpaths", {}).get("row_count"),
                "index": wellpaths_index.get("row_count"),
            },
        ))
        checks.append((
            "wellpath_summary_record_count_vs_records_len",
            int(wellpath_build_summary.get("record_count", -1)) == len(wp_records),
            {
                "record_count": wellpath_build_summary.get("record_count"),
                "len_records": len(wp_records),
            },
        ))
        checks.append((
            "wellpath_summary_current_view_count_vs_records_len",
            int(wellpath_build_summary.get("current_view_record_count", -1)) == len(wp_records),
            {
                "current_view_record_count": wellpath_build_summary.get("current_view_record_count"),
                "len_records": len(wp_records),
            },
        ))
        checks.append((
            "entry_registry_current_view_available",
            bool(wellpath_build_summary.get("entry_registry_current_view_available")) is True,
            {"actual": wellpath_build_summary.get("entry_registry_current_view_available")},
        ))

        idx_well_ids = [str(x.get("well_id") or "").strip() for x in idx_items]
        wp_summary_well_ids = [str(x.get("well_id") or "").strip() for x in wp_records]
        checks.append((
            "well_ids_index_vs_wellpath_summary",
            idx_well_ids == wp_summary_well_ids == list(wellpath_build_summary.get("current_view_well_ids", []) or []),
            {
                "index_well_ids": idx_well_ids,
                "summary_record_well_ids": wp_summary_well_ids,
                "current_view_well_ids": list(wellpath_build_summary.get("current_view_well_ids", []) or []),
            },
        ))

        registry_result["consistency_checks"] = [
            {"name": name, "passed": bool(ok), **detail} for name, ok, detail in checks
        ]
        for name, ok, detail in checks:
            if not ok:
                _add_issue(registry_issues, "error", name, f"Registry consistency check failed: {name}", **detail)

        idx_map = {str(item.get("wellpath_csv") or "").strip(): item for item in idx_items}
        wp_map = {str((rec.get("outputs", {}) if isinstance(rec.get("outputs", {}), dict) else {}).get("wellpath_csv") or rec.get("wellpath_csv") or "").strip(): rec for rec in wp_records}
        meta_paths = set(str(x) for x in list(assets.get("wellpath_metadata_jsons", []) or []))
        registry_result["wellpath_item_checks"] = []
        for rel, idx_item in idx_map.items():
            item_issues: List[Dict[str, Any]] = []
            rec = wp_map.get(rel)
            if rec is None:
                _add_issue(item_issues, "error", "missing_wellpath_summary_record", "No wellpath_build_summary record for wellpath", wellpath_csv=rel)
            meta_rel = str(idx_item.get("wellpath_metadata_json") or "").strip()
            meta_abs = os.path.join(module_dir, meta_rel) if meta_rel else None
            md = _json_load(meta_abs) if meta_abs and os.path.exists(meta_abs) else None
            if not meta_rel:
                _add_issue(item_issues, "error", "missing_metadata_relpath", "wellpaths_index item missing wellpath_metadata_json", wellpath_csv=rel)
            elif meta_rel not in meta_paths:
                _add_issue(item_issues, "error", "metadata_not_in_manifest", "wellpath metadata json not registered in manifest.assets.wellpath_metadata_jsons", metadata_json=meta_rel)
            elif md is None:
                _add_issue(item_issues, "error", "metadata_file_missing", "wellpath metadata json missing on disk", metadata_json=meta_rel)
            else:
                if str(md.get("well_id") or "").strip() != str(idx_item.get("well_id") or "").strip():
                    _add_issue(item_issues, "error", "well_id_mismatch_metadata_vs_index", "well_id mismatch between metadata and wellpaths_index", metadata_well_id=md.get("well_id"), index_well_id=idx_item.get("well_id"))
                if str(md.get("outputs", {}).get("wellpath_csv") or "").strip() != rel:
                    _add_issue(item_issues, "error", "wellpath_csv_mismatch_metadata_vs_index", "wellpath_csv mismatch between metadata and wellpaths_index", metadata_wellpath_csv=md.get("outputs", {}).get("wellpath_csv"), index_wellpath_csv=rel)
                if bool(md.get("index_included")) is not True:
                    _add_issue(item_issues, "error", "metadata_index_not_included", "metadata.index_included must be true after Step 4 finalization", actual=md.get("index_included"))
                if str(md.get("wellpaths_index_json") or "") != "meta/wellpaths_index.json":
                    _add_issue(item_issues, "error", "metadata_wellpaths_index_json_mismatch", "metadata.wellpaths_index_json should point to meta/wellpaths_index.json", actual=md.get("wellpaths_index_json"))
                if bool(md.get("full_baseline_registry_complete")) is not True:
                    _add_issue(item_issues, "error", "metadata_registry_complete_flag_false", "metadata.full_baseline_registry_complete must be true", actual=md.get("full_baseline_registry_complete"))
                idx_xyz = _canonical_xyz_from_index_item(idx_item)
                md_xyz = _canonical_xyz_from_metadata(md)
                if idx_xyz is not None and md_xyz is not None and not _compare_float_dict(idx_xyz, md_xyz):
                    _add_issue(item_issues, "error", "canonical_xyz_mismatch_metadata_vs_index", "Canonical wellhead XYZ mismatch between metadata and wellpaths_index", index_xyz=idx_xyz, metadata_xyz=md_xyz)

            if rec is not None:
                if str(rec.get("well_id") or "").strip() != str(idx_item.get("well_id") or "").strip():
                    _add_issue(item_issues, "error", "well_id_mismatch_summary_vs_index", "well_id mismatch between wellpath_build_summary and wellpaths_index", summary_well_id=rec.get("well_id"), index_well_id=idx_item.get("well_id"))
                rec_meta = str((rec.get("outputs", {}) if isinstance(rec.get("outputs", {}), dict) else {}).get("wellpath_metadata_json") or "").strip()
                if meta_rel and rec_meta != meta_rel:
                    _add_issue(item_issues, "error", "metadata_relpath_mismatch_summary_vs_index", "Metadata relpath mismatch between wellpath_build_summary and wellpaths_index", summary_metadata_json=rec_meta, index_metadata_json=meta_rel)
                idx_integrity = idx_item.get("wellpath_integrity")
                rec_integrity = rec.get("wellpath_integrity")
                if isinstance(idx_integrity, dict) and isinstance(rec_integrity, dict):
                    if int(idx_integrity.get("row_count", -1)) != int(rec_integrity.get("row_count", -2)):
                        _add_issue(item_issues, "error", "integrity_row_count_mismatch_summary_vs_index", "wellpath_integrity.row_count mismatch between summary and index", summary=rec_integrity.get("row_count"), index=idx_integrity.get("row_count"))

            registry_result["wellpath_item_checks"].append(
                {
                    "wellpath_csv": rel,
                    "well_id": idx_item.get("well_id"),
                    "status": "passed" if not any(x["severity"] == "error" for x in item_issues) else "failed",
                    "issues": item_issues,
                }
            )
            registry_issues.extend(item_issues)

        registry_result["status"] = "passed" if not any(i["severity"] == "error" for i in registry_issues) else "failed"

    wellpath_rels = _collect_wellpath_relpaths(
        module_dir=module_dir,
        manifest=manifest,
        wellpaths_index=wellpaths_index,
        explicit_wellpath_csvs=args.wellpath_csv,
    )
    idx_items_by_rel = {}
    if wellpaths_index:
        idx_items_by_rel = {str(item.get("wellpath_csv") or "").strip(): item for item in list(wellpaths_index.get("items", []) or [])}
    wp_records_by_rel = {}
    if wellpath_build_summary:
        wp_records_by_rel = {
            str((rec.get("outputs", {}) if isinstance(rec.get("outputs", {}), dict) else {}).get("wellpath_csv") or rec.get("wellpath_csv") or "").strip(): rec
            for rec in list(wellpath_build_summary.get("records", []) or [])
        }

    wellpath_items: List[Dict[str, Any]] = []
    wellpath_issues: List[Dict[str, Any]] = []
    for rel in wellpath_rels:
        abs_path = os.path.join(module_dir, rel) if not os.path.isabs(rel) else rel
        item_issues: List[Dict[str, Any]] = []
        item: Dict[str, Any] = {
            "wellpath_csv": rel,
            "path": abs_path,
            "exists": os.path.exists(abs_path),
            "issues": item_issues,
        }
        if not os.path.exists(abs_path):
            _add_issue(item_issues, "error", "wellpath_file_missing", "wellpath csv missing", path=abs_path)
            wellpath_items.append(item)
            wellpath_issues.extend(item_issues)
            continue

        df_wp = _read_csv_nonempty(abs_path)
        item["row_count"] = int(len(df_wp))
        item["column_count"] = int(len(df_wp.columns)) if not df_wp.empty else 0
        if df_wp.empty:
            _add_issue(item_issues, "error", "wellpath_empty", "wellpath csv is empty", path=abs_path)
            wellpath_items.append(item)
            wellpath_issues.extend(item_issues)
            continue

        contract = check_csv_contract(df_wp, "wellpath")
        item["contract"] = contract
        if contract.get("missing_required"):
            _add_issue(item_issues, "error", "wellpath_missing_required", "wellpath missing required columns", columns=list(contract.get("missing_required", [])))
        if contract.get("null_in_required"):
            _add_issue(item_issues, "error", "wellpath_null_in_required", "wellpath has null in required columns", columns=list(contract.get("null_in_required", [])))
        if contract.get("extra_columns"):
            _add_issue(item_issues, "warning", "wellpath_extra_columns", "wellpath has extra columns", columns=list(contract.get("extra_columns", [])))

        for col in ("MD", "TVD_m", "X_m", "Y_m", "Z_m"):
            if col in df_wp.columns:
                mask = ~_is_finite_series(df_wp[col])
                if mask.any():
                    _add_issue(item_issues, "error", f"invalid_{col}", f"Non-finite or invalid numeric values in {col}", row_indices=df_wp.index[mask].tolist()[:20])

        if "MD" in df_wp.columns:
            md = pd.to_numeric(df_wp["MD"], errors="coerce")
            if md.notna().sum() >= 2:
                diff = md.diff().iloc[1:]
                bad = diff.index[diff <= 0].tolist()[:20]
                if bad:
                    _add_issue(item_issues, "error", "md_not_strictly_increasing", "MD is not strictly increasing", row_indices=bad)

        csv_has_well_id = "well_id" in df_wp.columns
        item["csv_has_well_id"] = bool(csv_has_well_id)
        idx_item = idx_items_by_rel.get(rel)
        rec = wp_records_by_rel.get(rel)
        md_rel = str(idx_item.get("wellpath_metadata_json") or "").strip() if isinstance(idx_item, dict) else ""
        md_abs = os.path.join(module_dir, md_rel) if md_rel else None
        md = _json_load(md_abs) if md_abs and os.path.exists(md_abs) else None
        resolved_well_id = None
        if csv_has_well_id and df_wp["well_id"].astype(str).str.strip().ne("").any():
            vals = [x for x in sorted(set(df_wp["well_id"].astype(str).str.strip().tolist())) if x]
            item["csv_well_id_values"] = vals[:20]
            resolved_well_id = vals[0] if vals else None
            if len(vals) > 1:
                _add_issue(item_issues, "error", "multiple_well_id_in_csv", "Multiple non-empty well_id values found in wellpath csv", values=vals[:20])
        else:
            _add_issue(item_issues, "warning", WELLPATH_CSV_MISSING_WELL_ID_CODE, "wellpath csv does not carry well_id column; QC falls back to metadata/index association", path=rel)

        index_well_id = str(idx_item.get("well_id") or "").strip() if isinstance(idx_item, dict) else ""
        metadata_well_id = str(md.get("well_id") or "").strip() if isinstance(md, dict) else ""
        summary_well_id = str(rec.get("well_id") or "").strip() if isinstance(rec, dict) else ""
        item["resolved_identity"] = {
            "csv_well_id": resolved_well_id,
            "index_well_id": index_well_id or None,
            "metadata_well_id": metadata_well_id or None,
            "summary_well_id": summary_well_id or None,
        }
        if not any([resolved_well_id, index_well_id, metadata_well_id, summary_well_id]):
            _add_issue(item_issues, "error", "missing_well_id_identity_chain", "well_id is missing across csv + metadata + index + summary", wellpath_csv=rel)

        if isinstance(md, dict):
            if str(md.get("outputs", {}).get("wellpath_csv") or "").strip() != rel:
                _add_issue(item_issues, "error", "metadata_outputs_wellpath_csv_mismatch", "metadata outputs.wellpath_csv mismatch", metadata_value=md.get("outputs", {}).get("wellpath_csv"), expected=rel)
            integ = md.get("wellpath_integrity")
            if isinstance(integ, dict):
                row_count = int(integ.get("row_count", -1))
                if row_count != int(len(df_wp)):
                    _add_issue(item_issues, "error", "metadata_integrity_row_count_mismatch", "metadata wellpath_integrity.row_count mismatch with csv", metadata_row_count=row_count, csv_row_count=int(len(df_wp)))

                first_row = _safe_float_dict(integ.get("first_row", {}), ("MD", "TVD_m", "X_m", "Y_m", "Z_m"))
                if first_row is not None and len(df_wp) > 0:
                    actual_first = {
                        k: float(pd.to_numeric(df_wp.iloc[0][k], errors="coerce")) for k in ("MD", "TVD_m", "X_m", "Y_m", "Z_m")
                    }
                    if not _compare_float_dict(first_row, actual_first):
                        _add_issue(item_issues, "error", METADATA_INTEGRITY_FIRST_ROW_MISMATCH_CODE, "metadata wellpath_integrity.first_row mismatch with csv first row", metadata_first_row=first_row, csv_first_row=actual_first)
            else:
                _add_issue(item_issues, "error", "metadata_missing_wellpath_integrity", "wellpath metadata missing wellpath_integrity block", metadata_json=md_rel)
        else:
            _add_issue(item_issues, "error", "missing_wellpath_metadata", "wellpath metadata file missing or unreadable", metadata_json=md_rel or None)

        item["status"] = "passed" if not any(x["severity"] == "error" for x in item_issues) else "failed"
        wellpath_items.append(item)
        wellpath_issues.extend(item_issues)

    wellpaths_result = {
        "status": "passed" if not any(i["severity"] == "error" for i in wellpath_issues) else "failed",
        "count": len(wellpath_items),
        "items": wellpath_items,
        "issues": wellpath_issues,
    }

    all_issues = list(top_issues) + list(stations_issues) + list(registry_issues) + list(wellpath_issues)
    error_count = sum(1 for x in all_issues if x.get("severity") == "error")
    warning_count = sum(1 for x in all_issues if x.get("severity") == "warning")
    status = "passed" if error_count == 0 else "failed"

    normalized_warnings = [x for x in all_issues if x.get("severity") == "warning"]
    normalized_errors = [x for x in all_issues if x.get("severity") == "error"]

    payload = {
        "check_name": CHECK_NAME,
        "module_name": args.module_name,
        "generated_at": _now_ts(),
        "status": status,
        "counts": {
            "error_count": int(error_count),
            "warning_count": int(warning_count),
            "wellpath_count": int(len(wellpath_items)),
        },
        "inputs": {
            "out_root": args.out_root,
            "module_dir": module_dir,
        },
        "paths": {
            "stations_csv": stations_csv,
            "manifest_json": manifest_json,
            "baseline_build_summary_json": baseline_build_summary_json,
            "wellpaths_index_json": wellpaths_index_json,
            "wellpath_build_summary_json": wellpath_build_summary_json,
            "output_json": out_json,
        },
        "asset_existence": {
            "status": "passed" if not any(i.get("severity") == "error" for i in top_issues) else "failed",
            "items": existence_items,
            "issues": top_issues,
        },
        "stations_qc": stations_result,
        "wellpaths_qc": wellpaths_result,
        "registry_qc": registry_result,
        "warnings": normalized_warnings,
        "errors": normalized_errors,
        "issues": all_issues,
    }

    _json_dump(out_json, payload)
    print(f"[OK] saved: {out_json}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())