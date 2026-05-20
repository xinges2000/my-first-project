#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QC: Spatial overlay (Stations + formal wellheads + formal wellpaths)

Formal default outputs:
  <out_root>/qc/<module_name>/figures/qc_spatial_overlay.png
  <out_root>/qc/<module_name>/reports/qc_spatial_overlay.json

Axis convention:
  x-axis = Y_m (Easting, m)
  y-axis = X_m (Northing, m)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def _default_paths(out_root: str, module_name: str) -> Dict[str, str]:
    module_dir = os.path.join(out_root, module_name)
    qc_root = os.path.join(out_root, "qc", module_name)
    return {
        "module_dir": module_dir,
        "manifest_json": os.path.join(module_dir, "manifest.json"),
        "wellpaths_index_json": os.path.join(module_dir, "meta", "wellpaths_index.json"),
        "stations_csv": os.path.join(module_dir, "stations.csv"),
        "out_png": os.path.join(qc_root, "figures", "qc_spatial_overlay.png"),
        "out_json": os.path.join(qc_root, "reports", "qc_spatial_overlay.json"),
    }


def _ensure_xy_m(df: Optional[pd.DataFrame], kind: str) -> Optional[pd.DataFrame]:
    if df is None or len(df) == 0:
        return df
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]

    if "E_gk_m" in d.columns and "N_gk_m" in d.columns:
        d["E_gk_m"] = pd.to_numeric(d["E_gk_m"], errors="coerce")
        d["N_gk_m"] = pd.to_numeric(d["N_gk_m"], errors="coerce")
        d["X_m"] = d["N_gk_m"]
        d["Y_m"] = d["E_gk_m"]
        return d

    if "X_m" in d.columns and "Y_m" in d.columns:
        d["X_m"] = pd.to_numeric(d["X_m"], errors="coerce")
        d["Y_m"] = pd.to_numeric(d["Y_m"], errors="coerce")
        return d

    if "X" in d.columns and "Y" in d.columns:
        d["X_m"] = pd.to_numeric(d["X"], errors="coerce")
        d["Y_m"] = pd.to_numeric(d["Y"], errors="coerce")
        return d

    raise ValueError(
        f"[qc_plot_spatial_overlay] {kind} cannot infer coordinates. "
        f"Need (X_m,Y_m) or (E_gk_m,N_gk_m) or legacy (X,Y). "
        f"Columns={d.columns.tolist()}"
    )


def _finite_xy(d: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(d["X_m"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(d["Y_m"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _robust_limits(xs: np.ndarray, ys: np.ndarray, pad_ratio: float = 0.10) -> Tuple[float, float, float, float]:
    if xs.size == 1:
        x1 = x99 = float(xs[0])
    else:
        x1, x99 = np.percentile(xs, [1, 99])
    if ys.size == 1:
        y1 = y99 = float(ys[0])
    else:
        y1, y99 = np.percentile(ys, [1, 99])

    dx = (x99 - x1) if x99 > x1 else 1.0
    dy = (y99 - y1) if y99 > y1 else 1.0
    px = dx * pad_ratio
    py = dy * pad_ratio
    return (y1 - py, y99 + py, x1 - px, x99 + px)


def _load_formal_wellpath_relpaths(module_dir: str, manifest_json: Optional[str], wellpaths_index_json: Optional[str]) -> List[str]:
    rels: List[str] = []
    if manifest_json and os.path.exists(manifest_json):
        manifest = _json_load(manifest_json)
        assets = manifest.get("assets", {}) if isinstance(manifest.get("assets", {}), dict) else {}
        rels.extend([str(x) for x in list(assets.get("wellpath_csvs", []) or []) if str(x).strip()])
    if wellpaths_index_json and os.path.exists(wellpaths_index_json):
        wp_idx = _json_load(wellpaths_index_json)
        for item in list(wp_idx.get("items", []) or []):
            rel = str(item.get("wellpath_csv") or "").strip()
            if rel:
                rels.append(rel)
    seen = set()
    out: List[str] = []
    for rel in rels:
        if rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def _load_formal_wellheads_from_index(wellpaths_index_json: Optional[str]) -> Optional[pd.DataFrame]:
    if not wellpaths_index_json or not os.path.exists(wellpaths_index_json):
        return None
    payload = _json_load(wellpaths_index_json)
    rows: List[Dict[str, Any]] = []
    for item in list(payload.get("items", []) or []):
        xyz = item.get("wellhead_xyz_m")
        if isinstance(xyz, dict) and all(k in xyz for k in ("X_m", "Y_m", "Z_m")):
            rows.append(
                {
                    "well_id": item.get("well_id"),
                    "X_m": float(xyz["X_m"]),
                    "Y_m": float(xyz["Y_m"]),
                    "Z_m": float(xyz["Z_m"]),
                }
            )
    return pd.DataFrame(rows) if rows else None


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="prepared_project", help="prepared_project root")
    ap.add_argument("--module_name", default="baseline", help="module name for qc folder")
    ap.add_argument("--stations_csv", default=None, help="Default: <out_root>/<module_name>/stations.csv")
    ap.add_argument("--manifest_json", default=None, help="Default: <out_root>/<module_name>/manifest.json")
    ap.add_argument("--wellpaths_index_json", default=None, help="Default: <out_root>/<module_name>/meta/wellpaths_index.json")
    ap.add_argument("--wellheads_csv", default=None, help="Compatibility-only optional external wellheads csv")
    ap.add_argument("--wellpath_csv", action="append", default=None, help="Optional explicit wellpath csv(s), repeatable")
    ap.add_argument("--out_png", default=None, help="Default: <out_root>/qc/<module_name>/figures/qc_spatial_overlay.png")
    ap.add_argument("--out_json", default=None, help="Default: <out_root>/qc/<module_name>/reports/qc_spatial_overlay.json")
    ap.add_argument("--title", default=None)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    defaults = _default_paths(args.out_root, args.module_name)
    module_dir = defaults["module_dir"]
    stations_csv = args.stations_csv or defaults["stations_csv"]
    manifest_json = args.manifest_json or defaults["manifest_json"]
    wellpaths_index_json = args.wellpaths_index_json or defaults["wellpaths_index_json"]
    out_png = args.out_png or defaults["out_png"]
    out_json = args.out_json or defaults["out_json"]

    _ensure_dir(os.path.dirname(out_png))
    _ensure_dir(os.path.dirname(out_json))

    warnings: List[str] = []
    issues: List[Dict[str, Any]] = []

    if not os.path.exists(stations_csv):
        raise FileNotFoundError(f"stations_csv not found: {stations_csv}")

    df_s = pd.read_csv(stations_csv)
    df_s = _ensure_xy_m(df_s, "stations")

    df_wh: Optional[pd.DataFrame] = None
    if args.wellheads_csv:
        if not os.path.exists(args.wellheads_csv):
            raise FileNotFoundError(f"wellheads_csv not found: {args.wellheads_csv}")
        df_wh = _ensure_xy_m(pd.read_csv(args.wellheads_csv), "wellheads")
        warnings.append("Overlay QC used compatibility wellheads_csv input instead of formal wellpaths_index-derived wellheads.")
    else:
        df_wh = _load_formal_wellheads_from_index(wellpaths_index_json)
        if df_wh is None:
            warnings.append("No formal wellheads derived from wellpaths_index; overlay will use stations and available wellpaths only.")

    wellpath_paths: List[str] = []
    if args.wellpath_csv:
        wellpath_paths = [str(p) for p in args.wellpath_csv]
    else:
        rels = _load_formal_wellpath_relpaths(module_dir, manifest_json, wellpaths_index_json)
        wellpath_paths = [os.path.join(module_dir, rel) for rel in rels]

    df_wp: Optional[pd.DataFrame] = None
    used_wellpath_paths: List[str] = []
    if wellpath_paths:
        chunks: List[pd.DataFrame] = []
        for p in wellpath_paths:
            if not os.path.exists(p):
                issues.append({"severity": "warning", "code": "wellpath_missing_for_overlay", "message": "Wellpath file missing for overlay", "path": p})
                continue
            d = pd.read_csv(p)
            if "well_id" not in d.columns:
                base = os.path.basename(p)
                wid = os.path.splitext(base)[0].replace("wellpath_", "").replace("_gk", "")
                d["well_id"] = wid
            chunks.append(d)
            used_wellpath_paths.append(p)
        if chunks:
            df_wp = pd.concat(chunks, ignore_index=True)
            df_wp = _ensure_xy_m(df_wp, "wellpaths")

    X_all: List[np.ndarray] = []
    Y_all: List[np.ndarray] = []

    Xs, Ys = _finite_xy(df_s)
    if Xs.size == 0:
        raise ValueError("[qc_plot_spatial_overlay] No valid station coordinates to plot.")
    X_all.append(Xs)
    Y_all.append(Ys)

    if df_wh is not None and len(df_wh) > 0:
        Xh, Yh = _finite_xy(df_wh)
        if Xh.size > 0:
            X_all.append(Xh)
            Y_all.append(Yh)

    if df_wp is not None and len(df_wp) > 0:
        Xw, Yw = _finite_xy(df_wp)
        if Xw.size > 0:
            X_all.append(Xw)
            Y_all.append(Yw)

    X_cat = np.concatenate(X_all)
    Y_cat = np.concatenate(Y_all)
    if X_cat.size == 0 or Y_cat.size == 0:
        raise ValueError("[qc_plot_spatial_overlay] No valid coordinates to plot.")

    y_min, y_max, x_min, x_max = _robust_limits(X_cat, Y_cat, pad_ratio=0.10)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(df_s["Y_m"].values, df_s["X_m"].values, marker="^", s=30, label="stations")

    if df_wh is not None and len(df_wh) > 0:
        ax.scatter(df_wh["Y_m"].values, df_wh["X_m"].values, marker="o", s=40, label="wellheads")
        if "well_id" in df_wh.columns:
            for _, r in df_wh.iterrows():
                try:
                    ax.text(r["Y_m"], r["X_m"], str(r["well_id"]), fontsize=9)
                except Exception:
                    pass

    if df_wp is not None and len(df_wp) > 0:
        sort_col = None
        for c in ["MD", "md_m", "MD_m", "md"]:
            if c in df_wp.columns:
                sort_col = c
                break
        if "well_id" in df_wp.columns:
            for wid, g in df_wp.groupby("well_id", sort=True):
                g2 = g.sort_values(by=sort_col) if sort_col else g
                ax.plot(g2["Y_m"].values, g2["X_m"].values, linewidth=1.2, label=f"wellpath:{wid}")
        else:
            g2 = df_wp.sort_values(by=sort_col) if sort_col else df_wp
            ax.plot(g2["Y_m"].values, g2["X_m"].values, linewidth=1.2, label="wellpath")

    ax.set_xlabel("Y_m (Easting, m)")
    ax.set_ylabel("X_m (Northing, m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best")
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(x_min, x_max)
    ax.set_title(args.title or "QC Spatial overlay (Stations + Wells)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "check_name": "qc_spatial_overlay",
        "module_name": args.module_name,
        "generated_at": _now_ts(),
        "status": "passed",
        "inputs": {
            "stations_csv": stations_csv,
            "manifest_json": manifest_json if os.path.exists(manifest_json) else None,
            "wellpaths_index_json": wellpaths_index_json if os.path.exists(wellpaths_index_json) else None,
            "wellheads_csv": args.wellheads_csv,
            "wellpath_csvs": used_wellpath_paths,
        },
        "counts": {
            "station_count": int(len(df_s)),
            "wellhead_count": int(len(df_wh)) if df_wh is not None else 0,
            "wellpath_point_count": int(len(df_wp)) if df_wp is not None else 0,
            "wellpath_count": int(df_wp["well_id"].nunique()) if (df_wp is not None and "well_id" in df_wp.columns) else (1 if df_wp is not None else 0),
        },
        "plot": {
            "out_png": out_png,
            "x_axis": "Y_m (Easting, m)",
            "y_axis": "X_m (Northing, m)",
            "limits": {
                "y_min": float(y_min),
                "y_max": float(y_max),
                "x_min": float(x_min),
                "x_max": float(x_max),
            },
        },
        "warnings": warnings,
        "issues": issues,
    }
    if any(i.get("severity") == "error" for i in issues):
        payload["status"] = "failed"

    _json_dump(out_json, payload)
    print("[OK] saved:", out_png)
    print("[OK] saved:", out_json)
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())