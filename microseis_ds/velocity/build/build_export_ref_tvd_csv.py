from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from microseis_ds.velocity.runtime.ref_md_tvd import load_wellpath_full, load_ref_model


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# ✅ ONLY for md_min/md_max logic: bin width is set from ref_tvd dz in export_ref_tvd_csv()
_MD_ENVELOPE_BIN_W_M: Optional[float] = None


def _build_md_envelope_from_wellpath_csv(wellpath_csv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build TVD-keyed md_min/md_max envelopes from the RAW wellpath CSV.

    Fix for horizontal wells:
      - Instead of grouping by (rounded) raw TVD values, bin TVD into bands.
      - Band width is auto-set to ref_tvd.csv dz (via module-level _MD_ENVELOPE_BIN_W_M).
      - For each TVD bin, compute md_min/md_max.

    NOTE:
      - Signature unchanged to keep rest of file untouched.
    """
    df = pd.read_csv(wellpath_csv)

    md_col = _pick_column(df, ["MD_m", "MD", "md_m", "md"])
    tvd_col = _pick_column(df, ["TVD_m", "TVD", "tvd_m", "tvd", "Z_m", "Z", "z_m", "z"])
    if md_col is None or tvd_col is None:
        raise ValueError(f"wellpath missing MD/TVD columns for md envelope. columns={list(df.columns)}")

    md = pd.to_numeric(df[md_col], errors="coerce").to_numpy(float)
    tvd = pd.to_numeric(df[tvd_col], errors="coerce").to_numpy(float)

    m = np.isfinite(md) & np.isfinite(tvd)
    md = md[m]
    tvd = tvd[m]
    if md.size < 2:
        raise ValueError("wellpath too few valid samples for md envelope.")

    # -------------------------------
    # ✅ NEW: bin width = ref_tvd dz
    # -------------------------------
    bin_w = _MD_ENVELOPE_BIN_W_M
    if bin_w is None or (not np.isfinite(bin_w)) or bin_w <= 0:
        bin_w = 1.0  # fallback (should rarely happen)

    tvd0 = float(np.nanmin(tvd))
    bin_idx = np.floor((tvd - tvd0) / float(bin_w)).astype(np.int64)

    tmp = pd.DataFrame({"bin": bin_idx, "tvd": tvd, "md": md})
    g = tmp.groupby("bin", sort=True)

    tvd_u = g["tvd"].median().to_numpy(float)
    md_min_u = g["md"].min().to_numpy(float)
    md_max_u = g["md"].max().to_numpy(float)

    order = np.argsort(tvd_u)
    tvd_u = tvd_u[order]
    md_min_u = md_min_u[order]
    md_max_u = md_max_u[order]

    # Ensure strictly increasing TVD for interpolation.
    if tvd_u.size < 2 or not np.all(np.diff(tvd_u) > 0):
        # fallback: slightly coarser bins
        bin_w2 = float(bin_w) * 2.0
        bin_idx = np.floor((tvd - tvd0) / bin_w2).astype(np.int64)
        tmp = pd.DataFrame({"bin": bin_idx, "tvd": tvd, "md": md})
        g = tmp.groupby("bin", sort=True)

        tvd_u = g["tvd"].median().to_numpy(float)
        md_min_u = g["md"].min().to_numpy(float)
        md_max_u = g["md"].max().to_numpy(float)

        order = np.argsort(tvd_u)
        tvd_u = tvd_u[order]
        md_min_u = md_min_u[order]
        md_max_u = md_max_u[order]

        if tvd_u.size < 2 or not np.all(np.diff(tvd_u) > 0):
            # last resort: rounding strategy
            tvd_key = np.round(tvd, 2)
            g = pd.DataFrame({"tvd": tvd, "tvd_key": tvd_key, "md": md}).groupby("tvd_key", sort=False)
            tvd_u = g["tvd"].median().to_numpy(float)
            md_min_u = g["md"].min().to_numpy(float)
            md_max_u = g["md"].max().to_numpy(float)
            order = np.argsort(tvd_u)
            tvd_u = tvd_u[order]
            md_min_u = md_min_u[order]
            md_max_u = md_max_u[order]
            if tvd_u.size < 2 or not np.all(np.diff(tvd_u) > 0):
                raise ValueError("Failed to build strictly increasing TVD envelope from wellpath.")

    return tvd_u, md_min_u, md_max_u


def export_ref_tvd_csv(
    ref_dir: str | Path,
    *,
    wellpath_path: Optional[str | Path] = None,
    out_csv: Optional[str | Path] = None,
    include_md_xy: bool = True,
    ref_depth_axis: str = "tvd",
    md_xy_extrapolate_mode: str = "hold",
) -> Path:
    refp = Path(ref_dir)
    out_csv_path = Path(out_csv) if out_csv is not None else (refp / "ref_tvd.csv")
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    ref_depth_axis = str(ref_depth_axis).strip().lower()
    if ref_depth_axis not in {"tvd", "md"}:
        raise ValueError("--ref_depth_axis must be one of: tvd, md")

    md_xy_extrapolate_mode = str(md_xy_extrapolate_mode).strip().lower()
    if md_xy_extrapolate_mode not in {"hold", "nan"}:
        raise ValueError("--md_xy_extrapolate_mode must be one of: hold, nan")

    # --- 1) Load ref model on requested axis ---
    if ref_depth_axis == "tvd":
        if wellpath_path is None:
            raise ValueError("wellpath_path is required when ref_depth_axis='tvd'.")
        ref = load_ref_model(refp, depth_axis="tvd", wellpath_path=wellpath_path)
        depth_tvd = ref.z_m.astype(float)
        vp = ref.vp_mps.astype(float)
        vs = ref.vs_mps.astype(float)
        ref_meta = ref.meta
    else:
        depth_tvd = np.load(refp / "depth.npy").astype(float)
        vp = np.load(refp / "vp.npy").astype(float)
        vs = np.load(refp / "vs.npy").astype(float)
        ref_meta = {"depth_axis": "MD_or_Z_m (exported as tvd_m by request)", "ref_dir": str(refp)}

    if not (depth_tvd.ndim == vp.ndim == vs.ndim == 1 and depth_tvd.size == vp.size == vs.size):
        raise ValueError("depth/vp/vs must be 1D and same length.")
    if not np.all(np.isfinite(depth_tvd)) or not np.all(np.isfinite(vp)) or not np.all(np.isfinite(vs)):
        raise ValueError("depth/vp/vs contains NaN/Inf, cannot export.")
    if not np.all(np.diff(depth_tvd) > 0):
        idx = np.argsort(depth_tvd)
        depth_tvd, vp, vs = depth_tvd[idx], vp[idx], vs[idx]
        if not np.all(np.diff(depth_tvd) > 0):
            raise ValueError("tvd depth must be strictly increasing.")

    out = pd.DataFrame({"tvd_m": depth_tvd, "vp_mps": vp, "vs_mps": vs})

    wp_used = None
    md_ratio = 0.0
    envelope_added = False
    last_row_debug = {"tvd_last": float(depth_tvd[-1])}

    # --- 2) If wellpath exists: add md/x/y + md_min/md_max ---
    if wellpath_path is not None and ref_depth_axis == "tvd":
        wp = load_wellpath_full(wellpath_path)  # plateau-aware TVD->(MD,X,Y) single-valued mapping
        wp_used = wp.source_csv

        tvd0, tvd1 = float(wp.tvd_m[0]), float(wp.tvd_m[-1])

        # 2.1 md/x/y
        if include_md_xy:
            if md_xy_extrapolate_mode == "hold":
                md_interp = np.interp(depth_tvd, wp.tvd_m, wp.md_m, left=float(wp.md_m[0]), right=float(wp.md_m[-1]))
                x_interp = np.interp(depth_tvd, wp.tvd_m, wp.x_m, left=float(wp.x_m[0]), right=float(wp.x_m[-1]))
                y_interp = np.interp(depth_tvd, wp.tvd_m, wp.y_m, left=float(wp.y_m[0]), right=float(wp.y_m[-1]))
            else:
                inside = (depth_tvd >= tvd0) & (depth_tvd <= tvd1)
                md_interp = np.full_like(depth_tvd, np.nan, dtype=float)
                x_interp = np.full_like(depth_tvd, np.nan, dtype=float)
                y_interp = np.full_like(depth_tvd, np.nan, dtype=float)
                if inside.any():
                    md_interp[inside] = np.interp(depth_tvd[inside], wp.tvd_m, wp.md_m)
                    x_interp[inside] = np.interp(depth_tvd[inside], wp.tvd_m, wp.x_m)
                    y_interp[inside] = np.interp(depth_tvd[inside], wp.tvd_m, wp.y_m)

            out["md_m"] = md_interp
            out["x_m"] = x_interp
            out["y_m"] = y_interp
            md_ratio = float(np.isfinite(md_interp).mean())

        # 2.2 md_min/md_max envelope (RAW wellpath)
        # ✅ ONLY change here: set bin width to ref_tvd dz
        global _MD_ENVELOPE_BIN_W_M
        dz = float(np.nanmedian(np.diff(depth_tvd)))
        if not np.isfinite(dz) or dz <= 0:
            dz = 1.0
        _MD_ENVELOPE_BIN_W_M = dz

        tvd_u, md_min_u, md_max_u = _build_md_envelope_from_wellpath_csv(wp_used)

        if md_xy_extrapolate_mode == "hold":
            md_min_interp = np.interp(depth_tvd, tvd_u, md_min_u, left=float(md_min_u[0]), right=float(md_min_u[-1]))
            md_max_interp = np.interp(depth_tvd, tvd_u, md_max_u, left=float(md_max_u[0]), right=float(md_max_u[-1]))
        else:
            inside = (depth_tvd >= float(tvd_u[0])) & (depth_tvd <= float(tvd_u[-1]))
            md_min_interp = np.full_like(depth_tvd, np.nan, dtype=float)
            md_max_interp = np.full_like(depth_tvd, np.nan, dtype=float)
            if inside.any():
                md_min_interp[inside] = np.interp(depth_tvd[inside], tvd_u, md_min_u)
                md_max_interp[inside] = np.interp(depth_tvd[inside], tvd_u, md_max_u)

        out["md_min"] = md_min_interp
        out["md_max"] = md_max_interp
        envelope_added = True

        last_row_debug.update(
            {
                "wellpath_tvd_max": tvd1,
                "md_last": None if ("md_m" not in out.columns or not np.isfinite(out["md_m"].to_numpy(float)[-1])) else float(out["md_m"].to_numpy(float)[-1]),
                "x_last": None if ("x_m" not in out.columns or not np.isfinite(out["x_m"].to_numpy(float)[-1])) else float(out["x_m"].to_numpy(float)[-1]),
                "y_last": None if ("y_m" not in out.columns or not np.isfinite(out["y_m"].to_numpy(float)[-1])) else float(out["y_m"].to_numpy(float)[-1]),
                "md_min_last": None if not np.isfinite(out["md_min"].to_numpy(float)[-1]) else float(out["md_min"].to_numpy(float)[-1]),
                "md_max_last": None if not np.isfinite(out["md_max"].to_numpy(float)[-1]) else float(out["md_max"].to_numpy(float)[-1]),
                "md_envelope_bin_w_m": float(dz),
            }
        )

    out.to_csv(out_csv_path, index=False, float_format="%.6f")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ref_dir": str(refp),
        "out_csv": str(out_csv_path),
        "ref_depth_axis": ref_depth_axis,
        "ref_meta": ref_meta,
        "wellpath_used": wp_used,
        "md_xy_extrapolate_mode": md_xy_extrapolate_mode,
        "md_available_ratio": md_ratio,
        "md_envelope_added": bool(envelope_added),
        "n_rows": int(out.shape[0]),
        "tvd_minmax_m": [float(out["tvd_m"].min()), float(out["tvd_m"].max())],
        "last_row_debug": last_row_debug,
    }
    (out_csv_path.parent / "ref_tvd_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[OK] Exported ref_tvd.csv (added md_min/md_max when wellpath is available)")
    print(f"  csv : {out_csv_path.resolve()}")
    print(f"  json: {(out_csv_path.parent / 'ref_tvd_summary.json').resolve()}")
    return out_csv_path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_export_ref_tvd_csv",
        description="Export ref_tvd.csv (+ref_tvd_summary.json) from a ref model directory.",
    )
    p.add_argument("--ref_dir", required=True, help="Ref model dir containing depth.npy/vp.npy/vs.npy and velocity_model.json.")
    p.add_argument("--wellpath_path", default=None, help="Wellpath CSV or directory (required if --ref_depth_axis tvd).")
    p.add_argument("--out_csv", default=None, help="Output CSV path (default: <ref_dir>/ref_tvd.csv)")
    p.add_argument("--no_md_xy", action="store_true", help="Do not include md_m/x_m/y_m columns.")
    p.add_argument(
        "--ref_depth_axis",
        type=str,
        default="tvd",
        choices=["tvd", "md"],
        help="Interpretation of ref depth.npy. "
             "'tvd' means ref depth is MD and will be converted to TVD using wellpath_path (delivery default). "
             "'md' means ref depth is already vertical depth (legacy/debug).",
    )
    p.add_argument(
        "--md_xy_extrapolate_mode",
        type=str,
        default="hold",
        choices=["hold", "nan"],
        help="How to fill md_m/x_m/y_m and md_min/md_max outside wellpath TVD range. "
             "'hold' keeps endpoint values (recommended, fills last ceil(TVD_max) row). "
             "'nan' keeps NaN outside wellpath range (legacy).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    export_ref_tvd_csv(
        ref_dir=args.ref_dir,
        wellpath_path=args.wellpath_path,
        out_csv=args.out_csv,
        include_md_xy=(not args.no_md_xy),
        ref_depth_axis=args.ref_depth_axis,
        md_xy_extrapolate_mode=args.md_xy_extrapolate_mode,
    )


if __name__ == "__main__":
    main()