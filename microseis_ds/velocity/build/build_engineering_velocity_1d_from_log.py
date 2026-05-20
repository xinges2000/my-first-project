# microseis_ds/velocity/build/build_engineering_velocity_1d_from_log.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def _norm_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _resolve_column(df: pd.DataFrame, wanted: str | None, aliases: list[str], role_name: str) -> str:
    cols = list(df.columns)
    norm_map = {_norm_col(c): c for c in cols}

    def pick_by_norm_key(key_norm: str) -> str | None:
        return norm_map.get(key_norm, None)

    if wanted is not None:
        if wanted in cols:
            return wanted
        wn = _norm_col(wanted)
        hit = pick_by_norm_key(wn)
        if hit is not None:
            return hit

    for a in aliases:
        hit = pick_by_norm_key(_norm_col(a))
        if hit is not None:
            return hit

    raise ValueError(
        f"{role_name} column not found. wanted={wanted!r}, aliases={aliases}. "
        f"available columns={cols}"
    )


def _median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        raise ValueError("median_k must be odd (e.g., 3,5,7,9).")
    n = x.size
    if n < k:
        return x
    r = k // 2
    out = x.copy()
    for i in range(r, n - r):
        out[i] = np.median(x[i - r:i + r + 1])
    return out


def _clean_depth_velocity(
    depth: np.ndarray,
    vp: np.ndarray,
    vs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate formal ref_engineering input arrays.

    Step 3 contract:
    - depth/vp/vs must all be present.
    - depth must be strictly increasing in the original input order.
    - duplicate depth is invalid because strict increase does not allow equality.
    - vp and vs must be strictly positive.
    - invalid rows must not be silently filtered, sorted, deduplicated, or repaired.
    """
    depth = np.asarray(depth, dtype=float)
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)

    if depth.shape != vp.shape or depth.shape != vs.shape:
        raise ValueError("depth, vp, and vs must have the same length.")

    if depth.size < 2:
        raise ValueError("Input must contain at least 2 valid samples.")

    if not np.all(np.isfinite(depth)):
        raise ValueError("depth must be finite.")
    if not np.all(np.isfinite(vp)):
        raise ValueError("vp must be finite.")
    if not np.all(np.isfinite(vs)):
        raise ValueError("vs must be finite.")

    if not np.all(np.diff(depth) > 0):
        raise ValueError("depth must be strictly increasing.")

    if np.any(vp <= 0):
        raise ValueError("vp must be > 0.")
    if np.any(vs <= 0):
        raise ValueError("vs must be > 0.")

    return depth, vp, vs


def _resample_to_uniform_grid(
    depth_raw: np.ndarray,
    vp_raw: np.ndarray,
    vs_raw: np.ndarray,
    z0: float,
    zmax: float,
    dz: float,
    extrapolate: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dz <= 0:
        raise ValueError("dz must be > 0")
    if zmax <= z0:
        raise ValueError("zmax must be > z0")

    z_grid = np.arange(z0, zmax + 1e-9, dz, dtype=np.float64)

    zmin_raw = float(depth_raw[0])
    zmax_raw = float(depth_raw[-1])
    zq = np.clip(z_grid, zmin_raw, zmax_raw) if not extrapolate else z_grid

    vp = np.interp(zq, depth_raw, vp_raw).astype(np.float64)
    vs = np.interp(zq, depth_raw, vs_raw).astype(np.float64)

    return z_grid, vp, vs


def build_engineering_velocity_1d_from_log(
    input_csv: str,
    out_dir: str,
    dz: float = 1.0,
    z0: float | None = None,
    zmax: float | None = None,
    vp_vs_ratio: float = 1.732,
    depth_col: str | None = None,
    vp_col: str | None = None,
    vs_col: str | None = None,
    delimiter: str | None = None,
    z_positive: str = "down",
    dtype: str = "float32",
    export_csv: bool = False,
    export_csv_name: str = "velocity_1d_engineering.csv",
    smooth_median_k: int = 0,
    extrapolate: bool = False,
) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv) if delimiter is None else pd.read_csv(input_csv, sep=delimiter)
    if df.shape[1] < 2:
        raise ValueError(f"Input must have at least 2 columns (depth, vp). Got {df.shape[1]}.")

    depth_aliases = ["depth", "depth_m", "md", "measureddepth", "measured_depth", "z", "z_m"]
    vp_aliases = ["vp", "vp_mps", "p", "pvel", "pvelocity", "p_wave_velocity", "pwave", "vpp"]
    vs_aliases = ["vs", "vs_mps", "s", "svel", "svelocity", "s_wave_velocity", "swave", "vss"]

    depth_col_resolved = _resolve_column(df, depth_col, depth_aliases, role_name="depth")
    vp_col_resolved = _resolve_column(df, vp_col, vp_aliases, role_name="vp")

    if vs_col is not None:
        vs_col_resolved = _resolve_column(df, vs_col, vs_aliases, role_name="vs")
    else:
        vs_col_resolved = _resolve_column(df, None, vs_aliases, role_name="vs")

    depth_raw = df[depth_col_resolved].to_numpy(dtype=float)
    vp_raw = df[vp_col_resolved].to_numpy(dtype=float)
    vs_raw = df[vs_col_resolved].to_numpy(dtype=float)

    depth_raw, vp_raw, vs_raw = _clean_depth_velocity(depth_raw, vp_raw, vs_raw)

    vs_rule = f"from column: {vs_col_resolved}"

    if smooth_median_k and smooth_median_k >= 3:
        vp_raw = _median_filter_1d(vp_raw, smooth_median_k)
        vs_raw = _median_filter_1d(vs_raw, smooth_median_k)

    z0_eff = float(depth_raw[0]) if z0 is None else float(z0)
    zmax_eff = float(depth_raw[-1]) if zmax is None else float(zmax)
    if zmax_eff <= z0_eff:
        raise ValueError("zmax must be greater than z0")

    z_grid, vp_grid, vs_grid = _resample_to_uniform_grid(
        depth_raw=depth_raw,
        vp_raw=vp_raw,
        vs_raw=vs_raw,
        z0=z0_eff,
        zmax=zmax_eff,
        dz=float(dz),
        extrapolate=bool(extrapolate),
    )

    if not np.all(np.diff(z_grid) > 0):
        raise ValueError("Engineering depth grid must be strictly increasing.")
    if np.any(vp_grid <= 0) or np.any(vs_grid <= 0):
        raise ValueError("Engineering Vp/Vs must be > 0.")

    np_dtype = np.float32 if dtype == "float32" else np.float64

    np.save(outp / "depth.npy", z_grid.astype(np_dtype))
    np.save(outp / "vp.npy", vp_grid.astype(np_dtype))
    np.save(outp / "vs.npy", vs_grid.astype(np_dtype))

    if export_csv:
        df_out = pd.DataFrame({"depth_m": z_grid.astype(float), "vp_mps": vp_grid.astype(float), "vs_mps": vs_grid.astype(float)})
        df_out.to_csv(outp / export_csv_name, index=False, float_format="%.6f")

    velocity_model = {
        "version": "1.0.0",
        "model_type": "1d_continuous_engineering",
        "units": {"length": "m", "velocity": "m/s"},
        "coordinate_system": {
            "name": "Engineering 1D depth profile (uniform grid)",
            "epsg": None,
            "axis_convention": {"x": "N/A", "y": "N/A", "z_positive": z_positive},
        },
        "grid": {
            "origin": {"x0": 0.0, "y0": 0.0, "z0": float(z_grid[0])},
            "spacing": {"dx": 0.0, "dy": 0.0, "dz": float(dz)},
            "shape": {"nx": 1, "ny": 1, "nz": int(len(z_grid))},
        },
        "fields": {"vp": True, "vs": True, "rho": False},
        "files": {"format": "npy", "dtype": dtype, "endian": "little", "vp": "vp.npy", "vs": "vs.npy", "rho": None},
        "qc": {"dz_nonuniform_ratio_max": 1.01, "dz_mean_min_m": 0.05, "dz_mean_max_m": 50.0},
        "provenance": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_file": str(input_csv),
            "depth_col_used": depth_col_resolved,
            "vp_col_used": vp_col_resolved,
            "vs_source": vs_rule,
            "raw_depth_range_m": [float(depth_raw[0]), float(depth_raw[-1])],
            "engineering_grid": {"z0": z0_eff, "zmax": zmax_eff, "dz": float(dz), "extrapolate": bool(extrapolate)},
            "smoothing": {"median_k": int(smooth_median_k)} if smooth_median_k else None,
            "export_csv": bool(export_csv),
            "export_csv_name": export_csv_name if export_csv else None,
        },
    }

    (outp / "velocity_model.json").write_text(json.dumps(velocity_model, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Engineering continuous 1D velocity model generated (Scheme A)")
    print(f"  input    : {input_csv}")
    print(f"  out_dir  : {outp.resolve()}")
    print(f"  depth_col_used : {depth_col_resolved}")
    print(f"  vp_col_used    : {vp_col_resolved}")
    print(f"  vs             : {vs_rule}")
    print(f"  grid    : z0={z0_eff}, zmax={zmax_eff}, dz={dz}, nz={len(z_grid)}")
    if smooth_median_k and smooth_median_k >= 3:
        print(f"  smooth  : median_k={smooth_median_k}")
    print(f"  vp range : {float(vp_grid.min()):.3f} ~ {float(vp_grid.max()):.3f} m/s")
    print(f"  vs range : {float(vs_grid.min()):.3f} ~ {float(vs_grid.max()):.3f} m/s")
    if export_csv:
        print(f"  csv      : {(outp / export_csv_name).resolve()}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_engineering_velocity_1d_from_log",
        description="Build engineering continuous 1D model from well log CSV. Optionally run travel-time-equivalent layerization.",
    )
    p.add_argument("--input", "-i", required=True, help="Input CSV path with required depth, vp, and vs columns.")
    p.add_argument("--out_dir", "-o", required=True, help="Output directory for engineering ref model.")
    p.add_argument("--dz", type=float, default=1.0, help="Engineering grid spacing dz (m). Default 1m.")
    p.add_argument("--z0", type=float, default=None, help="Engineering grid start depth (m). Default: raw min depth.")
    p.add_argument("--zmax", type=float, default=None, help="Engineering grid max depth (m). Default: raw max depth.")
    p.add_argument("--vp_vs_ratio", type=float, default=1.732, help="Deprecated compatibility argument; Vs is required and is not computed from Vp.")
    p.add_argument("--depth_col", default=None, help="Depth column name (robust match). Default: auto")
    p.add_argument("--vp_col", default=None, help="Vp column name (robust match). Default: auto")
    p.add_argument("--vs_col", default=None, help="Vs column name. If omitted, auto-detect by aliases; failure is an error.")
    p.add_argument("--delimiter", default=None, help="CSV delimiter. Default: auto")
    p.add_argument("--z_positive", default="down", choices=["down", "up"], help="Z positive direction in model")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="Saved array dtype")
    p.add_argument("--smooth_median_k", type=int, default=0, help="Optional median filter window (odd >=3). Default 0.")
    p.add_argument("--extrapolate", action="store_true", help="Allow extrapolation beyond raw depth range (default clips).")
    p.add_argument("--export_csv", action="store_true", help="Also export a human-readable CSV (depth/vp/vs).")
    p.add_argument("--export_csv_name", default="velocity_1d_engineering.csv", help="Export CSV file name within out_dir.")

    p.add_argument("--equiv_out_dir", default=None, help="If set, run ΔT-driven layerization and write layered model to this directory.")
    p.add_argument("--ref_depth_axis", type=str, default="tvd", choices=["tvd", "md"], help="Depth axis for ref model when running layerize.")
    p.add_argument("--stations_csv", type=str, default=None, help="stations.csv path (required for realistic geometry mode).")
    p.add_argument("--wellpath_path", type=str, default=None, help="wellpath CSV or directory (required for ref_depth_axis=tvd).")
    p.add_argument("--z_min", type=float, default=None, help="Source depth sampling min (m). Default: z0")
    p.add_argument("--z_max", type=float, default=None, help="Source depth sampling max (m). Default: zmax")
    p.add_argument("--n_z", type=int, default=81, help="Number of source depth samples along well. Default 81")
    p.add_argument("--max_stations", type=int, default=500, help="Max stations used for geometry. Default 500")
    p.add_argument("--tau_p_ms", type=float, default=2.0, help="Threshold max ΔT for P (ms). Default 2.0")
    p.add_argument("--tau_s_ms", type=float, default=4.0, help="Threshold max ΔT for S (ms). Default 4.0")
    p.add_argument("--n_init_layers", type=int, default=5, help="Initial number of layers. Default 5")
    p.add_argument("--max_layers", type=int, default=80, help="Maximum layers allowed. Default 80")
    p.add_argument("--min_thickness_m", type=float, default=10.0, help="Minimum layer thickness (m). Default 10m")
    p.add_argument("--top_frac", type=float, default=0.05, help="Top fraction worst rays used to guide splitting. Default 0.05")
    p.add_argument("--min_top", type=int, default=50, help="Minimum worst rays used to guide splitting. Default 50")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--quiet", action="store_true", help="Reduce console output during layerize.")
    p.add_argument("--export_ref_tvd_csv", action="store_true", help="Export <out_dir>/ref_tvd.csv (+summary json) for delivery/QC.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    build_engineering_velocity_1d_from_log(
        input_csv=args.input,
        out_dir=args.out_dir,
        dz=args.dz,
        z0=args.z0,
        zmax=args.zmax,
        vp_vs_ratio=args.vp_vs_ratio,
        depth_col=args.depth_col,
        vp_col=args.vp_col,
        vs_col=args.vs_col,
        delimiter=args.delimiter,
        z_positive=args.z_positive,
        dtype=args.dtype,
        export_csv=args.export_csv,
        export_csv_name=args.export_csv_name,
        smooth_median_k=args.smooth_median_k,
        extrapolate=args.extrapolate,
    )

    if args.export_ref_tvd_csv:
        if not args.wellpath_path:
            raise ValueError("--export_ref_tvd_csv requires --wellpath_path.")
        from microseis_ds.velocity.build.build_export_ref_tvd_csv import export_ref_tvd_csv
        export_ref_tvd_csv(ref_dir=str(args.out_dir), wellpath_path=str(args.wellpath_path), out_csv=None, include_md_xy=True)

    if args.equiv_out_dir:
        if not args.stations_csv:
            raise ValueError("--equiv_out_dir is set, but --stations_csv is missing.")
        if args.ref_depth_axis == "tvd" and not args.wellpath_path:
            raise ValueError("--ref_depth_axis tvd requires --wellpath_path.")

        from microseis_ds.velocity.build import layerize_1d_by_traveltime as layerize

        layerize_args = [
            "--ref_dir", str(args.out_dir),
            "--out_dir", str(args.equiv_out_dir),
            "--ref_depth_axis", str(args.ref_depth_axis),
            "--stations_csv", str(args.stations_csv),
            "--max_stations", str(args.max_stations),
            "--tau_p_ms", str(args.tau_p_ms),
            "--tau_s_ms", str(args.tau_s_ms),
            "--n_init_layers", str(args.n_init_layers),
            "--max_layers", str(args.max_layers),
            "--min_thickness_m", str(args.min_thickness_m),
            "--top_frac", str(args.top_frac),
            "--min_top", str(args.min_top),
            "--seed", str(args.seed),
            "--n_z", str(args.n_z),
        ]
        if args.wellpath_path:
            layerize_args += ["--wellpath_path", str(args.wellpath_path)]
        if args.z_min is not None:
            layerize_args += ["--z_min", str(args.z_min)]
        if args.z_max is not None:
            layerize_args += ["--z_max", str(args.z_max)]
        if args.quiet:
            layerize_args += ["--quiet"]

        print("[RUN] Running travel-time-equivalent layerization (ΔT-driven)")
        print(f"  ref_dir       : {args.out_dir}")
        print(f"  equiv_out_dir : {args.equiv_out_dir}")
        layerize.main(layerize_args)


if __name__ == "__main__":
    main()