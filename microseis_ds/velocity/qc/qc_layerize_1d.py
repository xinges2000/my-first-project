# microseis_ds/velocity/qc/qc_layerize_1d.py
"""
QC for travel-time-equivalent layered 1D model produced by
microseis_ds/velocity/build/layerize_1d_by_traveltime.py.

Produces:
- deltaT_hist_P.png / deltaT_hist_S.png
- deltaT_vs_depth_P.png / deltaT_vs_depth_S.png
- deltaT_vs_RoverZ_P.png / deltaT_vs_RoverZ_S.png
- report_layerize_1d.json  (summary metrics + file paths)

Example (PowerShell):
python -m microseis_ds.velocity.qc.qc_layerize_1d `
  --ref_dir prepared_project/velocity/ref_engineering `
  --ref_depth_axis tvd `
  --wellpath_path prepared_project/wellpath `
  --layered_dir prepared_project/velocity/equiv_layered `
  --out_dir prepared_project/qc/velocity/qc_layerize_1d
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Headless-safe for CI / servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NOTE: after refactor, layerize lives in velocity/build/
from microseis_ds.velocity.build.layerize_1d_by_traveltime import (
    build_raypaths,
    load_engineering_1d,
    make_ray_set,
    make_ray_set_from_stations,
    make_ray_set_from_stations_wellpath,
    raytrace_1d_snell_precomputed,
    summarize_deltaT,
)

QC_NAME = "layerize_1d"
MODULE_NAME = "velocity"


def _to_posix(p: Path) -> str:
    return p.as_posix()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return _to_posix(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _asset_record(role: str, path: Path, *, required: bool) -> Dict[str, Any]:
    exists = path.exists()
    return {
        "role": role,
        "path": _to_posix(path),
        "required": bool(required),
        "exists": bool(exists),
        "status": "present" if exists else ("missing" if required else "not_found"),
    }


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def load_layered_1d(layered_dir: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    p = Path(layered_dir)
    meta = _read_json(p / "velocity_model_equiv_layered.json")
    files = meta.get("files", {})
    vp_name = files.get("vp", "vp.npy")
    vs_name = files.get("vs", "vs.npy")
    z = np.load(p / "depth.npy").astype(np.float64)
    vp = np.load(p / vp_name).astype(np.float64)
    vs = np.load(p / vs_name).astype(np.float64)
    return meta, z, vp, vs


def _eval_deltaT_for_wave(v_ref: np.ndarray, v_simp: np.ndarray, raypaths) -> Dict[tuple[float, float], float]:
    # reference times
    Tref = {}
    for rp in raypaths:
        t, _, _ = raytrace_1d_snell_precomputed(v_ref, rp)
        Tref[rp.key()] = float(t)
    # deltas
    delta = {}
    for rp in raypaths:
        t, _, _ = raytrace_1d_snell_precomputed(v_simp, rp)
        delta[rp.key()] = abs(float(t) - float(Tref[rp.key()])) * 1000.0
    return delta


def _plot_hist(delta: Dict[tuple[float, float], float], out_png: Path, title: str) -> None:
    arr = np.array(list(delta.values()), dtype=float)
    plt.figure()
    plt.hist(arr, bins=60)
    plt.xlabel("ΔT (ms)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_vs_depth(delta: Dict[tuple[float, float], float], out_png: Path, title: str) -> None:
    zs = np.array([k[0] for k in delta.keys()], dtype=float)
    dt = np.array(list(delta.values()), dtype=float)
    plt.figure()
    plt.scatter(zs, dt, s=10)
    plt.xlabel("Source depth z_s (m)")
    plt.ylabel("ΔT (ms)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_vs_roverz(delta: Dict[tuple[float, float], float], out_png: Path, title: str) -> None:
    rz = np.array([k[1] / max(k[0], 1e-6) for k in delta.keys()], dtype=float)
    dt = np.array(list(delta.values()), dtype=float)
    plt.figure()
    plt.scatter(rz, dt, s=10)
    plt.xlabel("R / z_s")
    plt.ylabel("ΔT (ms)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _try_infer_wellpath_from_equiv_qc(layered_dir: str) -> Optional[str]:
    """
    After refactor, layerize writes:
      qc_traveltime_equiv.json
        equivalence.geometry.wellpath.wellpath_csv  (dict)
    Older variants may store:
      geometry.wellpath.path
    """
    qcp = Path(layered_dir) / "qc_traveltime_equiv.json"
    if not qcp.exists():
        return None
    try:
        qj = _read_json(qcp)
        geom = (qj.get("equivalence", {}) or {}).get("geometry", {}) or {}

        # New structure: geometry.wellpath.wellpath_csv
        wp = geom.get("wellpath")
        if isinstance(wp, dict):
            if wp.get("wellpath_csv"):
                return str(wp["wellpath_csv"])
            if wp.get("path"):
                return str(wp["path"])

        # Fallbacks (defensive)
        if isinstance(geom.get("wellpath_path"), str) and geom["wellpath_path"].strip():
            return str(geom["wellpath_path"])

        return None
    except Exception:
        return None


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qc_layerize_1d",
        description="QC plots + report for travel-time-equivalent layered 1D model",
    )
    p.add_argument("--ref_dir", required=True, help="Reference engineering model dir")
    p.add_argument("--layered_dir", required=True, help="Layered model dir (output of layerize_1d_by_traveltime)")
    p.add_argument("--out_dir", required=True, help="QC output dir")
    p.add_argument(
        "--ref_depth_axis",
        type=str,
        default="tvd",
        choices=["tvd", "md"],
        help=(
            "Depth axis for reference model. "
            "Use tvd to convert MD-based engineering model to TVD using wellpath (recommended)."
        ),
    )
    p.add_argument(
        "--wellpath_path",
        type=str,
        default=None,
        help=(
            "Wellpath CSV or directory. Required when ref_depth_axis=tvd. "
            "If omitted, will try to infer from layered_dir/qc_traveltime_equiv.json."
        ),
    )
    p.add_argument(
        "--report_path",
        type=str,
        default=None,
        help=(
            "Optional Step 7.3 aligned JSON report path. "
            "If omitted, keep legacy output: <out_dir>/report_layerize_1d.json."
        ),
    )
    p.add_argument(
        "--figures_dir",
        type=str,
        default=None,
        help=(
            "Optional Step 7.3 aligned figures directory. "
            "If omitted, keep legacy output: <out_dir>/*.png."
        ),
    )
    return p


def _align_ref_grid_to_layered_if_needed(
    z_ref: np.ndarray,
    vp_ref: np.ndarray,
    vs_ref: np.ndarray,
    z_lay: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fix ONLY the runtime mismatch reported by:
      ValueError: Reference and layered depth grids must match exactly (depth.npy).

    Policy:
    - If z_lay is a strict extension of z_ref (prefix matches exactly within tolerance),
      then extend ref (z/vp/vs) to z_lay using endpoint-hold extrapolation.
    - Otherwise, keep strict behavior and raise.
    """
    if z_ref.shape == z_lay.shape and np.allclose(z_ref, z_lay):
        return z_ref, vp_ref, vs_ref

    # Case: layered deeper than reference, and ref is a prefix of layered
    if z_lay.size > z_ref.size:
        prefix = z_lay[: z_ref.size]
        if np.allclose(prefix, z_ref):
            n_ext = z_lay.size - z_ref.size
            vp_last = float(vp_ref[-1])
            vs_last = float(vs_ref[-1])
            vp_ref2 = np.concatenate([vp_ref, np.full(n_ext, vp_last, dtype=np.float64)], axis=0)
            vs_ref2 = np.concatenate([vs_ref, np.full(n_ext, vs_last, dtype=np.float64)], axis=0)
            return z_lay, vp_ref2, vs_ref2

    # (Optional defensive) If reference deeper than layered and layered is prefix of reference,
    # we could trim ref, but that would change the QC meaning; keep strict.
    raise ValueError("Reference and layered depth grids must match exactly (depth.npy).")


def main(argv: List[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    outp = Path(args.out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    figures_dir = Path(args.figures_dir) if args.figures_dir else outp
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path) if args.report_path else (outp / "report_layerize_1d.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine wellpath used for TVD conversion (if needed)
    wp_path = args.wellpath_path
    if wp_path is None:
        wp_path = _try_infer_wellpath_from_equiv_qc(args.layered_dir)

    if args.ref_depth_axis == "tvd" and wp_path is None:
        raise ValueError(
            "ref_depth_axis=tvd requires --wellpath_path, "
            "or layered_dir/qc_traveltime_equiv.json must contain equivalence.geometry.wellpath.wellpath_csv."
        )

    # Load ref + layered
    _, z_ref, vp_ref, vs_ref = load_engineering_1d(
        args.ref_dir,
        depth_axis=args.ref_depth_axis,
        wellpath_path=wp_path,
        export_csv=False,
    )
    _, z_lay, vp_lay, vs_lay = load_layered_1d(args.layered_dir)

    # ---- FIX HERE: allow layered grid to extend ref grid (z_max increased) by extending ref with endpoint-hold ----
    z_ref, vp_ref, vs_ref = _align_ref_grid_to_layered_if_needed(z_ref, vp_ref, vs_ref, z_lay)
    # -----------------------------------------------------------------------------------------------------------

    # geometry from qc json (preferred), fallback to defaults
    qc_path = Path(args.layered_dir) / "qc_traveltime_equiv.json"
    if qc_path.exists():
        qc = _read_json(qc_path)
        geom = (qc.get("equivalence", {}) or {}).get("geometry", {}) or {}
        mode = str(geom.get("mode", "ratios"))
        z_min = float(geom.get("z_min_m", float(z_ref[0])))
        z_max = float(geom.get("z_max_m", float(z_ref[-1])))
        n_z = int(geom.get("n_z", 81))
        r_over_z = geom.get("r_over_z", [0.5, 0.8, 1.0, 1.2, 1.5])
    else:
        geom = {}
        mode = "ratios"
        z_min, z_max, n_z, r_over_z = float(z_ref[0]), float(z_ref[-1]), 81, [0.5, 0.8, 1.0, 1.2, 1.5]

    # Normalize mode names for backward/forward compatibility
    # - old: "stations_wellpath"
    # - new: "stations_wellpath_tvd"
    mode_norm = mode.strip().lower()
    if mode_norm == "stations_wellpath":
        mode_norm = "stations_wellpath_tvd"

    # Build ray set matching the layerize geometry
    if mode_norm == "stations_wellpath_tvd" and qc_path.exists():
        stations_csv = str(geom.get("stations_csv", ""))
        wp = geom.get("wellpath", {})
        wp_csv = ""
        if isinstance(wp, dict):
            wp_csv = str(wp.get("wellpath_csv", ""))

        # fallback: user-supplied wellpath_path (for reproducibility)
        wp_csv = wp_csv or (str(wp_path) if wp_path is not None else "")

        if not stations_csv:
            raise ValueError("Geometry mode stations_wellpath_tvd requires geometry.stations_csv in qc_traveltime_equiv.json")
        if not wp_csv:
            raise ValueError("Geometry mode stations_wellpath_tvd requires geometry.wellpath.wellpath_csv (or provide --wellpath_path).")

        rays, _ = make_ray_set_from_stations_wellpath(
            stations_csv=stations_csv,
            wellpath_path=wp_csv,
            z_min=z_min,
            z_max=z_max,
            n_z=n_z,
            max_stations=int(geom.get("max_stations", 500)),
            seed=int(geom.get("seed", 0)),
        )
    elif mode_norm == "stations" and qc_path.exists():
        stations_csv = str(geom.get("stations_csv", ""))
        if not stations_csv:
            raise ValueError("Geometry mode stations requires geometry.stations_csv in qc_traveltime_equiv.json")

        rays, _ = make_ray_set_from_stations(
            stations_csv=stations_csv,
            src_x_m=float(geom.get("src_x_m", 0.0)),
            src_y_m=float(geom.get("src_y_m", 0.0)),
            z_min=z_min,
            z_max=z_max,
            n_z=n_z,
            max_stations=int(geom.get("max_stations", 500)),
            seed=int(geom.get("seed", 0)),
        )
    else:
        rays = make_ray_set(z_min, z_max, n_z, r_over_z)

    raypaths = build_raypaths(z_ref, rays)

    # Evaluate deltaT
    deltaP = _eval_deltaT_for_wave(vp_ref, vp_lay, raypaths)
    deltaS = _eval_deltaT_for_wave(vs_ref, vs_lay, raypaths)

    mP = summarize_deltaT(deltaP)
    mS = summarize_deltaT(deltaS)

    # plots
    _plot_hist(deltaP, figures_dir / "deltaT_hist_P.png", "ΔT histogram (P)")
    _plot_hist(deltaS, figures_dir / "deltaT_hist_S.png", "ΔT histogram (S)")

    _plot_vs_depth(deltaP, figures_dir / "deltaT_vs_depth_P.png", "ΔT vs source depth (P)")
    _plot_vs_depth(deltaS, figures_dir / "deltaT_vs_depth_S.png", "ΔT vs source depth (S)")

    _plot_vs_roverz(deltaP, figures_dir / "deltaT_vs_RoverZ_P.png", "ΔT vs R/z (P)")
    _plot_vs_roverz(deltaS, figures_dir / "deltaT_vs_RoverZ_S.png", "ΔT vs R/z (S)")

    figure_outputs = {
        "deltaT_hist_P": str((figures_dir / "deltaT_hist_P.png").as_posix()),
        "deltaT_hist_S": str((figures_dir / "deltaT_hist_S.png").as_posix()),
        "deltaT_vs_depth_P": str((figures_dir / "deltaT_vs_depth_P.png").as_posix()),
        "deltaT_vs_depth_S": str((figures_dir / "deltaT_vs_depth_S.png").as_posix()),
        "deltaT_vs_RoverZ_P": str((figures_dir / "deltaT_vs_RoverZ_P.png").as_posix()),
        "deltaT_vs_RoverZ_S": str((figures_dir / "deltaT_vs_RoverZ_S.png").as_posix()),
    }
    warnings: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    checked_assets = [
        _asset_record("ref_engineering", Path(args.ref_dir), required=True),
        _asset_record("equiv_layered", Path(args.layered_dir), required=True),
        _asset_record("equiv_layered_model_json", Path(args.layered_dir) / "velocity_model_equiv_layered.json", required=True),
        _asset_record("equiv_layered_qc_geometry", qc_path, required=False),
    ]
    figure_count = sum(1 for item in figure_outputs.values() if Path(item).is_file())
    status = "failed" if errors else ("warning" if warnings else "passed")

    report = {
        "schema_version": "velocity_qc_layerize_1d_v1",
        "module": MODULE_NAME,
        "qc_name": QC_NAME,
        "status": status,
        "ref_dir": args.ref_dir,
        "layered_dir": args.layered_dir,
        "ref_depth_axis": args.ref_depth_axis,
        "wellpath_path_used": wp_path,
        "checked_assets": checked_assets,
        "warnings": warnings,
        "errors": errors,
        "results": {
            "P_max_ms": float(mP.get("max_ms", 0.0)),
            "P_p95_ms": float(mP.get("p95_ms", 0.0)),
            "P_rms_ms": float(mP.get("rms_ms", 0.0)),
            "S_max_ms": float(mS.get("max_ms", 0.0)),
            "S_p95_ms": float(mS.get("p95_ms", 0.0)),
            "S_rms_ms": float(mS.get("rms_ms", 0.0)),
            "figure_count": int(figure_count),
        },
        "geometry": {
            "mode": mode,
            "mode_normalized": mode_norm,
            "z_min_m": z_min,
            "z_max_m": z_max,
            "n_z": n_z,
            "r_over_z": r_over_z,
            "stations_csv": (str(geom.get("stations_csv")) if qc_path.exists() else None),
            "src_x_m": (float(geom.get("src_x_m", 0.0)) if qc_path.exists() else None),
            "src_y_m": (float(geom.get("src_y_m", 0.0)) if qc_path.exists() else None),
            "max_stations": (int(geom.get("max_stations", 500)) if qc_path.exists() else None),
            "seed": (int(geom.get("seed", 0)) if qc_path.exists() else None),
            "wellpath_csv": (
                str((geom.get("wellpath") or {}).get("wellpath_csv"))
                if (qc_path.exists() and isinstance(geom.get("wellpath"), dict))
                else None
            ),
        },
        "metrics": {"P": mP, "S": mS},
        "outputs": figure_outputs,
        "report_path": str(report_path.as_posix()),
        "figures_dir": str(figures_dir.as_posix()),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


    print("[OK] QC finished")
    print(f"  out_dir     : {outp.resolve()}")
    print(f"  figures_dir : {figures_dir.resolve()}")
    print(f"  report_path : {report_path.resolve()}")
    print(f"  P: max={mP['max_ms']:.3f}ms p95={mP['p95_ms']:.3f}ms rms={mP['rms_ms']:.3f}ms")
    print(f"  S: max={mS['max_ms']:.3f}ms p95={mS['p95_ms']:.3f}ms rms={mS['rms_ms']:.3f}ms")


if __name__ == "__main__":
    main()
