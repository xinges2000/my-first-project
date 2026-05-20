# scripts/build_velocity.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# Build entrypoints (module-internal)
from microseis_ds.velocity.build.build_engineering_velocity_1d_from_log import (
    main as _main_build_engineering,
)
from microseis_ds.velocity.build.layerize_1d_by_traveltime import (
    main as _main_layerize,
)
from microseis_ds.velocity.build.build_export_ref_tvd_csv import (
    main as _main_export_ref_tvd,
)
from microseis_ds.velocity.build.package_velocity_model_delivery import (
    main as _main_package_delivery,
)
from microseis_ds.velocity.build.build_velocity_manifest import (
    build_velocity_manifest as _build_velocity_manifest,
)
from microseis_ds.velocity.build.build_traveltime_ready import (
    main as _main_build_traveltime_ready,
)


def _p(p: Optional[str]) -> Optional[str]:
    return None if p is None else str(Path(p))


def _default_dir(out_root: str, module_name: str, name: str) -> str:
    return str(Path(out_root) / module_name / name)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_velocity",
        description=(
            "Velocity module build entry (1 script). "
            "Tasks: engineering, export_ref_tvd, layerize, traveltime_ready, package, all."
        ),
    )

    # Common
    p.add_argument("--module_name", default="velocity", help="Module name under prepared_project/. Default: velocity")
    p.add_argument("--out_root", default="prepared_project", help="Root output directory. Default: prepared_project")
    p.add_argument(
        "--task",
        choices=["engineering", "export_ref_tvd", "layerize", "traveltime_ready", "package", "all"],
        default="all",
        help="Which build task to run. Default: all",
    )

    # Input log (build engineering/ref model)
    p.add_argument("--log_csv", default=None, help="Well log CSV for building engineering ref model.")
    p.add_argument("--depth_col", default=None, help="Depth column name in log_csv (optional).")
    p.add_argument("--vp_col", default=None, help="Vp column name in log_csv (optional).")
    p.add_argument("--vs_col", default=None, help="Vs column name. If omitted, auto-detect by aliases; failure is an error.")
    p.add_argument("--vp_vs_ratio", type=float, default=1.732, help="Deprecated compatibility argument; Vs is required and is not computed from Vp.")
    p.add_argument("--dz", type=float, default=1.0, help="Engineering grid dz (m). Default 1.0")
    p.add_argument("--z0", type=float, default=None, help="Engineering grid start depth (m). Default: from log")
    p.add_argument("--zmax", type=float, default=None, help="Engineering grid max depth (m). Default: from log")
    p.add_argument("--smooth_median_k", type=int, default=0, help="Median smoothing window (odd>=3). Default 0")
    p.add_argument("--extrapolate", action="store_true", help="Allow extrapolation beyond log depth range (default clips).")
    p.add_argument("--export_profile_csv", action="store_true", help="Export readable CSV from engineering model.")
    p.add_argument("--profile_csv_name", default="velocity_1d_engineering.csv", help="CSV name under ref_dir.")

    # Wellpath/stations for TVD conversion & geometry-based layerize
    p.add_argument("--wellpath_path", default=None, help="Wellpath CSV or directory (required for ref_depth_axis=tvd).")
    p.add_argument("--stations_csv", default=None, help="stations.csv (recommended for realistic geometry in layerize).")
    p.add_argument(
        "--baseline_root",
        default=None,
        help="Baseline handoff root used only for velocity manifest provenance/hash evidence. Default: <out_root>/baseline if it exists.",
    )
    p.add_argument(
        "--z_positive",
        default="down",
        help="Depth/Z positive direction recorded in baseline_handoff evidence. Default: down.",
    )

    # Layerize settings
    p.add_argument("--ref_depth_axis", choices=["tvd", "md"], default="tvd", help="Ref depth axis for layerize. Default tvd.")
    p.add_argument("--z_min", type=float, default=None, help="Source depth sampling min (m). Default: ref z0")
    p.add_argument("--z_max", type=float, default=None, help="Source depth sampling max (m). Default: ref zmax")
    p.add_argument("--n_z", type=int, default=81, help="Number of source depth samples. Default 81")
    p.add_argument("--max_stations", type=int, default=500, help="Max stations used in geometry. Default 500")
    p.add_argument("--tau_p_ms", type=float, default=2.0, help="Target max ΔT for P (ms). Default 2.0")
    p.add_argument("--tau_s_ms", type=float, default=4.0, help="Target max ΔT for S (ms). Default 4.0")
    p.add_argument("--n_init_layers", type=int, default=5, help="Initial layers. Default 5")
    p.add_argument("--max_layers", type=int, default=80, help="Maximum layers allowed. Default 80")
    p.add_argument("--min_thickness_m", type=float, default=10.0, help="Minimum layer thickness (m). Default 10")
    p.add_argument("--top_frac", type=float, default=0.05, help="Top fraction of worst rays to guide split. Default 0.05")
    p.add_argument("--min_top", type=int, default=50, help="Minimum number of worst rays. Default 50")
    p.add_argument("--seed", type=int, default=0, help="Random seed. Default 0")
    p.add_argument("--quiet", action="store_true", help="Reduce console output for layerize.")

    # Export ref_tvd.csv settings
    p.add_argument("--ref_csv_out", default=None, help="Export path for ref_tvd.csv (default: <ref_dir>/ref_tvd.csv)")
    p.add_argument("--no_md_xy", action="store_true", help="Export ref_tvd.csv without md/x/y columns.")

    # Packaging settings
    p.add_argument("--delivery_out_dir", default=None, help="Output directory for delivery bundle (default: prepared_project/velocity_delivery)")
    p.add_argument("--zip", action="store_true", help="Also create zip for delivery bundle.")

    # Optional override output dirs
    p.add_argument("--ref_dir", default=None, help="Override ref engineering model dir.")
    p.add_argument("--layered_dir", default=None, help="Override equiv layered model dir.")
    p.add_argument(
        "--traveltime_ready_dir",
        default=None,
        help="Override traveltime_ready output dir (default: <out_root>/<module_name>/traveltime_ready).",
    )
    p.add_argument("--qc_dir", default=None, help="Optional QC dir to include in delivery bundle (png/json).")

    return p


def _build_manifest_source_inputs(
    *,
    args: argparse.Namespace,
    ref_dir: str,
    layered_dir: str,
    traveltime_ready_dir: str,
    delivery_out_dir: str,
) -> Dict[str, Any]:
    """Build lightweight provenance for manifest/summary without changing build semantics."""
    return {
        "task": str(args.task),
        "log_csv": _p(args.log_csv),
        "stations_csv": _p(args.stations_csv),
        "wellpath_path": _p(args.wellpath_path),
        "baseline_root": _p(args.baseline_root) if args.baseline_root else _p(Path(args.out_root) / "baseline"),
        "z_positive": str(args.z_positive),
        "ref_depth_axis": str(args.ref_depth_axis),
        "out_root": str(Path(args.out_root)),
        "module_name": str(args.module_name),
        "ref_dir": _p(ref_dir),
        "layered_dir": _p(layered_dir),
        "traveltime_ready_dir": _p(traveltime_ready_dir),
        "delivery_out_dir": _p(delivery_out_dir),
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    out_root = str(Path(args.out_root))
    module_name = str(args.module_name)

    # Default outputs under prepared_project/<module_name>/
    ref_dir = args.ref_dir or _default_dir(out_root, module_name, "ref_engineering")
    layered_dir = args.layered_dir or _default_dir(out_root, module_name, "equiv_layered")
    traveltime_ready_dir = args.traveltime_ready_dir or _default_dir(out_root, module_name, "traveltime_ready")
    delivery_out_dir = args.delivery_out_dir or str(Path(out_root) / "velocity_delivery")

    # ---- Task: engineering ----
    def run_engineering() -> None:
        if not args.log_csv:
            raise SystemExit("--log_csv is required for task=engineering (or task=all).")

        # Call internal build script main() with a constructed argv list
        a = [
            "--input", _p(args.log_csv),
            "--out_dir", _p(ref_dir),
            "--dz", str(args.dz),
            "--vp_vs_ratio", str(args.vp_vs_ratio),
            "--ref_depth_axis", str(args.ref_depth_axis),
        ]
        if args.depth_col:
            a += ["--depth_col", args.depth_col]
        if args.vp_col:
            a += ["--vp_col", args.vp_col]
        if args.vs_col:
            a += ["--vs_col", args.vs_col]
        if args.z0 is not None:
            a += ["--z0", str(args.z0)]
        if args.zmax is not None:
            a += ["--zmax", str(args.zmax)]
        if args.smooth_median_k:
            a += ["--smooth_median_k", str(args.smooth_median_k)]
        if args.extrapolate:
            a += ["--extrapolate"]
        if args.export_profile_csv:
            a += ["--export_csv", "--export_csv_name", str(args.profile_csv_name)]
        # NOTE: exporting ref_tvd.csv is separated as its own task in this wrapper.
        _main_build_engineering(a)

    # ---- Task: export_ref_tvd ----
    def run_export_ref_tvd() -> None:
        a = [
            "--ref_dir", _p(ref_dir),
            "--ref_depth_axis", str(args.ref_depth_axis),
        ]
        if args.wellpath_path:
            a += ["--wellpath_path", _p(args.wellpath_path)]
        if args.ref_csv_out:
            a += ["--out_csv", _p(args.ref_csv_out)]
        if args.no_md_xy:
            a += ["--no_md_xy"]
        _main_export_ref_tvd(a)

    # ---- Task: layerize ----
    def run_layerize() -> None:
        # layerize reads ref_dir + writes layered_dir
        a = [
            "--ref_dir", _p(ref_dir),
            "--out_dir", _p(layered_dir),
            "--ref_depth_axis", str(args.ref_depth_axis),
            "--tau_p_ms", str(args.tau_p_ms),
            "--tau_s_ms", str(args.tau_s_ms),
            "--n_init_layers", str(args.n_init_layers),
            "--max_layers", str(args.max_layers),
            "--min_thickness_m", str(args.min_thickness_m),
            "--top_frac", str(args.top_frac),
            "--min_top", str(args.min_top),
            "--seed", str(args.seed),
            "--n_z", str(args.n_z),
            "--max_stations", str(args.max_stations),
        ]
        if args.wellpath_path:
            a += ["--wellpath_path", _p(args.wellpath_path)]
        if args.stations_csv:
            a += ["--stations_csv", _p(args.stations_csv)]
        if args.z_min is not None:
            a += ["--z_min", str(args.z_min)]
        if args.z_max is not None:
            a += ["--z_max", str(args.z_max)]
        if args.quiet:
            a += ["--quiet"]
        _main_layerize(a)

    # ---- Task: traveltime_ready ----
    def run_traveltime_ready() -> None:
        # Step 6.5 integration point: only orchestrate the dedicated builder.
        # Core traveltime_ready generation stays in
        # microseis_ds.velocity.build.build_traveltime_ready.
        a = [
            "--equiv_layered_dir", _p(layered_dir),
            "--out_dir", _p(traveltime_ready_dir),
        ]
        _main_build_traveltime_ready(a)

    # ---- Task: package ----
    def run_package() -> None:
        a = [
            "--ref_dir", _p(ref_dir),
            "--layered_dir", _p(layered_dir),
            "--out_dir", _p(delivery_out_dir),
        ]
        if args.qc_dir:
            a += ["--qc_dir", _p(args.qc_dir)]
        if args.zip:
            a += ["--zip"]
        _main_package_delivery(a)

    def run_manifest() -> None:
        # Step 5.2 integration point: run only after the selected build task completes.
        # This indexes existing formal velocity artifacts; it does not create or alter
        # ref_engineering/equiv_layered/traveltime_ready artifacts.
        include_delivery = args.task in ("package", "all")
        result = _build_velocity_manifest(
            out_root=out_root,
            module_name=module_name,
            delivery_dir=delivery_out_dir if include_delivery else None,
            source_inputs=_build_manifest_source_inputs(
                args=args,
                ref_dir=ref_dir,
                layered_dir=layered_dir,
                traveltime_ready_dir=traveltime_ready_dir,
                delivery_out_dir=delivery_out_dir,
            ),
            baseline_root=args.baseline_root or str(Path(out_root) / "baseline"),
            z_positive=str(args.z_positive),
            build_entry="scripts/build_velocity.py",
            write_files=True,
        )
        print("[OK] velocity manifest generated")
        print(f"  manifest_path : {Path(str(result['manifest_path'])).resolve()}")
        print(f"  summary_path  : {Path(str(result['summary_path'])).resolve()}")
        print(f"  asset_count   : {result['asset_count']}")

    # Dispatch
    if args.task == "engineering":
        run_engineering()
    elif args.task == "export_ref_tvd":
        run_export_ref_tvd()
    elif args.task == "layerize":
        run_layerize()
    elif args.task == "traveltime_ready":
        run_traveltime_ready()
    elif args.task == "package":
        run_package()
    elif args.task == "all":
        run_engineering()
        # export_ref_tvd is optional but strongly recommended for delivery/QC readability
        run_export_ref_tvd()
        run_layerize()
        run_traveltime_ready()
        run_package()
    else:
        raise SystemExit(f"Unknown task: {args.task}")

    run_manifest()

    print("[OK] build_velocity finished")
    print(f"  ref_dir              : {Path(ref_dir).resolve()}")
    print(f"  layered_dir          : {Path(layered_dir).resolve()}")
    print(f"  traveltime_ready_dir : {Path(traveltime_ready_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
