# microseis_ds/velocity/qc/qc_check_velocity_model_artifact.py
"""
Check a built velocity model artifact (engineering output) without creating any temp folders.

Usage (from repo root, PowerShell):
  $env:PYTHONPATH="."
  python -m microseis_ds.velocity.qc.qc_check_velocity_model_artifact `
    --path prepared_project/velocity/ref_engineering

  python -m microseis_ds.velocity.qc.qc_check_velocity_model_artifact `
    --path prepared_project/velocity/ref_engineering `
    --z 0 100 500 1000 `
    --export_csv prepared_project/qc/velocity/qc_velocity_artifact_query.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# NOTE: align with your refactor: VelocityModel is runtime callable
from microseis_ds.velocity.runtime.velocity_model import VelocityModel


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Check velocity model artifact (load/validate/query).")
    ap.add_argument(
        "--path",
        default="prepared_project/velocity/ref_engineering",
        help="Path to velocity model dir or velocity_model.json (default: prepared_project/velocity/ref_engineering)",
    )
    ap.add_argument(
        "--z",
        nargs="+",
        type=float,
        default=[0.0, 100.0, 500.0, 1000.0],
        help="Query depths in meters (default: 0 100 500 1000)",
    )
    ap.add_argument(
        "--export_csv",
        default=None,
        help="Optional CSV output path for queried vp/vs at given z values.",
    )
    ap.add_argument(
        "--no_extrapolate",
        action="store_true",
        help="If set, clip query depths to model range before querying.",
    )

    args = ap.parse_args(argv)
    path = args.path
    z_list = list(map(float, args.z))

    vm = VelocityModel.load(path)

    print("[OK] Loaded velocity model artifact")
    print(f"  path   : {Path(path).resolve()}")
    print(f"  summary: {vm.summary()}")

    zq = np.asarray(z_list, dtype=float)
    if args.no_extrapolate:
        zq = np.clip(zq, float(vm.depth_m.min()), float(vm.depth_m.max()))

    vp, vs = vm.query(zq.tolist())

    print("  query z (m):", zq.tolist())
    print("  vp (m/s)  :", [float(x) for x in vp])
    print("  vs (m/s)  :", [float(x) for x in vs])

    if args.export_csv:
        out_csv = Path(args.export_csv)
        _ensure_parent_dir(out_csv)
        lines = ["depth_m,vp_mps,vs_mps\n"]
        for zi, vpi, vsi in zip(zq.tolist(), vp.tolist(), vs.tolist()):
            lines.append(f"{zi:.6f},{float(vpi):.6f},{float(vsi):.6f}\n")
        out_csv.write_text("".join(lines), encoding="utf-8")
        print(f"[OK] Wrote CSV: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
