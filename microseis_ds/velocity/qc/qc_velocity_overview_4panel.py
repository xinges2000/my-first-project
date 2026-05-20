# microseis_ds/velocity/qc/qc_velocity_overview_4panel.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Headless-safe (CI/server)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from microseis_ds.velocity.build.layerize_1d_by_traveltime import load_engineering_1d
from microseis_ds.velocity.runtime.ref_md_tvd import load_wellpath_md_tvd


def _read_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    header = [h.strip() for h in rows[0]]
    data = rows[1:]
    # pad short rows
    ncol = len(header)
    arr = []
    for row in data:
        if not row:
            continue
        row = (row + [""] * ncol)[:ncol]
        arr.append(row)
    return header, np.asarray(arr, dtype=object)


def _pick_col(header: List[str], candidates: List[str]) -> Optional[int]:
    h2i = {h: i for i, h in enumerate(header)}
    for c in candidates:
        if c in h2i:
            return h2i[c]
    return None


def _to_float(a: np.ndarray) -> np.ndarray:
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    for i, v in enumerate(a):
        try:
            out[i] = float(v)
        except Exception:
            out[i] = np.nan
    return out


def _load_log_velocity_csv(
    log_csv: str,
    *,
    wellpath_path: Optional[str],
) -> Dict[str, np.ndarray | str | bool]:
    """
    Load vel_AC_md_v1.csv and return depth + vp (+ optional vs).
    Depth preference:
      - if tvd column exists -> use TVD
      - else if md column exists -> use MD
          * if wellpath_path provided -> convert MD->TVD by interpolation
    """
    p = Path(log_csv)
    header, data = _read_csv(p)

    i_tvd = _pick_col(header, ["TVD_m", "tvd_m", "TVD", "tvd"])
    i_md = _pick_col(header, ["MD_m", "md_m", "MD", "md"])

    # Velocity columns (robust)
    i_vp = _pick_col(header, ["vp_mps", "VP_mps", "Vp_mps", "vp", "VP", "Vp", "vel_mps", "VEL_mps", "vel"])
    i_vs = _pick_col(header, ["vs_mps", "VS_mps", "Vs_mps", "vs", "VS", "Vs"])

    if i_vp is None:
        raise ValueError(f"Log CSV missing Vp column. header={header}")

    vp = _to_float(data[:, i_vp])
    vs = _to_float(data[:, i_vs]) if i_vs is not None else None

    used_axis: str
    depth: np.ndarray
    converted = False

    if i_tvd is not None:
        depth = _to_float(data[:, i_tvd])
        used_axis = "tvd"
    elif i_md is not None:
        md = _to_float(data[:, i_md])
        used_axis = "md"
        if wellpath_path:
            md_wp, tvd_wp = load_wellpath_md_tvd(wellpath_path)
            md_clip = np.clip(md, float(np.nanmin(md_wp)), float(np.nanmax(md_wp)))
            depth = np.interp(md_clip, md_wp, tvd_wp)
            used_axis = "tvd"
            converted = True
        else:
            depth = md
    else:
        raise ValueError(f"Log CSV must have either TVD or MD column. header={header}")

    m = np.isfinite(depth) & np.isfinite(vp)
    depth = depth[m]
    vp = vp[m]
    if vs is not None:
        vs = vs[m]

    order = np.argsort(depth)
    depth = depth[order]
    vp = vp[order]
    if vs is not None:
        vs = vs[order]

    return {
        "depth_m": depth,
        "vp_mps": vp,
        "vs_mps": vs,
        "depth_axis": used_axis,
        "converted_md_to_tvd": converted,
        "log_csv": str(p),
    }


def _load_ref_tvd_csv(ref_tvd_csv: str) -> Dict[str, np.ndarray | str]:
    p = Path(ref_tvd_csv)
    header, data = _read_csv(p)

    i_tvd = _pick_col(header, ["tvd_m", "TVD_m", "tvd", "TVD", "z_m", "Z_m"])
    i_vp = _pick_col(header, ["vp_mps", "VP_mps", "vp", "VP", "Vp"])
    i_vs = _pick_col(header, ["vs_mps", "VS_mps", "vs", "VS", "Vs"])

    if i_tvd is None or i_vp is None:
        raise ValueError(f"ref_tvd.csv must contain tvd and vp columns. header={header}")

    z = _to_float(data[:, i_tvd])
    vp = _to_float(data[:, i_vp])
    vs = _to_float(data[:, i_vs]) if i_vs is not None else None

    m = np.isfinite(z) & np.isfinite(vp)
    z = z[m]
    vp = vp[m]
    if vs is not None:
        vs = vs[m]

    order = np.argsort(z)
    z = z[order]
    vp = vp[order]
    if vs is not None:
        vs = vs[order]

    return {"depth_m": z, "vp_mps": vp, "vs_mps": vs, "ref_tvd_csv": str(p)}


def _load_equiv_layered(layered_dir: str) -> Dict[str, np.ndarray | str]:
    p = Path(layered_dir)
    meta_path = p / "velocity_model_equiv_layered.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    files = meta.get("files", {})
    vp_name = files.get("vp", "vp.npy")
    vs_name = files.get("vs", "vs.npy")

    z = np.load(p / "depth.npy").astype(np.float64)
    vp = np.load(p / vp_name).astype(np.float64)
    vs = np.load(p / vs_name).astype(np.float64)

    if z.shape != vp.shape or z.shape != vs.shape:
        raise ValueError("equiv_layered depth/vp/vs shape mismatch")

    return {"depth_m": z, "vp_mps": vp, "vs_mps": vs, "layered_dir": str(p)}


def _plot_one(
    ax,
    depth_m: np.ndarray,
    vp: np.ndarray,
    vs: Optional[np.ndarray],
    title: str,
    global_dmin: float,
    global_dmax: float,
) -> None:
    ax.plot(vp, depth_m, label="Vp")
    if vs is not None and np.any(np.isfinite(vs)):
        ax.plot(vs, depth_m, label="Vs")

    ax.set_title(title)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # ✅ 统一纵轴范围 + 0在上、深度向下为正
    ax.set_ylim(global_dmax, global_dmin)

    ax.legend(loc="best", fontsize=8)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qc_velocity_overview_4panel",
        description="Delivery QC: 4-panel velocity profiles (log, ref_engineering, ref_tvd.csv, equiv_layered).",
    )
    p.add_argument("--log_csv", required=True, help="Log velocity CSV, e.g., data_pre/vel_AC_md_v1.csv")
    p.add_argument("--ref_dir", required=True, help="Reference engineering model dir, e.g., prepared_project/velocity/ref_engineering")
    p.add_argument("--ref_tvd_csv", required=True, help="ref_tvd.csv path, e.g., prepared_project/velocity/ref_engineering/ref_tvd.csv")
    p.add_argument("--layered_dir", required=True, help="Equiv layered model dir, e.g., prepared_project/velocity/equiv_layered")
    p.add_argument("--out_dir", required=True, help="QC output dir, e.g., prepared_project/qc/velocity/qc_velocity_overview_4panel")
    p.add_argument("--ref_depth_axis", type=str, default="tvd", choices=["tvd", "md"], help="Depth axis for reference model load.")
    p.add_argument("--wellpath_path", type=str, default=None, help="Wellpath CSV or dir; used for ref (if tvd) and for log MD->TVD conversion.")
    return p


def main(argv: List[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    outp = Path(args.out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # 1) log
    log = _load_log_velocity_csv(args.log_csv, wellpath_path=args.wellpath_path)

    # 2) ref_engineering (continuous)
    _, z_ref, vp_ref, vs_ref = load_engineering_1d(
        args.ref_dir,
        depth_axis=args.ref_depth_axis,
        wellpath_path=args.wellpath_path,
        export_csv=False,
    )

    # 3) ref_tvd.csv
    ref_csv = _load_ref_tvd_csv(args.ref_tvd_csv)

    # 4) equiv layered
    lay = _load_equiv_layered(args.layered_dir)

    # ==============================
    # ✅ 统一纵轴范围（四数据全局深度）
    # ==============================
    depth_sets = [
        log["depth_m"],
        z_ref,
        ref_csv["depth_m"],
        lay["depth_m"],
    ]
    global_dmin = float(min(np.nanmin(d) for d in depth_sets))
    global_dmax = float(max(np.nanmax(d) for d in depth_sets))

    # Plot (1x4)
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)

    _plot_one(
        axes[0],
        log["depth_m"],
        log["vp_mps"],
        log["vs_mps"],
        "Log vel_AC_md_v1.csv",
        global_dmin,
        global_dmax,
    )
    _plot_one(
        axes[1],
        z_ref,
        vp_ref,
        vs_ref,
        "ref_engineering (continuous 1D)",
        global_dmin,
        global_dmax,
    )
    _plot_one(
        axes[2],
        ref_csv["depth_m"],
        ref_csv["vp_mps"],
        ref_csv["vs_mps"],
        "ref_tvd.csv (exported)",
        global_dmin,
        global_dmax,
    )
    _plot_one(
        axes[3],
        lay["depth_m"],
        lay["vp_mps"],
        lay["vs_mps"],
        "equiv_layered (TT-equivalent)",
        global_dmin,
        global_dmax,
    )

    fig.suptitle("Velocity QC Overview (Left→Right: Log, Ref, ref_tvd.csv, Equiv Layered)", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_png = outp / "velocity_overview_4panel.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    report = {
        "inputs": {
            "log_csv": str(args.log_csv),
            "ref_dir": str(args.ref_dir),
            "ref_tvd_csv": str(args.ref_tvd_csv),
            "layered_dir": str(args.layered_dir),
            "ref_depth_axis": args.ref_depth_axis,
            "wellpath_path": args.wellpath_path,
        },
        "log_depth_axis_used": str(log["depth_axis"]),
        "log_converted_md_to_tvd": bool(log["converted_md_to_tvd"]),
        "depth_ranges_m": {
            "log": [float(np.nanmin(log["depth_m"])), float(np.nanmax(log["depth_m"]))],
            "ref_engineering": [float(z_ref[0]), float(z_ref[-1])],
            "ref_tvd_csv": [float(ref_csv["depth_m"][0]), float(ref_csv["depth_m"][-1])],
            "equiv_layered": [float(lay["depth_m"][0]), float(lay["depth_m"][-1])],
            "global": [global_dmin, global_dmax],
        },
        "outputs": {
            "velocity_overview_4panel_png": str(out_png.as_posix()),
        },
    }
    (outp / "report_velocity_overview_4panel.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[OK] QC velocity overview 4-panel done")
    print(f"  png: {out_png.resolve()}")
    print(f"  json: {(outp / 'report_velocity_overview_4panel.json').resolve()}")


if __name__ == "__main__":
    main()
