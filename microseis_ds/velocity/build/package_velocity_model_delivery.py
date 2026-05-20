# microseis_ds/velocity/build/package_velocity_model_delivery.py
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _must_exist(p: Path, desc: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing {desc}: {p}")


def _copy_files(src_dir: Path, dst_dir: Path, files: List[str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        sp = src_dir / name
        _must_exist(sp, name)
        shutil.copy2(sp, dst_dir / name)


def _copy_glob(src_dir: Path, dst_dir: Path, pattern: str) -> List[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for sp in sorted(src_dir.glob(pattern)):
        if sp.is_file():
            shutil.copy2(sp, dst_dir / sp.name)
            out.append(sp.name)
    return out


def _npy_stats(path: Path) -> Dict:
    arr = np.load(path).astype(float)
    return {"path": str(path), "shape": list(arr.shape), "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr))}


def write_readme(out_dir: Path) -> None:
    text = """# 速度模型交付包（不含走时表）

本交付包包含两套 1D 速度模型：

1. **ref_engineering/**：工程连续参考模型（TVD 轴，dz=1m），用于作为“真实参考”与溯源。
2. **equiv_layered/**：走时等效分层模型（分段常数），用于后续走时表计算与微地震定位（Snell 1D 假设）。

## 文件说明
- `depth.npy / vp.npy / vs.npy`：程序读取用的数值数组（单位：m / m/s）。
- `ref_tvd.csv`：面向工程验收的可读版本（TVD-Vp-Vs，并附 MD/XY 插值列）。
- `qc_traveltime_equiv.json`：走时等效 QC 统计摘要（P/S 的 max / p95 / rms 等）。
- `qc/*.png`：QC 图（ΔT 分布、随深度/偏移变化等）。

## 使用约定
- 深度为 **TVD（垂向深度）**，单位 m，默认向下为正。
- 坐标为工程投影坐标（米），与项目 stations/wellpath 坐标系一致。
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def package_delivery(
    ref_dir: str,
    layered_dir: str,
    qc_dir: Optional[str],
    out_dir: str,
    make_zip: bool = False,
) -> Path:
    refp = Path(ref_dir)
    layp = Path(layered_dir)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    ref_files = ["velocity_model.json", "depth.npy", "vp.npy", "vs.npy", "ref_tvd.csv", "ref_tvd_summary.json"]
    layered_files = ["velocity_model_equiv_layered.json", "depth.npy", "vp.npy", "vs.npy", "qc_traveltime_equiv.json"]

    _copy_files(refp, outp / "ref_engineering", ref_files)
    _copy_files(layp, outp / "equiv_layered", layered_files)

    copied_qc = []
    if qc_dir is not None:
        qcp = Path(qc_dir)
        _must_exist(qcp, "qc_dir")
        copied_qc += _copy_glob(qcp, outp / "qc", "*.png")
        copied_qc += _copy_glob(qcp, outp / "qc", "*.json")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bundle": str(outp),
        "contents": {
            "ref_engineering": {
                "dir": "ref_engineering",
                "files": ref_files,
                "stats": {
                    "depth": _npy_stats(outp / "ref_engineering" / "depth.npy"),
                    "vp": _npy_stats(outp / "ref_engineering" / "vp.npy"),
                    "vs": _npy_stats(outp / "ref_engineering" / "vs.npy"),
                },
            },
            "equiv_layered": {
                "dir": "equiv_layered",
                "files": layered_files,
                "stats": {
                    "depth": _npy_stats(outp / "equiv_layered" / "depth.npy"),
                    "vp": _npy_stats(outp / "equiv_layered" / "vp.npy"),
                    "vs": _npy_stats(outp / "equiv_layered" / "vs.npy"),
                },
            },
            "qc": {"dir": "qc", "files": copied_qc},
        },
        "notes": {
            "depth_axis": "TVD (m, positive down)",
            "usage": "Use equiv_layered for traveltime tables and localization; keep ref_engineering for traceability/QC baseline.",
        },
    }
    (outp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(outp)

    if make_zip:
        zip_path = outp.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(outp), "zip", root_dir=outp)
        print(f"[OK] Wrote zip: {zip_path.resolve()}")

    print(f"[OK] Delivery bundle ready: {outp.resolve()}")
    return outp


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="package_velocity_model_delivery", description="Package velocity model delivery bundle (no traveltime table).")
    p.add_argument("--ref_dir", required=True, help="Ref engineering model dir (velocity_model.json + npy + ref_tvd.csv).")
    p.add_argument("--layered_dir", required=True, help="Equiv layered model dir (velocity_model_equiv_layered.json + npy + qc_traveltime_equiv.json).")
    p.add_argument("--qc_dir", default=None, help="QC output dir (png/json). Optional.")
    p.add_argument("--out_dir", required=True, help="Delivery bundle output dir.")
    p.add_argument("--zip", action="store_true", help="Also create <out_dir>.zip")
    return p


def main(argv=None) -> None:
    args = _build_argparser().parse_args(argv)
    package_delivery(ref_dir=args.ref_dir, layered_dir=args.layered_dir, qc_dir=args.qc_dir, out_dir=args.out_dir, make_zip=bool(args.zip))


if __name__ == "__main__":
    main()