#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QC: Coordinate type detection / wellhead_to_gk sanity check.

Behavior:
  - Print results to console
  - Save the same text to:
      <out_root>/qc/<module_name>/qc_coor_<timestamp>.txt   (default)
"""

import argparse
import os
from datetime import datetime

from microseis_ds.common.coords import GKConfig, wellhead_to_gk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="prepared_project", help="prepared_project root")
    ap.add_argument("--module_name", default="baseline", help="module name for qc folder")

    ap.add_argument("--x", type=float, required=True, help="Raw X (could be GK-with-zone, web mercator, etc.)")
    ap.add_argument("--y", type=float, required=True, help="Raw Y")
    ap.add_argument("--lon0", type=float, default=105.0, help="Canonical GK lon0 (default 105)")
    ap.add_argument("--validate_china", action="store_true", help="Enable China-range validation")

    ap.add_argument(
        "--out_txt",
        default=None,
        help="Optional explicit output txt path. Default: <out_root>/qc/<module_name>/qc_coor_<ts>.txt",
    )
    args = ap.parse_args()

    qc_dir = os.path.join(args.out_root, "qc", args.module_name)
    os.makedirs(qc_dir, exist_ok=True)

    if args.out_txt:
        out_txt = args.out_txt
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_txt = os.path.join(qc_dir, f"qc_coor_{ts}.txt")

    gk_target = GKConfig(lon_0=float(args.lon0))
    res = wellhead_to_gk(float(args.x), float(args.y), gk_target, validate_china=bool(args.validate_china))

    lines = []
    lines.append("QC: qc_coor (coordinate detection)")
    lines.append(f"input_x: {args.x}")
    lines.append(f"input_y: {args.y}")
    lines.append(f"lon0_target: {args.lon0}")
    lines.append(f"validate_china: {bool(args.validate_china)}")
    lines.append("")

    lines.append(f"detected_type: {getattr(res, 'detected_type', None)}")
    lines.append(f"lon/lat: {getattr(res, 'lon', None)}  {getattr(res, 'lat', None)}")
    lines.append(f"GK(E,N): {getattr(res, 'gk_e', None)}  {getattr(res, 'gk_n', None)}")
    lines.append(f"recommended_lon0: {getattr(res, 'recommended_lon0', None)}")

    if getattr(res, "detected_type", None) == "GK_WITH_ZONE_6DEG":
        lines.append("")
        lines.append(f"zone: {getattr(res, 'gk_zone', None)}")
        lines.append(f"zone_on: {getattr(res, 'gk_zone_on', None)}")
        lines.append(f"lon0_from_zone: {getattr(res, 'gk_lon0_from_zone', None)}")

    text = "\n".join(lines)

    # console
    print(text)

    # file
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print("[OK] saved txt:", out_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())