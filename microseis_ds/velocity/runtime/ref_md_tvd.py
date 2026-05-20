from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence, List
import json
import math

import numpy as np
import pandas as pd


# -----------------------------
# Data classes
# -----------------------------
@dataclass(frozen=True)
class Ref1DModel:
    """Reference 1D model on vertical depth axis."""
    z_m: np.ndarray
    vp_mps: np.ndarray
    vs_mps: np.ndarray
    meta: Dict


@dataclass(frozen=True)
class WellpathFull:
    """Wellpath table for TVD -> (MD, X, Y) interpolation."""
    md_m: np.ndarray
    tvd_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    source_csv: str


# -----------------------------
# Small utilities
# -----------------------------
def _pick_latest_csv(path: Path) -> Path:
    if path.is_file():
        return path
    csvs = sorted(path.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No *.csv found under: {path}")
    return csvs[0]


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _norm_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def _resolve_col(df: pd.DataFrame, preferred: Optional[str], aliases: List[str], role: str) -> str:
    cols = list(df.columns)
    norm_map = {_norm_col(c): c for c in cols}

    if preferred:
        if preferred in cols:
            return preferred
        k = _norm_col(preferred)
        if k in norm_map:
            return norm_map[k]

    for a in aliases:
        k = _norm_col(a)
        if k in norm_map:
            return norm_map[k]

    raise ValueError(f"Cannot find {role} column. preferred={preferred!r}, aliases={aliases}, available={cols}")


def _ensure_strictly_increasing(x: np.ndarray, name: str) -> None:
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"{name} must be 1D with >=2 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN/Inf")
    if not np.all(np.diff(x) > 0):
        raise ValueError(f"{name} must be strictly increasing")


def _build_uniform_grid_including_end(z_min: float, z_max: float, dz: float) -> np.ndarray:
    """
    Build a uniform grid [z_min, ..., z_max] with step dz,
    guaranteeing z[-1] == z_max exactly (end padding).
    """
    if dz <= 0:
        raise ValueError("dz must be > 0")
    if z_max <= z_min:
        raise ValueError(f"z_max must be > z_min, got {z_min}, {z_max}")

    # robust n computation (avoid float drift)
    n = int(math.floor((z_max - z_min) / dz + 1e-12))
    z = z_min + dz * np.arange(n + 1, dtype=float)

    # ensure inclusion of z_max even if (z_max-z_min) is not an exact multiple of dz
    if z[-1] < z_max - 1e-9:
        z = np.append(z, z_max)

    # FORCE last exactly equals z_max (your "补齐最后一行" requirement)
    z[-1] = z_max
    return z


# -----------------------------
# Wellpath loading
# -----------------------------
def load_wellpath_md_tvd(wellpath_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (MD, TVD) from wellpath CSV or directory.
    TVD is forced to be non-decreasing vs MD (horizontal section plateau allowed).
    """
    p = Path(wellpath_path)
    if p.is_dir():
        p = _pick_latest_csv(p)

    df = pd.read_csv(p)

    md_col = _pick_column(df, ["MD_m", "MD", "md_m", "md"])
    if md_col is None:
        raise ValueError(f"wellpath missing MD column. columns={list(df.columns)}")

    tvd_col = _pick_column(df, ["TVD_m", "TVD", "tvd_m", "tvd"])
    z_col = _pick_column(df, ["Z_m", "Z", "z_m", "z"])

    if tvd_col is None and z_col is None:
        raise ValueError(f"wellpath missing TVD_m or Z_m. columns={list(df.columns)}")

    md = pd.to_numeric(df[md_col], errors="coerce").to_numpy(float)
    tvd = pd.to_numeric(df[tvd_col], errors="coerce").to_numpy(float) if tvd_col is not None else pd.to_numeric(df[z_col], errors="coerce").to_numpy(float)

    m = np.isfinite(md) & np.isfinite(tvd)
    md = md[m]
    tvd = tvd[m]
    if md.size < 2:
        raise ValueError("wellpath has too few valid MD/TVD samples")

    order = np.argsort(md)
    md = md[order]
    tvd = tvd[order]

    # unique MD, keep first occurrence
    md_u, idx = np.unique(md, return_index=True)
    md = md_u
    tvd = tvd[idx]

    # enforce non-decreasing tvd along MD
    tvd = np.maximum.accumulate(tvd)

    if md.size < 2:
        raise ValueError("wellpath has too few unique MD samples")
    return md, tvd


def _aggregate_duplicate_tvd_keep_mdmax(dfc: pd.DataFrame) -> pd.DataFrame:
    """
    For horizontal wells, many MD map to nearly same TVD.
    For a single-valued TVD->MD mapping, keep the row with maximum MD for each TVD (tail of plateau).
    """
    dfc = dfc.sort_values(["tvd", "md"], kind="mergesort")
    dfc = dfc.drop_duplicates(subset=["tvd"], keep="last")
    return dfc


def load_wellpath_full(wellpath_path: str | Path) -> WellpathFull:
    """
    Load (MD, TVD, X, Y) from wellpath for interpolation TVD->(MD,X,Y).

    Key behavior:
      - If TVD has duplicates (plateau), aggregate by TVD and keep md_max(tvd) row.
      - Require strictly increasing TVD after aggregation.
    """
    p = _pick_latest_csv(Path(wellpath_path))
    df = pd.read_csv(p)

    md_col = _resolve_col(df, None, ["MD_m", "MD", "md_m", "md"], "wellpath MD")
    tvd_col = _resolve_col(df, None, ["TVD_m", "TVD", "tvd_m", "tvd", "Z_m", "Z", "z_m", "z"], "wellpath TVD")
    x_col = _resolve_col(df, None, ["X_m", "X", "E_gk_m", "E"], "wellpath X")
    y_col = _resolve_col(df, None, ["Y_m", "Y", "N_gk_m", "N"], "wellpath Y")

    md = pd.to_numeric(df[md_col], errors="coerce").to_numpy(float)
    tvd = pd.to_numeric(df[tvd_col], errors="coerce").to_numpy(float)
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)

    m = np.isfinite(md) & np.isfinite(tvd) & np.isfinite(x) & np.isfinite(y)
    dfc = pd.DataFrame({"md": md[m], "tvd": tvd[m], "x": x[m], "y": y[m]})
    if dfc.shape[0] < 2:
        raise ValueError("wellpath has too few valid samples after cleaning")

    dfc = dfc.sort_values(["tvd", "md"], kind="mergesort")
    if (dfc["tvd"].diff().fillna(1.0) == 0).any():
        dfc = _aggregate_duplicate_tvd_keep_mdmax(dfc)

    tvd_u = dfc["tvd"].to_numpy(float)
    md_u = dfc["md"].to_numpy(float)
    x_u = dfc["x"].to_numpy(float)
    y_u = dfc["y"].to_numpy(float)

    _ensure_strictly_increasing(tvd_u, "wellpath TVD (after plateau aggregation)")

    return WellpathFull(md_m=md_u, tvd_m=tvd_u, x_m=x_u, y_m=y_u, source_csv=str(p))


# -----------------------------
# Ref model loading / MD->TVD conversion
# -----------------------------
def load_ref_model_as_tvd(
    ref_dir: str | Path,
    wellpath_path: str | Path,
    *,
    out_dz_m: Optional[float] = None,
    extend_to_wellpath: bool = True,
    extrapolate_mode: str = "hold",
    tvd_max_rounding: str = "ceil",
) -> Ref1DModel:
    """
    Convert ref model from MD-axis (log depth) to TVD-axis using wellpath.

    - extend_to_wellpath=True:
        TVD range covers [wellpath_tvd_min, rounded(wellpath_tvd_max)]
    - tvd_max_rounding='ceil':
        tvd_max = ceil(wellpath_tvd_max)  (per your delivery requirement)
    - extrapolate_mode:
        'hold' (default): endpoint hold beyond ref-supported TVD (stable)
        'linear': linear extrapolation beyond ref-supported TVD (use with caution)
    """
    extrapolate_mode = str(extrapolate_mode).strip().lower()
    if extrapolate_mode not in {"hold", "linear"}:
        raise ValueError("extrapolate_mode must be one of: hold, linear")

    tvd_max_rounding = str(tvd_max_rounding).strip().lower()
    if tvd_max_rounding not in {"ceil", "none"}:
        raise ValueError("tvd_max_rounding must be one of: ceil, none")

    ref_dir = Path(ref_dir)
    j = json.loads((ref_dir / "velocity_model.json").read_text(encoding="utf-8"))
    files = j.get("files", {})
    vp_name = files.get("vp", "vp.npy")
    vs_name = files.get("vs", "vs.npy")

    md_grid = np.load(ref_dir / "depth.npy").astype(float)
    vp_md = np.load(ref_dir / vp_name).astype(float)
    vs_md = np.load(ref_dir / vs_name).astype(float)

    _ensure_strictly_increasing(md_grid, "ref MD grid")
    if not (vp_md.ndim == vs_md.ndim == 1 and vp_md.size == md_grid.size and vs_md.size == md_grid.size):
        raise ValueError("ref vp/vs must be 1D and same length as depth.npy")
    if not (np.all(np.isfinite(vp_md)) and np.all(np.isfinite(vs_md))):
        raise ValueError("ref vp/vs contains NaN/Inf")

    # dz preference: meta grid.dz -> fallback median diff on ref depth
    if out_dz_m is None:
        dz = None
        try:
            dz = float(j.get("grid", {}).get("spacing", {}).get("dz", None))
        except Exception:
            dz = None
        if dz is None or not np.isfinite(dz) or dz <= 0:
            dz = float(np.nanmedian(np.diff(md_grid)))
        out_dz_m = float(dz)
    if out_dz_m <= 0:
        raise ValueError("out_dz_m must be > 0")

    # wellpath MD->TVD (plateau allowed)
    md_wp, tvd_wp = load_wellpath_md_tvd(wellpath_path)
    tvd_at_md = np.interp(md_grid, md_wp, tvd_wp)

    # build unique TVD mapping from ref samples: keep max-MD for each TVD
    order = np.lexsort((md_grid, tvd_at_md))
    tvd_s = tvd_at_md[order]
    md_s = md_grid[order]
    vp_s = vp_md[order]
    vs_s = vs_md[order]

    # keep last of each TVD => md_max(tvd)
    tvd_rev = tvd_s[::-1]
    idx_rev = np.unique(tvd_rev, return_index=True)[1]
    idx_last = (tvd_s.size - 1 - idx_rev)
    idx_last.sort()

    tvd_u = tvd_s[idx_last]
    vp_u = vp_s[idx_last]
    vs_u = vs_s[idx_last]
    md_u = md_s[idx_last]

    _ensure_strictly_increasing(tvd_u, "ref-derived unique TVD")

    tvd_wp_min = float(np.min(tvd_wp))
    tvd_wp_max = float(np.max(tvd_wp))
    tvd_ref_min = float(tvd_u[0])
    tvd_ref_max = float(tvd_u[-1])

    if extend_to_wellpath:
        tvd_min = tvd_wp_min
        if tvd_max_rounding == "ceil":
            tvd_max = float(math.ceil(tvd_wp_max))
        else:
            tvd_max = float(tvd_wp_max)
    else:
        tvd_min = tvd_ref_min
        tvd_max = tvd_ref_max

    # build TVD grid with strict end padding ("最后一行补齐")
    z_tvd = _build_uniform_grid_including_end(tvd_min, tvd_max, out_dz_m)

    # interpolate vp/vs with chosen extrapolation
    if extrapolate_mode == "hold":
        vp_tvd = np.interp(z_tvd, tvd_u, vp_u, left=float(vp_u[0]), right=float(vp_u[-1]))
        vs_tvd = np.interp(z_tvd, tvd_u, vs_u, left=float(vs_u[0]), right=float(vs_u[-1]))
    else:
        # linear extrapolation using endpoint slopes (in TVD domain)
        sL_vp = (vp_u[1] - vp_u[0]) / (tvd_u[1] - tvd_u[0])
        sR_vp = (vp_u[-1] - vp_u[-2]) / (tvd_u[-1] - tvd_u[-2])
        sL_vs = (vs_u[1] - vs_u[0]) / (tvd_u[1] - tvd_u[0])
        sR_vs = (vs_u[-1] - vs_u[-2]) / (tvd_u[-1] - tvd_u[-2])

        vp_mid = np.interp(z_tvd, tvd_u, vp_u)
        vs_mid = np.interp(z_tvd, tvd_u, vs_u)

        vp_tvd = vp_mid.copy()
        vs_tvd = vs_mid.copy()

        left = z_tvd < tvd_u[0]
        right = z_tvd > tvd_u[-1]
        vp_tvd[left] = vp_u[0] + sL_vp * (z_tvd[left] - tvd_u[0])
        vp_tvd[right] = vp_u[-1] + sR_vp * (z_tvd[right] - tvd_u[-1])
        vs_tvd[left] = vs_u[0] + sL_vs * (z_tvd[left] - tvd_u[0])
        vs_tvd[right] = vs_u[-1] + sR_vs * (z_tvd[right] - tvd_u[-1])

    meta = {
        "ref_dir": str(ref_dir),
        "depth_axis": "TVD_m",
        "source_depth_axis": "MD_m",
        "dz_m": float(out_dz_m),
        "tvd_range_m": [float(z_tvd[0]), float(z_tvd[-1])],
        "wellpath": str(Path(wellpath_path)),
        "extend_to_wellpath": bool(extend_to_wellpath),
        "extrapolate_mode": extrapolate_mode,
        "tvd_max_rounding": tvd_max_rounding,
        "mapping_debug": {
            "tvd_supported_by_ref_m": [tvd_ref_min, tvd_ref_max],
            "tvd_wellpath_m": [tvd_wp_min, tvd_wp_max],
            "tvd_wellpath_max_rounded_m": float(math.ceil(tvd_wp_max)) if tvd_max_rounding == "ceil" else float(tvd_wp_max),
            "md_u_minmax_m": [float(np.min(md_u)), float(np.max(md_u))],
        },
        "note": (
            "TVD grid is generated with strict end padding: last sample equals TVD_max exactly. "
            "Horizontal well TVD plateau handled by keeping md_max(tvd) when building single-valued mapping."
        ),
    }

    return Ref1DModel(z_m=z_tvd, vp_mps=vp_tvd, vs_mps=vs_tvd, meta=meta)


def load_ref_model(
    ref_dir: str | Path,
    *,
    depth_axis: str = "tvd",
    wellpath_path: Optional[str | Path] = None,
) -> Ref1DModel:
    """
    Public API for other modules (layerize, QC, export).

    depth_axis:
      - 'tvd': interpret ref depth.npy as MD and convert to TVD using wellpath_path (delivery default)
      - 'md' : treat ref depth.npy as already vertical depth (legacy/debug)
    """
    depth_axis = str(depth_axis).strip().lower()
    if depth_axis not in {"tvd", "md"}:
        raise ValueError("depth_axis must be 'tvd' or 'md'")

    ref_dir = Path(ref_dir)
    j = json.loads((ref_dir / "velocity_model.json").read_text(encoding="utf-8"))
    files = j.get("files", {})
    vp_name = files.get("vp", "vp.npy")
    vs_name = files.get("vs", "vs.npy")

    if depth_axis == "md":
        z = np.load(ref_dir / "depth.npy").astype(float)
        vp = np.load(ref_dir / vp_name).astype(float)
        vs = np.load(ref_dir / vs_name).astype(float)
        return Ref1DModel(z_m=z, vp_mps=vp, vs_mps=vs, meta={"ref_dir": str(ref_dir), "depth_axis": "MD_or_Z_m"})

    if wellpath_path is None:
        raise ValueError("wellpath_path is required when depth_axis='tvd'")

    # delivery defaults
    return load_ref_model_as_tvd(
        ref_dir,
        wellpath_path,
        extend_to_wellpath=True,
        extrapolate_mode="hold",
        tvd_max_rounding="ceil",
    )
