# microseis_ds/velocity/build/layerize_1d_by_traveltime.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

from microseis_ds.velocity.runtime.ref_md_tvd import load_ref_model, load_wellpath_md_tvd

# NOTE: export_ref_tvd_csv writes files -> must live in build
from microseis_ds.velocity.build.build_export_ref_tvd_csv import export_ref_tvd_csv


def _read_csv_columns(path: Path) -> Tuple[List[str], List[List[str]]]:
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    header = [h.strip() for h in rows[0]]
    data = rows[1:]
    return header, data


def _col_idx(header: List[str], candidates: List[str]) -> Optional[int]:
    h2i = {h: i for i, h in enumerate(header)}
    for c in candidates:
        if c in h2i:
            return h2i[c]
    return None


def load_stations_xy(stations_csv: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    p = Path(stations_csv)
    header, rows = _read_csv_columns(p)
    ix = _col_idx(header, ["X_m", "X", "E_gk_m", "E"])
    iy = _col_idx(header, ["Y_m", "Y", "N_gk_m", "N"])
    if ix is None or iy is None:
        raise ValueError(
            f"stations CSV must contain X/Y columns. Found header={header}. "
            "Tried X_m/X/E_gk_m/E and Y_m/Y/N_gk_m/N."
        )
    xs: List[float] = []
    ys: List[float] = []
    for r in rows:
        if not r or len(r) <= max(ix, iy):
            continue
        try:
            xs.append(float(r[ix]))
            ys.append(float(r[iy]))
        except Exception:
            continue
    if len(xs) < 1:
        raise ValueError(f"No valid station rows in {stations_csv}")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), {"ix": ix, "iy": iy}


def _find_single_csv_in_dir(d: Path) -> Path:
    csvs = sorted([p for p in d.glob("*.csv") if p.is_file()])
    if len(csvs) == 0:
        raise ValueError(f"No .csv found in wellpath dir: {d}")
    if len(csvs) > 1:
        csvs = sorted(csvs, key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0]


def load_wellpath_md_xy(wellpath_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    wp = Path(wellpath_path)
    if wp.is_dir():
        wp = _find_single_csv_in_dir(wp)
    if not wp.exists():
        raise FileNotFoundError(f"wellpath not found: {wellpath_path}")

    header, rows = _read_csv_columns(wp)
    i_md = _col_idx(header, ["MD_m", "MD", "md_m", "md"])
    i_x = _col_idx(header, ["X_m", "X", "E_gk_m", "E"])
    i_y = _col_idx(header, ["Y_m", "Y", "N_gk_m", "N"])
    if i_md is None or i_x is None or i_y is None:
        raise ValueError(
            f"wellpath CSV must contain MD and X/Y columns. Found header={header}. "
            "Tried MD_m/MD and X_m/X/E_gk_m/E and Y_m/Y/N_gk_m/N."
        )

    md, xs, ys = [], [], []
    for r in rows:
        if not r or len(r) <= max(i_md, i_x, i_y):
            continue
        try:
            mdv = float(r[i_md]); xv = float(r[i_x]); yv = float(r[i_y])
        except Exception:
            continue
        if not np.isfinite(mdv) or not np.isfinite(xv) or not np.isfinite(yv):
            continue
        md.append(mdv); xs.append(xv); ys.append(yv)

    if len(md) < 2:
        raise ValueError(f"Not enough valid wellpath points in {wp}")
    md = np.asarray(md, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    order = np.argsort(md)
    md, xs, ys = md[order], xs[order], ys[order]
    uniq = np.concatenate([[True], np.diff(md) > 1e-9])
    md, xs, ys = md[uniq], xs[uniq], ys[uniq]

    if md.size < 2 or not np.all(np.diff(md) > 0):
        raise ValueError(f"wellpath MD must be strictly increasing after cleanup: {wp}")

    meta = {
        "wellpath_csv": str(wp),
        "columns": {"md": i_md, "x": i_x, "y": i_y},
        "n_points": int(md.size),
        "md_min_m": float(md.min()),
        "md_max_m": float(md.max()),
    }
    return md, xs, ys, meta


@dataclass(frozen=True)
class Ray:
    z_s: float
    R: float

    def key(self) -> tuple[float, float]:
        return (float(self.z_s), float(self.R))


def make_ray_set(z_min: float, z_max: float, n_z: int, r_over_z: Iterable[float]) -> list[Ray]:
    if n_z < 2:
        raise ValueError("n_z must be >= 2")
    zs = np.linspace(z_min, z_max, n_z, dtype=float)
    r_over_z = list(r_over_z)
    rays: list[Ray] = []
    for z in zs:
        for k in r_over_z:
            rays.append(Ray(z_s=float(z), R=float(k) * float(z)))
    return rays


def make_ray_set_from_stations(
    stations_csv: str,
    src_x_m: float,
    src_y_m: float,
    z_min: float,
    z_max: float,
    n_z: int,
    max_stations: int = 500,
    seed: int = 0,
) -> tuple[list[Ray], dict]:
    xs, ys, info = load_stations_xy(stations_csv)
    n_sta = int(xs.size)

    if n_sta > max_stations:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n_sta, size=int(max_stations), replace=False)
        xs = xs[idx]
        ys = ys[idx]
        n_sta_used = int(xs.size)
    else:
        n_sta_used = n_sta

    zs = np.linspace(float(z_min), float(z_max), int(n_z), dtype=float)
    rays: list[Ray] = []
    dx = xs - float(src_x_m)
    dy = ys - float(src_y_m)
    Rs = np.sqrt(dx * dx + dy * dy)
    for z_s in zs:
        for R in Rs:
            rays.append(Ray(z_s=float(z_s), R=float(R)))

    meta = {
        "mode": "stations",
        "stations_csv": str(stations_csv),
        "stations_total": n_sta,
        "stations_used": n_sta_used,
        "src_x_m": float(src_x_m),
        "src_y_m": float(src_y_m),
        "z_min_m": float(z_min),
        "z_max_m": float(z_max),
        "n_z": int(n_z),
        "max_stations": int(max_stations),
        "seed": int(seed),
        "columns": info,
    }
    return rays, meta


def make_ray_set_from_stations_wellpath(
    stations_csv: str,
    wellpath_path: str,
    z_min: float,
    z_max: float,
    n_z: int,
    max_stations: int = 500,
    seed: int = 0,
) -> tuple[list[Ray], dict]:
    xs, ys, info = load_stations_xy(stations_csv)
    n_sta = int(xs.size)

    if n_sta > max_stations:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n_sta, size=int(max_stations), replace=False)
        xs = xs[idx]
        ys = ys[idx]
        n_sta_used = int(xs.size)
    else:
        n_sta_used = n_sta

    md_xy, wx, wy, wp_meta = load_wellpath_md_xy(wellpath_path)
    md_tvd, tvd = load_wellpath_md_tvd(wellpath_path)

    tvd_at_md_xy = np.interp(md_xy, md_tvd, tvd)

    zs_tvd = np.linspace(float(z_min), float(z_max), int(n_z), dtype=float)

    zmin_wp = float(np.nanmin(tvd_at_md_xy))
    zmax_wp = float(np.nanmax(tvd_at_md_xy))
    zs_clamped = np.clip(zs_tvd, zmin_wp, zmax_wp)

    md_src = np.interp(zs_clamped, tvd_at_md_xy, md_xy)

    x_src = np.interp(md_src, md_xy, wx)
    y_src = np.interp(md_src, md_xy, wy)

    rays: list[Ray] = []
    for z_s, sx, sy in zip(zs_tvd, x_src, y_src):
        dx = xs - float(sx)
        dy = ys - float(sy)
        Rs = np.sqrt(dx * dx + dy * dy)
        for R in Rs:
            rays.append(Ray(z_s=float(z_s), R=float(R)))

    meta = {
        "mode": "stations_wellpath_tvd",
        "stations_csv": str(stations_csv),
        "stations_total": n_sta,
        "stations_used": n_sta_used,
        "wellpath": wp_meta,
        "z_min_m": float(z_min),
        "z_max_m": float(z_max),
        "n_z": int(n_z),
        "max_stations": int(max_stations),
        "seed": int(seed),
        "columns": {"stations": info, "wellpath": wp_meta.get("columns", {})},
        "src_samples": [
            {"z_s_tvd_m": float(zs_tvd[i]), "md_m": float(md_src[i]), "x_m": float(x_src[i]), "y_m": float(y_src[i])}
            for i in np.linspace(0, len(zs_tvd) - 1, num=min(5, len(zs_tvd)), dtype=int)
        ],
        "zs_clamped_to_wellpath": bool(not np.allclose(zs_tvd, zs_clamped)),
        "wellpath_tvd_range_m": [zmin_wp, zmax_wp],
    }
    return rays, meta


@dataclass
class RayPath:
    z_s: float
    R: float
    idx: np.ndarray
    frac: np.ndarray
    dz: np.ndarray
    zmid: np.ndarray

    def key(self) -> tuple[float, float]:
        return (float(self.z_s), float(self.R))


def _build_raypath(z: np.ndarray, ray: Ray) -> RayPath:
    z0 = float(z[0])
    z_s = float(ray.z_s)
    if z_s <= z0:
        idx = np.array([0, 0], dtype=np.int32)
        frac = np.array([0.0, 0.0], dtype=np.float64)
        dz = np.array([max(z_s - z0, 0.0)], dtype=np.float64)
        zmid = np.array([0.5 * (z0 + z_s)], dtype=np.float64)
        return RayPath(z_s=z_s, R=float(ray.R), idx=idx, frac=frac, dz=dz, zmid=zmid)

    j = int(np.searchsorted(z, z_s, side="right") - 1)
    j = max(0, min(j, len(z) - 2))
    z_nodes = z[: j + 1].astype(np.float64, copy=True)
    if abs(float(z_nodes[-1]) - z_s) > 1e-12:
        z_nodes = np.concatenate([z_nodes, np.array([z_s], dtype=np.float64)])
        idx_last = j
        frac_last = (z_s - float(z[j])) / (float(z[j + 1]) - float(z[j]))
        idx = np.concatenate([np.arange(0, j + 1, dtype=np.int32), np.array([idx_last], dtype=np.int32)])
        frac = np.concatenate([np.zeros(j + 1, dtype=np.float64), np.array([frac_last], dtype=np.float64)])
    else:
        idx = np.arange(0, j + 1, dtype=np.int32)
        frac = np.zeros_like(idx, dtype=np.float64)

    dz = np.diff(z_nodes)
    zmid = 0.5 * (z_nodes[:-1] + z_nodes[1:])
    return RayPath(z_s=z_s, R=float(ray.R), idx=idx, frac=frac, dz=dz, zmid=zmid)


def build_raypaths(z: np.ndarray, rays: list[Ray]) -> list[RayPath]:
    return [_build_raypath(z, r) for r in rays]


def _sample_v_nodes(v: np.ndarray, rp: RayPath) -> np.ndarray:
    i = rp.idx
    f = rp.frac
    v0 = v[i]
    v1 = v[i + 1]
    return (1.0 - f) * v0 + f * v1


def _integrate_x_t_from_nodes(v_nodes: np.ndarray, dz: np.ndarray, p: float) -> tuple[float, float, np.ndarray]:
    vm = 0.5 * (v_nodes[:-1] + v_nodes[1:])
    pv = p * vm
    if np.any(pv >= 1.0):
        return np.inf, np.inf, np.full_like(dz, np.inf, dtype=np.float64)
    cos = np.sqrt(1.0 - pv * pv)
    dx = (p * vm * vm / cos) * dz
    dt = ((1.0 / vm) / cos) * dz
    return float(np.sum(dx)), float(np.sum(dt)), dt


def raytrace_1d_snell_precomputed(
    v: np.ndarray,
    rp: RayPath,
    max_iter: int = 20,
    p_tol: float = 1e-10,
) -> tuple[float, float, np.ndarray]:
    R = float(rp.R)
    if R <= 0.0 or rp.z_s <= float(rp.zmid[0]) * 0 + 1e-12:
        v_nodes = _sample_v_nodes(v, rp)
        vm = 0.5 * (v_nodes[:-1] + v_nodes[1:])
        dt = (rp.dz / vm)
        return float(np.sum(dt)), 0.0, dt

    v_nodes = _sample_v_nodes(v, rp)
    vm = 0.5 * (v_nodes[:-1] + v_nodes[1:])
    vmin = float(np.min(vm))
    p_hi = 0.999999 / max(vmin, 1e-12)
    p_lo = 0.0

    x_hi, t_hi, _ = _integrate_x_t_from_nodes(v_nodes, rp.dz, p_hi)
    if not np.isfinite(x_hi):
        p_hi = 0.999 / max(vmin, 1e-12)
        x_hi, t_hi, _ = _integrate_x_t_from_nodes(v_nodes, rp.dz, p_hi)

    if x_hi < R:
        return float(t_hi), float(p_hi), np.full_like(rp.dz, np.nan, dtype=np.float64)

    p_mid = 0.0
    t_mid = np.inf
    dt_mid = None
    for _ in range(max_iter):
        p_mid = 0.5 * (p_lo + p_hi)
        x_mid, t_mid, dt_inc = _integrate_x_t_from_nodes(v_nodes, rp.dz, p_mid)
        if not np.isfinite(x_mid):
            p_hi = p_mid
            continue
        if abs(x_mid - R) <= max(1e-9, 1e-9 * R):
            dt_mid = dt_inc
            break
        if x_mid < R:
            p_lo = p_mid
        else:
            p_hi = p_mid
        if (p_hi - p_lo) < p_tol * max(1.0, p_hi):
            dt_mid = dt_inc
            break
        dt_mid = dt_inc

    if dt_mid is None:
        dt_mid = np.full_like(rp.dz, np.nan, dtype=np.float64)
    return float(t_mid), float(p_mid), dt_mid


@dataclass
class Layer:
    z_top: float
    z_bot: float
    v: float

    def to_json(self, *, layer_id: str | None = None) -> dict:
        out = {
            "z_top_m": float(self.z_top),
            "z_bot_m": float(self.z_bot),
            "type": "constant",
            "v_mps": float(self.v),
        }
        if layer_id is not None:
            out = {"layer_id": int(layer_id), **out}
        return out


def layers_to_contract_json(phase: str, layers: List[Layer]) -> list[dict]:
    phase_name = str(phase).strip().upper()
    if phase_name not in {"P", "S"}:
        raise ValueError(f"Unsupported phase for layer_id generation: {phase!r}")
    return [
        layer.to_json(layer_id=i)
        for i, layer in enumerate(layers, start=1)
    ]


def init_layers_uniform(z0: float, zmax: float, n_layers: int) -> list[tuple[float, float]]:
    edges = np.linspace(z0, zmax, n_layers + 1, dtype=float)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n_layers)]


def layers_to_staircase(z: np.ndarray, layers: List[Layer]) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    li = 0
    for i, zz in enumerate(z):
        while li < len(layers) - 1 and zz > layers[li].z_bot + 1e-12:
            li += 1
        out[i] = layers[li].v
    return out


def compute_ref_times_and_kernels(
    v_ref: np.ndarray,
    raypaths: List[RayPath],
) -> tuple[Dict[tuple[float, float], float], Dict[tuple[float, float], np.ndarray]]:
    Tref: Dict[tuple[float, float], float] = {}
    dtinc: Dict[tuple[float, float], np.ndarray] = {}
    for rp in raypaths:
        t, _, dt = raytrace_1d_snell_precomputed(v_ref, rp)
        Tref[rp.key()] = float(t)
        dtinc[rp.key()] = dt.astype(np.float64, copy=False)
    return Tref, dtinc


def harmonic_mean_by_kernel(
    layer_bounds: List[tuple[float, float]],
    z: np.ndarray,
    v_ref: np.ndarray,
    raypaths: List[RayPath],
    dtinc: Dict[tuple[float, float], np.ndarray],
    ray_weights: Dict[tuple[float, float], float] | None = None,
) -> List[Layer]:
    edges = np.array([layer_bounds[0][0]] + [b for (_, b) in layer_bounds], dtype=np.float64)
    num = np.zeros(len(layer_bounds), dtype=np.float64)
    den = np.zeros(len(layer_bounds), dtype=np.float64)

    for rp in raypaths:
        key = rp.key()
        w_ray = 1.0 if ray_weights is None else float(ray_weights.get(key, 1.0))
        dt = dtinc[key]
        zm = rp.zmid
        slowness = 1.0 / np.interp(zm, z, v_ref)

        h_num, _ = np.histogram(zm, bins=edges, weights=(w_ray * dt * slowness))
        h_den, _ = np.histogram(zm, bins=edges, weights=(w_ray * dt))
        num += h_num
        den += h_den

    layers: List[Layer] = []
    for i, (a, b) in enumerate(layer_bounds):
        if den[i] <= 0 or num[i] <= 0:
            mref = (z >= a) & (z <= b)
            v0 = float(np.median(v_ref[mref])) if np.any(mref) else float(np.median(v_ref))
        else:
            v0 = 1.0 / (num[i] / den[i])
        layers.append(Layer(a, b, float(v0)))
    return layers


def eval_deltaT_ms(
    v_simp: np.ndarray,
    raypaths: List[RayPath],
    Tref: Dict[tuple[float, float], float],
) -> Dict[tuple[float, float], float]:
    out: Dict[tuple[float, float], float] = {}
    for rp in raypaths:
        ts, _, _ = raytrace_1d_snell_precomputed(v_simp, rp)
        out[rp.key()] = abs(float(ts) - float(Tref[rp.key()])) * 1000.0
    return out


def summarize_deltaT(deltaT_ms: Dict[tuple[float, float], float]) -> dict:
    arr = np.array(list(deltaT_ms.values()), dtype=float)
    if arr.size == 0:
        return {"max_ms": 0.0, "p95_ms": 0.0, "rms_ms": 0.0, "mean_ms": 0.0}
    return {
        "max_ms": float(np.max(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "rms_ms": float(np.sqrt(np.mean(arr * arr))),
        "mean_ms": float(np.mean(arr)),
    }


def choose_layer_to_split(
    layer_bounds: List[tuple[float, float]],
    raypaths: List[RayPath],
    deltaT_ms: Dict[tuple[float, float], float],
    dtinc: Dict[tuple[float, float], np.ndarray],
    top_frac: float = 0.05,
    min_top: int = 50,
) -> int:
    items = sorted(deltaT_ms.items(), key=lambda kv: kv[1], reverse=True)
    k = max(min_top, int(len(items) * top_frac))
    k = min(k, len(items))
    top = items[:k]

    edges = np.array([layer_bounds[0][0]] + [b for (_, b) in layer_bounds], dtype=np.float64)
    scores = np.zeros(len(layer_bounds), dtype=np.float64)

    rp_map = {rp.key(): rp for rp in raypaths}
    for (key, dt_ms) in top:
        rp = rp_map[key]
        zm = rp.zmid
        w = float(dt_ms) * dtinc[key]
        h, _ = np.histogram(zm, bins=edges, weights=w)
        scores += h

    return int(np.argmax(scores))


def weighted_median_split_depth(
    a: float,
    b: float,
    raypaths: List[RayPath],
    deltaT_ms: Dict[tuple[float, float], float],
    dtinc: Dict[tuple[float, float], np.ndarray],
    top_frac: float = 0.05,
    min_top: int = 50,
) -> float:
    items = sorted(deltaT_ms.items(), key=lambda kv: kv[1], reverse=True)
    k = max(min_top, int(len(items) * top_frac))
    k = min(k, len(items))
    top = items[:k]
    rp_map = {rp.key(): rp for rp in raypaths}

    zz_list = []
    ww_list = []
    for (key, dt_ms) in top:
        rp = rp_map[key]
        zm = rp.zmid
        dt = dtinc[key]
        m = (zm >= a) & (zm < b + 1e-12)
        if not np.any(m):
            continue
        zz_list.append(zm[m])
        ww_list.append((float(dt_ms) * dt[m]).astype(np.float64))

    if not zz_list:
        return 0.5 * (a + b)
    zz = np.concatenate(zz_list)
    ww = np.concatenate(ww_list)
    order = np.argsort(zz)
    zz = zz[order]
    ww = ww[order]
    c = np.cumsum(ww)
    if c[-1] <= 0:
        return 0.5 * (a + b)
    j = int(np.searchsorted(c, 0.5 * c[-1]))
    return float(np.clip(zz[min(j, len(zz) - 1)], a, b))


def split_layer(
    layer_bounds: List[tuple[float, float]],
    idx: int,
    z_split: float,
    min_thickness: float,
) -> List[tuple[float, float]]:
    a, b = layer_bounds[idx]
    if (z_split - a) < min_thickness or (b - z_split) < min_thickness:
        z_split = 0.5 * (a + b)
    if (z_split - a) < min_thickness or (b - z_split) < min_thickness:
        raise ValueError(f"Cannot split layer [{a},{b}] with min_thickness={min_thickness}")
    return layer_bounds[:idx] + [(a, z_split), (z_split, b)] + layer_bounds[idx + 1:]


def build_equivalent_layers_for_wave(
    z: np.ndarray,
    v_ref: np.ndarray,
    raypaths: List[RayPath],
    tau_ms: float,
    n_init_layers: int,
    max_layers: int,
    min_thickness_m: float,
    top_frac: float,
    min_top: int,
    verbose: bool = True,
) -> tuple[List[Layer], dict]:
    layer_bounds = init_layers_uniform(float(z[0]), float(z[-1]), int(n_init_layers))

    Tref, dtinc = compute_ref_times_and_kernels(v_ref, raypaths)

    history = []
    for it in range(1, 201):
        layers = harmonic_mean_by_kernel(layer_bounds, z, v_ref, raypaths, dtinc)
        v_simp = layers_to_staircase(z, layers)
        deltaT = eval_deltaT_ms(v_simp, raypaths, Tref)
        metrics = summarize_deltaT(deltaT)
        history.append({"iter": it, "n_layers": len(layer_bounds), **metrics})

        if verbose:
            print(
                f"  iter={it:02d} layers={len(layer_bounds):02d} "
                f"max={metrics['max_ms']:.3f}ms p95={metrics['p95_ms']:.3f}ms rms={metrics['rms_ms']:.3f}ms"
            )

        if metrics["max_ms"] <= tau_ms:
            return layers, {"metrics": metrics, "history": history}

        if len(layer_bounds) >= max_layers:
            raise RuntimeError(
                f"Failed to reach threshold max ΔT <= {tau_ms} ms with max_layers={max_layers}. "
                f"Current max={metrics['max_ms']:.3f} ms"
            )

        idx = choose_layer_to_split(layer_bounds, raypaths, deltaT, dtinc, top_frac=top_frac, min_top=min_top)
        a, b = layer_bounds[idx]
        z_split = weighted_median_split_depth(a, b, raypaths, deltaT, dtinc, top_frac=top_frac, min_top=min_top)
        layer_bounds = split_layer(layer_bounds, idx, z_split, min_thickness=min_thickness_m)

    raise RuntimeError("Layerization did not converge (iteration cap reached)")


def write_equiv_model(
    out_dir: str,
    ref_meta: dict,
    z: np.ndarray,
    layers_P: List[Layer],
    layers_S: List[Layer],
    qc: dict,
) -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    vp = layers_to_staircase(z, layers_P).astype(np.float32)
    vs = layers_to_staircase(z, layers_S).astype(np.float32)
    np.save(outp / "depth.npy", z.astype(np.float32))
    np.save(outp / "vp.npy", vp)
    np.save(outp / "vs.npy", vs)

    meta = {
        "version": "1.0.0",
        "model_type": "1d_layered_equiv_traveltime",
        "z_positive": "down",
        "phases": ["P", "S"],
        "units": ref_meta.get("units", {"length": "m", "velocity": "m/s"}),
        "coordinate_system": ref_meta.get("coordinate_system", {}),
        "grid": ref_meta.get("grid", {}),
        "fields": {"vp": True, "vs": True, "rho": False},
        "files": {
            "format": "npy",
            "dtype": "float32",
            "endian": "little",
            "vp": "vp.npy",
            "vs": "vs.npy",
            "rho": None,
        },
        "equivalence": qc.get("equivalence", {}),
        "layers": {
            "P": layers_to_contract_json("P", layers_P),
            "S": layers_to_contract_json("S", layers_S),
        },
    }
    (outp / "velocity_model_equiv_layered.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (outp / "qc_traveltime_equiv.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")


def load_engineering_1d(
    ref_dir: str,
    *,
    depth_axis: str = "tvd",
    wellpath_path: str | None = None,
    export_csv: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    ref = load_ref_model(ref_dir, depth_axis=depth_axis, wellpath_path=wellpath_path)

    if export_csv:
        try:
            export_ref_tvd_csv(
                ref_dir=ref_dir,
                wellpath_path=wellpath_path,
                out_csv=None,
                include_md_xy=True,
            )
        except Exception as e:
            print(f"[WARN] export_ref_tvd_csv failed: {e}")

    meta = {"ref": ref.meta}
    return meta, ref.z_m.astype(np.float64), ref.vp_mps.astype(np.float64), ref.vs_mps.astype(np.float64)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="layerize_1d_by_traveltime",
        description="Build travel-time-equivalent piecewise-constant 1D velocity model from engineering continuous 1D model.",
    )
    p.add_argument("--ref_dir", required=True, help="Reference engineering model dir")
    p.add_argument("--out_dir", required=True, help="Output dir for equivalent layered model")

    p.add_argument("--ref_depth_axis", type=str, default="tvd", choices=["tvd", "md"], help="Depth axis for reference model.")

    p.add_argument("--z_min", type=float, default=None, help="(Advanced override) Source depth sampling min (m). Default: wellpath TVD min (if available), else ref z0.")
    p.add_argument("--z_max", type=float, default=None, help="(Advanced override) Source depth sampling max (m). Default: ceil(wellpath TVD max) (if available), else ref zmax.")
    p.add_argument("--n_z", type=int, default=81, help="Number of source depth samples along well. Default 81")

    p.add_argument("--r_over_z", type=str, default="0.5,0.8,1.0,1.2,1.5", help="(Mode=ratios) Comma-separated R/z ratios")

    p.add_argument("--stations_csv", type=str, default=None, help="(Recommended) stations.csv path. If provided, R is derived from stations")
    p.add_argument("--src_x_m", type=float, default=0.0, help="(Mode=stations) Source well X (m).")
    p.add_argument("--src_y_m", type=float, default=0.0, help="(Mode=stations) Source well Y (m).")
    p.add_argument("--wellpath_path", type=str, default=None, help="(Mode=stations_wellpath) wellpath CSV or dir containing CSV(s).")
    p.add_argument("--max_stations", type=int, default=500, help="Max stations used for geometry. Default 500")

    p.add_argument("--tau_p_ms", type=float, default=2.0, help="Threshold max ΔT for P (ms). Default 2.0")
    p.add_argument("--tau_s_ms", type=float, default=4.0, help="Threshold max ΔT for S (ms). Default 4.0")

    p.add_argument("--n_init_layers", type=int, default=3, help="Initial number of layers. Default 3")
    p.add_argument("--max_layers", type=int, default=30, help="Maximum layers allowed. Default 30")
    p.add_argument("--min_thickness_m", type=float, default=20.0, help="Minimum layer thickness (m). Default 20m")
    p.add_argument("--top_frac", type=float, default=0.05, help="Top fraction worst rays used to guide splitting. Default 0.05")
    p.add_argument("--min_top", type=int, default=50, help="Minimum worst rays used to guide splitting. Default 50")

    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--quiet", action="store_true", help="Reduce console output")
    p.add_argument("--export_ref_tvd_csv", action="store_true", help="Export <ref_dir>/ref_tvd.csv (+summary json) for delivery/QC.")
    return p


def main(argv: List[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)

    ref_meta, z, vp_ref, vs_ref = load_engineering_1d(
        args.ref_dir,
        depth_axis=args.ref_depth_axis,
        wellpath_path=args.wellpath_path,
        export_csv=bool(args.export_ref_tvd_csv),
    )

    z0 = float(z[0])
    zmax = float(z[-1])

    # -----------------------------
    # z_min / z_max policy + engineering extension (ONLY CHANGE HERE):
    #   - Default: use wellpath TVD range when (ref_depth_axis=tvd AND wellpath_path is provided)
    #   - Advanced override: user-provided --z_min/--z_max take precedence
    #   - If z_max exceeds ref-supported max depth:
    #       * extend the ref grid z to z_max (keep dz, force last == z_max)
    #       * extend vp_ref/vs_ref with endpoint-hold (engineering stable)
    # -----------------------------
    verbose = not bool(args.quiet)

    if args.z_min is None and args.z_max is None and args.ref_depth_axis == "tvd" and args.wellpath_path:
        _, tvd_wp = load_wellpath_md_tvd(args.wellpath_path)
        z_min = float(np.nanmin(tvd_wp))
        z_max = float(np.ceil(float(np.nanmax(tvd_wp))))
    else:
        z_min = z0 if args.z_min is None else float(args.z_min)
        z_max = zmax if args.z_max is None else float(args.z_max)

    # lower bound must be within ref grid support (cannot integrate above z0)
    if z_min < z0:
        raise ValueError(f"Invalid z_min={z_min}. Must satisfy z_min >= ref_z0={z0}")

    # must have non-empty interval (we will handle z_max extension below)
    if z_min >= z_max:
        raise ValueError(f"Invalid z_min/z_max. Must satisfy z_min < z_max. Got z_min={z_min}, z_max={z_max}")

    # extend ref model grid if needed
    if z_max > zmax:
        if z.size < 2:
            raise ValueError("Cannot extend ref grid: ref depth grid has <2 samples.")
        dz0 = float(np.nanmedian(np.diff(z)))
        if not np.isfinite(dz0) or dz0 <= 0:
            raise ValueError(f"Cannot extend ref grid: invalid dz from ref grid, dz={dz0}")

        # build extension points: start at zmax + dz0, stop at z_max, force last exactly equals z_max
        zs_ext = []
        cur = zmax + dz0
        # robust loop: avoid infinite due to float error
        for _ in range(10_000_000):
            if cur >= z_max - 1e-10:
                break
            zs_ext.append(cur)
            cur += dz0
        if not zs_ext or abs(zs_ext[-1] - z_max) > 1e-10:
            zs_ext.append(z_max)
        else:
            zs_ext[-1] = z_max

        z_ext = np.asarray(zs_ext, dtype=np.float64)

        # ensure strict increasing and no duplication with zmax
        if z_ext.size > 0 and z_ext[0] <= zmax + 1e-12:
            z_ext = z_ext[z_ext > zmax + 1e-12]

        if z_ext.size > 0:
            z = np.concatenate([z.astype(np.float64, copy=False), z_ext], axis=0)

            # endpoint-hold extrapolation for velocities (engineering stable)
            vp_last = float(vp_ref[-1])
            vs_last = float(vs_ref[-1])
            vp_ref = np.concatenate([vp_ref.astype(np.float64, copy=False), np.full(z_ext.size, vp_last, dtype=np.float64)], axis=0)
            vs_ref = np.concatenate([vs_ref.astype(np.float64, copy=False), np.full(z_ext.size, vs_last, dtype=np.float64)], axis=0)

            zmax = float(z[-1])

            if verbose:
                print(
                    f"[INFO] z_max={z_max} m exceeds ref max depth. "
                    f"Extended ref grid from {float(z[0])}–{float(z[len(z)-1])} m (dz≈{dz0:g} m), "
                    f"vp/vs endpoint-hold extrapolated."
                )

    # Final sanity: z_max must now be within extended z[-1]
    if z_max > zmax + 1e-9:
        raise ValueError(f"Internal error: z_max={z_max} still > z_grid_max={zmax} after extension.")

    if args.stations_csv and args.wellpath_path:
        rays, geom_meta = make_ray_set_from_stations_wellpath(
            stations_csv=str(args.stations_csv),
            wellpath_path=str(args.wellpath_path),
            z_min=float(z_min),
            z_max=float(z_max),
            n_z=int(args.n_z),
            max_stations=int(args.max_stations),
            seed=int(args.seed),
        )
    elif args.stations_csv:
        rays, geom_meta = make_ray_set_from_stations(
            stations_csv=str(args.stations_csv),
            src_x_m=float(args.src_x_m),
            src_y_m=float(args.src_y_m),
            z_min=float(z_min),
            z_max=float(z_max),
            n_z=int(args.n_z),
            max_stations=int(args.max_stations),
            seed=int(args.seed),
        )
    else:
        r_over_z = [float(s) for s in args.r_over_z.split(",") if s.strip()]
        rays = make_ray_set(z_min, z_max, args.n_z, r_over_z)
        geom_meta = {
            "mode": "ratios",
            "z_min_m": float(z_min),
            "z_max_m": float(z_max),
            "n_z": int(args.n_z),
            "r_over_z": r_over_z,
            "seed": int(args.seed),
        }

    rng = np.random.default_rng(int(args.seed))
    idx = np.arange(len(rays))
    rng.shuffle(idx)
    rays = [rays[i] for i in idx]
    raypaths = build_raypaths(z, rays)

    if verbose:
        print("[RUN] Travel-time-equivalent layerization (piecewise constant, Snell 1D)")
        print(f"  ref_dir : {args.ref_dir}")
        print(f"  out_dir : {args.out_dir}")
        print(f"  geometry: {geom_meta.get('mode')}")
        print(f"  rays    : n={len(rays)}")
        print(f"  thresholds: P={args.tau_p_ms} ms, S={args.tau_s_ms} ms")
        print(f"  layers init={args.n_init_layers}, max={args.max_layers}, min_thickness={args.min_thickness_m} m")

    layers_P, info_P = build_equivalent_layers_for_wave(
        z=z, v_ref=vp_ref, raypaths=raypaths,
        tau_ms=float(args.tau_p_ms),
        n_init_layers=int(args.n_init_layers),
        max_layers=int(args.max_layers),
        min_thickness_m=float(args.min_thickness_m),
        top_frac=float(args.top_frac),
        min_top=int(args.min_top),
        verbose=verbose,
    )
    layers_S, info_S = build_equivalent_layers_for_wave(
        z=z, v_ref=vs_ref, raypaths=raypaths,
        tau_ms=float(args.tau_s_ms),
        n_init_layers=int(args.n_init_layers),
        max_layers=int(args.max_layers),
        min_thickness_m=float(args.min_thickness_m),
        top_frac=float(args.top_frac),
        min_top=int(args.min_top),
        verbose=verbose,
    )

    qc = {
        "equivalence": {"geometry": geom_meta, "threshold_ms": {"P": float(args.tau_p_ms), "S": float(args.tau_s_ms)}},
        "P": info_P,
        "S": info_S,
    }
    write_equiv_model(args.out_dir, ref_meta, z, layers_P, layers_S, qc)

    if verbose:
        outp = Path(args.out_dir)
        print("[OK] Done. Wrote:")
        print(f"  - {outp/'velocity_model_equiv_layered.json'}")
        print(f"  - {outp/'qc_traveltime_equiv.json'}")
        print(f"  - {outp/'depth.npy'} / vp.npy / vs.npy")


if __name__ == "__main__":
    main()