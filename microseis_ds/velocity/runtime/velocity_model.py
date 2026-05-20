# microseis_ds/velocity/runtime/velocity_model.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Literal

import numpy as np


JsonLike = Dict[str, Any]
ArrayLike = Union[float, int, np.ndarray, list]
InterpMethod = Literal["linear", "nearest"]
PhaseName = Literal["P", "S"]

# Engineering QC defaults (tunable)
DEFAULT_DZ_MEAN_MIN_M = 0.05   # 5 cm
DEFAULT_DZ_MEAN_MAX_M = 50.0   # 50 m
DEFAULT_DZ_NONUNIFORM_RATIO_MAX = 10.0  # dz_max / dz_min

try:
    from microseis_ds.velocity.types import (
        VELOCITY_EQUIV_MODEL_TYPE,
        VELOCITY_REQUIRED_PHASES,
        VELOCITY_V1_Z_POSITIVE,
    )
except Exception:  # pragma: no cover - keeps this runtime file importable in isolated checks
    VELOCITY_EQUIV_MODEL_TYPE = "1d_layered_equiv_traveltime"
    VELOCITY_REQUIRED_PHASES = ("P", "S")
    VELOCITY_V1_Z_POSITIVE = "down"


class VelocityModelError(RuntimeError):
    pass


def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _load_json(p: Path) -> JsonLike:
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError as e:
        raise VelocityModelError(f"Invalid JSON file: {p}; {e}") from e
    except OSError as e:
        raise VelocityModelError(f"Failed to read JSON file: {p}; {e}") from e

    if not isinstance(obj, dict):
        raise VelocityModelError(f"JSON root must be an object in {p}, got {type(obj)!r}")
    return obj


def _require(d: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise VelocityModelError(f"Missing key '{key}' in {ctx}")
    return d[key]


def _build_depth_from_grid(meta: Mapping[str, Any]) -> np.ndarray:
    grid = _require(meta, "grid", "velocity_model.json")
    if not isinstance(grid, Mapping):
        raise VelocityModelError("grid must be a mapping in velocity_model.json")

    origin = _require(grid, "origin", "grid")
    spacing = _require(grid, "spacing", "grid")
    shape = _require(grid, "shape", "grid")
    if not isinstance(origin, Mapping):
        raise VelocityModelError("grid.origin must be a mapping")
    if not isinstance(spacing, Mapping):
        raise VelocityModelError("grid.spacing must be a mapping")
    if not isinstance(shape, Mapping):
        raise VelocityModelError("grid.shape must be a mapping")

    z0 = float(_require(origin, "z0", "grid.origin"))
    dz = float(_require(spacing, "dz", "grid.spacing"))
    nz = int(_require(shape, "nz", "grid.shape"))

    if nz <= 0:
        raise VelocityModelError(f"grid.shape.nz must be > 0, got {nz}")
    if dz <= 0:
        raise VelocityModelError(f"grid.spacing.dz must be > 0, got {dz}")

    return z0 + dz * np.arange(nz, dtype=np.float64)


def _is_strictly_monotonic_increasing(x: np.ndarray) -> bool:
    if x.size < 2:
        return True
    dx = np.diff(x)
    return bool(np.all(dx > 0))


def _to_1d_float_array(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise VelocityModelError(f"'{name}' must be 1D array, got shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


def _finite_float(value: Any, name: str, ctx: str) -> float:
    if value in (None, ""):
        raise VelocityModelError(f"Missing key '{name}' in {ctx}")
    try:
        out = float(value)
    except Exception as e:
        raise VelocityModelError(f"{ctx}: {name} must be a finite number, got {value!r}") from e
    if not np.isfinite(out):
        raise VelocityModelError(f"{ctx}: {name} must be finite, got {value!r}")
    return out


def _positive_float(value: Any, name: str, ctx: str) -> float:
    out = _finite_float(value, name, ctx)
    if out <= 0:
        raise VelocityModelError(f"{ctx}: {name} must be > 0, got {out}")
    return out


def _required_int(value: Any, name: str, ctx: str) -> int:
    if value in (None, ""):
        raise VelocityModelError(f"Missing key '{name}' in {ctx}")
    try:
        out = int(value)
    except Exception as e:
        raise VelocityModelError(f"{ctx}: {name} must be an integer, got {value!r}") from e
    return out


def _phase_name(phase: str) -> PhaseName:
    p = str(phase).upper()
    if p not in ("P", "S"):
        raise VelocityModelError(f"Unsupported phase={phase!r}; expected 'P' or 'S'.")
    return p  # type: ignore[return-value]


def _resolve_equiv_json_path(path: Union[str, Path]) -> Tuple[Path, Path]:
    p = _as_path(path)
    if p.is_dir():
        json_path = p / "velocity_model_equiv_layered.json"
        root = p
    else:
        json_path = p
        root = p.parent

    if not json_path.exists():
        raise VelocityModelError(f"velocity_model_equiv_layered.json not found: {json_path}")
    if not json_path.is_file():
        raise VelocityModelError(f"velocity_model_equiv_layered.json path is not a file: {json_path}")
    return json_path, root


@dataclass(frozen=True)
class EquivalentVelocityLayer:
    """One constant-velocity layer in a velocity V1 P/S equivalent layered model."""

    layer_id: int
    z_top_m: float
    z_bot_m: float
    v_mps: float

    def to_json(self) -> JsonLike:
        return {
            "layer_id": int(self.layer_id),
            "z_top_m": float(self.z_top_m),
            "z_bot_m": float(self.z_bot_m),
            "v_mps": float(self.v_mps),
        }


@dataclass
class EquivalentLayeredVelocityModel:
    """
    Runtime reader for velocity V1 travel-time-equivalent layered models.

    Canonical input artifact:
      - velocity_model_equiv_layered.json
      - model_type == '1d_layered_equiv_traveltime'
      - z_positive == 'down'
      - phases includes P and S
      - layers.P / layers.S are non-empty ordered layer lists
      - each layer contains layer_id, z_top_m, z_bot_m, v_mps

    The runtime does not repair, infer, sort, or silently fill missing contract
    fields. Builder-side output completion and full schema reporting remain in
    the build/schema layers.
    """

    meta: JsonLike
    layers: Dict[PhaseName, Tuple[EquivalentVelocityLayer, ...]]
    root_dir: Optional[Path] = None

    @staticmethod
    def load(path: Union[str, Path]) -> "EquivalentLayeredVelocityModel":
        """
        Load from:
          - a directory that contains velocity_model_equiv_layered.json
          - a direct path to velocity_model_equiv_layered.json
        """
        json_path, root = _resolve_equiv_json_path(path)
        meta = _load_json(json_path)
        model = EquivalentLayeredVelocityModel.from_dict(meta, root_dir=root)
        model.validate(strict=True)
        return model

    @staticmethod
    def from_dict(
        meta: Mapping[str, Any],
        root_dir: Optional[Path] = None,
    ) -> "EquivalentLayeredVelocityModel":
        if not isinstance(meta, Mapping):
            raise VelocityModelError(f"equiv layered metadata must be a mapping, got {type(meta)!r}")

        meta_dict: JsonLike = dict(meta)
        layers = _parse_equiv_layers(meta_dict)
        return EquivalentLayeredVelocityModel(meta=meta_dict, layers=layers, root_dir=root_dir)

    def validate(self, strict: bool = True) -> None:
        meta = self.meta

        model_type = _require(meta, "model_type", "velocity_model_equiv_layered.json")
        if model_type != VELOCITY_EQUIV_MODEL_TYPE:
            raise VelocityModelError(
                f"model_type must be {VELOCITY_EQUIV_MODEL_TYPE!r}, got {model_type!r}."
            )

        z_positive = _require(meta, "z_positive", "velocity_model_equiv_layered.json")
        if z_positive != VELOCITY_V1_Z_POSITIVE:
            raise VelocityModelError(
                f"z_positive must be {VELOCITY_V1_Z_POSITIVE!r} for velocity V1, got {z_positive!r}."
            )

        units = _require(meta, "units", "velocity_model_equiv_layered.json")
        if not isinstance(units, Mapping):
            raise VelocityModelError("units must be a mapping in velocity_model_equiv_layered.json.")
        if units.get("length") != "m":
            raise VelocityModelError(f"units.length must be 'm', got {units.get('length')!r}.")
        if units.get("velocity") != "m/s":
            raise VelocityModelError(f"units.velocity must be 'm/s', got {units.get('velocity')!r}.")

        phases = _require(meta, "phases", "velocity_model_equiv_layered.json")
        if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes, bytearray)):
            raise VelocityModelError("phases must be a non-string sequence containing 'P' and 'S'.")
        phase_set = set(phases)
        missing_phases = [phase for phase in VELOCITY_REQUIRED_PHASES if phase not in phase_set]
        if missing_phases:
            raise VelocityModelError(
                f"phases must include {list(VELOCITY_REQUIRED_PHASES)!r}; missing {missing_phases!r}."
            )

        for phase in VELOCITY_REQUIRED_PHASES:
            p = _phase_name(phase)
            phase_layers = self.layers.get(p)
            if not phase_layers:
                raise VelocityModelError(f"layers.{p} must be a non-empty layer sequence.")
            _validate_layer_sequence(p, phase_layers, strict=strict)

        if strict:
            p_range = self.depth_range("P")
            s_range = self.depth_range("S")
            if abs(p_range[0] - s_range[0]) > 1e-9 or abs(p_range[1] - s_range[1]) > 1e-9:
                raise VelocityModelError(
                    "P/S layer depth ranges must match: "
                    f"P={p_range}, S={s_range}."
                )

    def get_layers(self, phase: PhaseName | str) -> Tuple[EquivalentVelocityLayer, ...]:
        p = _phase_name(phase)
        return self.layers[p]

    def depth_range(self, phase: PhaseName | str = "P") -> Tuple[float, float]:
        p = _phase_name(phase)
        phase_layers = self.layers[p]
        return float(phase_layers[0].z_top_m), float(phase_layers[-1].z_bot_m)

    def query_phase(
        self,
        z_m: ArrayLike,
        phase: PhaseName | str,
        *,
        extrapolate: bool = False,
    ) -> np.ndarray:
        """
        Query piecewise-constant velocity for one phase.

        By default, out-of-range depth queries fail. Set extrapolate=True only
        when the caller explicitly accepts endpoint-hold behavior.
        """
        p = _phase_name(phase)
        phase_layers = self.layers[p]

        z = np.asarray(z_m, dtype=np.float64)
        z_shape = z.shape
        z_flat = z.reshape(-1)

        if z_flat.size == 0:
            return np.asarray([], dtype=np.float64).reshape(z_shape)
        if not np.all(np.isfinite(z_flat)):
            raise VelocityModelError(f"Depth query contains non-finite value for phase {p}.")

        zmin, zmax = self.depth_range(p)
        if extrapolate:
            zq = np.clip(z_flat, zmin, zmax)
        else:
            below = z_flat < zmin
            above = z_flat > zmax
            if np.any(below) or np.any(above):
                raise VelocityModelError(
                    f"Depth query out of range for phase {p}: "
                    f"valid=[{zmin}, {zmax}], "
                    f"query_min={float(np.min(z_flat))}, query_max={float(np.max(z_flat))}."
                )
            zq = z_flat

        bottoms = np.array([layer.z_bot_m for layer in phase_layers], dtype=np.float64)
        values = np.array([layer.v_mps for layer in phase_layers], dtype=np.float64)
        idx = np.searchsorted(bottoms, zq, side="right")
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx].reshape(z_shape)

    def query(
        self,
        z_m: ArrayLike,
        *,
        extrapolate: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (vp_mps, vs_mps) for depth z_m."""
        vp = self.query_phase(z_m, "P", extrapolate=extrapolate)
        vs = self.query_phase(z_m, "S", extrapolate=extrapolate)
        return vp, vs

    def to_layer_dict(self) -> Dict[str, List[JsonLike]]:
        return {
            "P": [layer.to_json() for layer in self.layers["P"]],
            "S": [layer.to_json() for layer in self.layers["S"]],
        }

    def to_metadata(self) -> JsonLike:
        meta = dict(self.meta)
        meta["layers"] = self.to_layer_dict()
        return meta

    def summary(self) -> str:
        p0, p1 = self.depth_range("P")
        n_p = len(self.layers["P"])
        n_s = len(self.layers["S"])
        vp_vals = np.array([layer.v_mps for layer in self.layers["P"]], dtype=np.float64)
        vs_vals = np.array([layer.v_mps for layer in self.layers["S"]], dtype=np.float64)
        return (
            "EquivalentLayeredVelocityModel("
            f"model_type={self.meta.get('model_type')}, "
            f"z=[{p0:.3f}, {p1:.3f}] m, "
            f"layers.P={n_p}, layers.S={n_s}, "
            f"vp=[{float(vp_vals.min()):.3f}, {float(vp_vals.max()):.3f}] m/s, "
            f"vs=[{float(vs_vals.min()):.3f}, {float(vs_vals.max()):.3f}] m/s)"
        )


VelocityEquivLayeredModel = EquivalentLayeredVelocityModel


def load_equiv_layered_velocity_model(path: Union[str, Path]) -> EquivalentLayeredVelocityModel:
    """Load velocity_model_equiv_layered.json as a strict velocity V1 runtime model."""
    return EquivalentLayeredVelocityModel.load(path)


def _parse_equiv_layers(meta: Mapping[str, Any]) -> Dict[PhaseName, Tuple[EquivalentVelocityLayer, ...]]:
    layers_obj = _require(meta, "layers", "velocity_model_equiv_layered.json")
    if not isinstance(layers_obj, Mapping):
        if any(k in meta for k in ("velocity", "v", "velocity_mps")):
            raise VelocityModelError("single-velocity fallback is not allowed; require layers.P and layers.S.")
        raise VelocityModelError("layers must be a mapping with keys 'P' and 'S'.")

    parsed: Dict[PhaseName, Tuple[EquivalentVelocityLayer, ...]] = {}
    for phase in VELOCITY_REQUIRED_PHASES:
        p = _phase_name(phase)
        phase_rows = layers_obj.get(p)
        if phase_rows is None:
            raise VelocityModelError(f"Missing required phase layer list layers.{p}.")
        if not isinstance(phase_rows, Sequence) or isinstance(phase_rows, (str, bytes, bytearray)):
            raise VelocityModelError(f"layers.{p} must be a non-string sequence.")
        if len(phase_rows) == 0:
            raise VelocityModelError(f"layers.{p} must be a non-empty sequence.")
        parsed[p] = tuple(_parse_phase_layer_row(row, p, idx) for idx, row in enumerate(phase_rows))
    return parsed


def _parse_phase_layer_row(row: Any, phase: PhaseName, idx: int) -> EquivalentVelocityLayer:
    if not isinstance(row, Mapping):
        raise VelocityModelError(f"layers.{phase}[{idx}] must be a mapping.")
    ctx = f"layers.{phase}[{idx}]"

    layer_id = _required_int(_require(row, "layer_id", ctx), "layer_id", ctx)
    z_top = _finite_float(_require(row, "z_top_m", ctx), "z_top_m", ctx)
    z_bot = _finite_float(_require(row, "z_bot_m", ctx), "z_bot_m", ctx)
    v_mps = _positive_float(_require(row, "v_mps", ctx), "v_mps", ctx)

    if z_bot <= z_top:
        raise VelocityModelError(f"{ctx}: z_bot_m must be greater than z_top_m.")

    return EquivalentVelocityLayer(layer_id=layer_id, z_top_m=z_top, z_bot_m=z_bot, v_mps=v_mps)


def _validate_layer_sequence(
    phase: str,
    layers: Sequence[EquivalentVelocityLayer],
    *,
    strict: bool = True,
) -> None:
    if len(layers) == 0:
        raise VelocityModelError(f"layers.{phase} must not be empty.")

    seen_ids = set()
    prev_bot: Optional[float] = None
    for idx, layer in enumerate(layers):
        ctx = f"layers.{phase}[{idx}]"
        if layer.layer_id in seen_ids:
            raise VelocityModelError(f"{ctx}: duplicate layer_id={layer.layer_id}.")
        seen_ids.add(layer.layer_id)

        if not np.isfinite(layer.z_top_m) or not np.isfinite(layer.z_bot_m):
            raise VelocityModelError(f"{ctx}: z_top_m/z_bot_m must be finite.")
        if not np.isfinite(layer.v_mps) or layer.v_mps <= 0:
            raise VelocityModelError(f"{ctx}: v_mps must be finite and > 0.")
        if layer.z_bot_m <= layer.z_top_m:
            raise VelocityModelError(f"{ctx}: z_bot_m must be greater than z_top_m.")

        if prev_bot is not None:
            if layer.z_top_m < prev_bot - 1e-9:
                raise VelocityModelError(f"{ctx}: layer overlaps previous layer.")
            if strict and abs(layer.z_top_m - prev_bot) > 1e-9:
                raise VelocityModelError(
                    f"{ctx}: layer sequence must be contiguous; "
                    f"previous z_bot_m={prev_bot}, current z_top_m={layer.z_top_m}."
                )
        prev_bot = layer.z_bot_m


@dataclass
class VelocityModel:
    """
    Canonical sampled velocity model used across MicroSeis-DS.

    Supports:
      - model_type = '1d_layered' with files depth.npy/vp.npy/vs.npy
      - model_type = '1d_continuous_engineering' with files depth.npy/vp.npy/vs.npy
      - model_type = 'homogeneous_1d' described by grid + parameters

    Primary API:
      - load(path)
      - validate()
      - query(z)

    Travel-time-equivalent layered models are handled by
    EquivalentLayeredVelocityModel in the same runtime module.
    """

    meta: JsonLike
    depth_m: np.ndarray  # shape (nz,)
    vp_mps: np.ndarray   # shape (nz,)
    vs_mps: np.ndarray   # shape (nz,)
    root_dir: Optional[Path] = None

    @staticmethod
    def load(path: Union[str, Path]) -> "VelocityModel":
        """
        Load from:
          - a directory that contains velocity_model.json
          - a direct path to velocity_model.json
        """
        p = _as_path(path)
        if p.is_dir():
            json_path = p / "velocity_model.json"
            root = p
        else:
            json_path = p
            root = p.parent

        if not json_path.exists():
            raise VelocityModelError(f"velocity_model.json not found: {json_path}")

        meta = _load_json(json_path)

        model_type = meta.get("model_type", None)

        if model_type not in ("1d_layered", "1d_continuous_engineering", "homogeneous_1d"):
            raise VelocityModelError(
                "Unsupported model_type={!r}. Currently supported: "
                "'1d_layered', '1d_continuous_engineering', 'homogeneous_1d'.".format(model_type)
            )

        # ---- homogeneous_1d: synthesize arrays from grid + parameters ----
        if model_type == "homogeneous_1d":
            depth = _build_depth_from_grid(meta)

            params = meta.get("parameters", {}) or meta.get("params", {}) or meta.get("homogeneous", {})
            if not isinstance(params, dict):
                raise VelocityModelError(
                    "homogeneous_1d expects dict in meta['parameters'] (or params/homogeneous)."
                )

            vp0 = params.get("vp0_mps", params.get("vp0", None))
            if vp0 is None:
                raise VelocityModelError("homogeneous_1d requires parameters.vp0_mps (or vp0).")
            vp0 = float(vp0)

            vs0 = params.get("vs0_mps", params.get("vs0", None))
            if vs0 is None:
                ratio = float(params.get("vp_vs_ratio", params.get("vp_over_vs", 1.732)))
                if ratio <= 0:
                    raise VelocityModelError(f"vp_vs_ratio must be > 0, got {ratio}")
                vs0 = vp0 / ratio
            else:
                vs0 = float(vs0)

            vp = np.full_like(depth, vp0, dtype=np.float64)
            vs = np.full_like(depth, vs0, dtype=np.float64)

            vm = VelocityModel(meta=meta, depth_m=depth, vp_mps=vp, vs_mps=vs, root_dir=root)
            vm.validate(strict=True)
            return vm

        # ---- 1d_layered / 1d_continuous_engineering from npy files ----
        files = _require(meta, "files", "velocity_model.json")
        if not isinstance(files, Mapping):
            raise VelocityModelError("files must be a mapping in velocity_model.json")
        fmt = files.get("format", None)
        if fmt != "npy":
            raise VelocityModelError(f"Unsupported files.format={fmt!r}, expected 'npy'.")

        vp_file = files.get("vp", None)
        vs_file = files.get("vs", None)
        if not vp_file or not vs_file:
            raise VelocityModelError(
                f"files.vp / files.vs must be provided for model_type={model_type!r} (npy format)."
            )

        depth_path = root / "depth.npy"
        vp_path = root / str(vp_file)
        vs_path = root / str(vs_file)

        if not depth_path.exists():
            raise VelocityModelError(f"Missing depth.npy: {depth_path}")
        if not vp_path.exists():
            raise VelocityModelError(f"Missing vp file: {vp_path}")
        if not vs_path.exists():
            raise VelocityModelError(f"Missing vs file: {vs_path}")

        depth = np.load(depth_path)
        vp = np.load(vp_path)
        vs = np.load(vs_path)

        depth = _to_1d_float_array(depth, "depth")
        vp = _to_1d_float_array(vp, "vp")
        vs = _to_1d_float_array(vs, "vs")

        vm = VelocityModel(meta=meta, depth_m=depth, vp_mps=vp, vs_mps=vs, root_dir=root)
        vm.validate(strict=True)
        return vm

    def save(
        self,
        out_dir: Union[str, Path],
        dtype: Literal["float32", "float64"] = "float32",
        write_arrays: bool = True,
    ) -> Path:
        """
        Save model to out_dir in the canonical format:
          - depth.npy / vp.npy / vs.npy
          - velocity_model.json
        """
        outp = _as_path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)

        np_dtype = np.float32 if dtype == "float32" else np.float64

        if write_arrays:
            np.save(outp / "depth.npy", self.depth_m.astype(np_dtype))
            np.save(outp / "vp.npy", self.vp_mps.astype(np_dtype))
            np.save(outp / "vs.npy", self.vs_mps.astype(np_dtype))

        meta = dict(self.meta)
        meta.setdefault("files", {})

        if meta.get("model_type") == "homogeneous_1d":
            meta.setdefault("parameters", {})
            if not isinstance(meta["parameters"], dict):
                meta["parameters"] = {}
            vp0 = float(self.vp_mps[0]) if self.vp_mps.size else 0.0
            vs0 = float(self.vs_mps[0]) if self.vs_mps.size else 0.0
            meta["parameters"].setdefault("vp0_mps", vp0)
            meta["parameters"].setdefault("vs0_mps", vs0)
            if vs0 > 0:
                meta["parameters"].setdefault("vp_vs_ratio", vp0 / vs0)

        meta["files"] = dict(meta["files"])
        meta["files"].update(
            {
                "format": "npy" if write_arrays else meta["files"].get("format", "inline"),
                "dtype": dtype if write_arrays else meta["files"].get("dtype", None),
                "endian": "little" if write_arrays else meta["files"].get("endian", None),
                "vp": "vp.npy" if write_arrays else meta["files"].get("vp", None),
                "vs": "vs.npy" if write_arrays else meta["files"].get("vs", None),
                "rho": meta["files"].get("rho", None),
            }
        )

        if meta.get("model_type") == "homogeneous_1d" and not write_arrays:
            meta.pop("files", None)

        dz_mean = float(np.mean(np.diff(self.depth_m))) if len(self.depth_m) > 1 else 0.0
        meta.setdefault("grid", {})
        meta["grid"] = dict(meta["grid"])
        meta["grid"].setdefault("origin", {})
        meta["grid"].setdefault("spacing", {})
        meta["grid"].setdefault("shape", {})
        meta["grid"]["origin"] = dict(meta["grid"]["origin"])
        meta["grid"]["spacing"] = dict(meta["grid"]["spacing"])
        meta["grid"]["shape"] = dict(meta["grid"]["shape"])
        meta["grid"]["origin"]["z0"] = float(self.depth_m[0]) if len(self.depth_m) else 0.0
        meta["grid"]["spacing"]["dz"] = dz_mean
        meta["grid"]["shape"]["nz"] = int(len(self.depth_m))
        meta["grid"]["shape"].setdefault("nx", 1)
        meta["grid"]["shape"].setdefault("ny", 1)

        with open(outp / "velocity_model.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return outp / "velocity_model.json"

    def validate(self, strict: bool = True) -> None:
        meta = self.meta

        _require(meta, "version", "velocity_model.json")
        _require(meta, "model_type", "velocity_model.json")
        _require(meta, "units", "velocity_model.json")
        _require(meta, "grid", "velocity_model.json")

        model_type = meta.get("model_type", None)
        if model_type in ("1d_layered", "1d_continuous_engineering"):
            _require(meta, "files", "velocity_model.json")
        elif model_type == "homogeneous_1d":
            pass
        else:
            raise VelocityModelError(
                "model_type must be '1d_layered' or '1d_continuous_engineering' or 'homogeneous_1d', got {!r}".format(
                    model_type
                )
            )

        _require(meta, "fields", "velocity_model.json")
        _require(meta, "provenance", "velocity_model.json")

        n = int(self.depth_m.size)
        if self.vp_mps.size != n or self.vs_mps.size != n:
            raise VelocityModelError(
                f"Array length mismatch: depth={n}, vp={self.vp_mps.size}, vs={self.vs_mps.size}"
            )

        for name, arr in [("depth_m", self.depth_m), ("vp_mps", self.vp_mps), ("vs_mps", self.vs_mps)]:
            if not np.all(np.isfinite(arr)):
                raise VelocityModelError(f"{name} contains non-finite values (NaN/Inf).")

        if n >= 2 and not _is_strictly_monotonic_increasing(self.depth_m):
            raise VelocityModelError("depth_m must be strictly increasing (monotonic).")

        # dz engineering QC (allow non-uniform, but check reasonableness)
        if n >= 2:
            dz = np.diff(self.depth_m)
            dz_min = float(np.min(dz))
            dz_max = float(np.max(dz))
            dz_mean = float(np.mean(dz))

            qc = meta.get("qc", {}) if isinstance(meta.get("qc", {}), dict) else {}
            dz_mean_min = float(qc.get("dz_mean_min_m", DEFAULT_DZ_MEAN_MIN_M))
            dz_mean_max = float(qc.get("dz_mean_max_m", DEFAULT_DZ_MEAN_MAX_M))
            nonuniform_ratio_max = float(qc.get("dz_nonuniform_ratio_max", DEFAULT_DZ_NONUNIFORM_RATIO_MAX))

            if dz_min <= 0:
                raise VelocityModelError("Non-positive dz detected in depth sampling.")

            if dz_mean < dz_mean_min and strict:
                raise VelocityModelError(
                    f"dz_mean={dz_mean:.6g} m is too small (< {dz_mean_min} m). "
                    "Check units or depth sampling."
                )
            if dz_mean > dz_mean_max and strict:
                raise VelocityModelError(
                    f"dz_mean={dz_mean:.6g} m is too large (> {dz_mean_max} m). "
                    "Model may be too coarse."
                )

            nonuniform_ratio = dz_max / dz_min
            if nonuniform_ratio > nonuniform_ratio_max and strict:
                raise VelocityModelError(
                    f"Depth sampling is highly non-uniform: dz_max/dz_min={nonuniform_ratio:.3g} "
                    f"(> {nonuniform_ratio_max})."
                )

        if np.any(self.vp_mps <= 0) or np.any(self.vs_mps <= 0):
            raise VelocityModelError("vp_mps and vs_mps must be > 0.")

        grid = meta.get("grid", {})
        origin = grid.get("origin", {}) if isinstance(grid, dict) else {}
        shape = grid.get("shape", {}) if isinstance(grid, dict) else {}
        z0 = origin.get("z0", None) if isinstance(origin, dict) else None
        nz = shape.get("nz", None) if isinstance(shape, dict) else None

        if z0 is not None:
            if abs(float(z0) - float(self.depth_m[0])) > 1e-6:
                msg = f"grid.origin.z0 ({z0}) != depth[0] ({self.depth_m[0]})."
                if strict:
                    raise VelocityModelError(msg)

        if nz is not None:
            if int(nz) != n:
                msg = f"grid.shape.nz ({nz}) != len(depth) ({n})."
                if strict:
                    raise VelocityModelError(msg)

    def query(
        self,
        z_m: ArrayLike,
        method: InterpMethod = "linear",
        extrapolate: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z_m, dtype=np.float64)
        z_shape = z.shape
        z_flat = z.reshape(-1)

        zmin = float(self.depth_m[0])
        zmax = float(self.depth_m[-1])

        zq = np.clip(z_flat, zmin, zmax) if not extrapolate else z_flat

        if method == "nearest":
            idx = np.searchsorted(self.depth_m, zq, side="left")
            idx = np.clip(idx, 0, len(self.depth_m) - 1)
            idx0 = np.clip(idx - 1, 0, len(self.depth_m) - 1)
            choose_left = (np.abs(zq - self.depth_m[idx0]) <= np.abs(zq - self.depth_m[idx]))
            idx_final = np.where(choose_left, idx0, idx)
            vp = self.vp_mps[idx_final]
            vs = self.vs_mps[idx_final]
        elif method == "linear":
            vp = np.interp(zq, self.depth_m, self.vp_mps)
            vs = np.interp(zq, self.depth_m, self.vs_mps)
        else:
            raise VelocityModelError(f"Unsupported interpolation method: {method!r}")

        return vp.reshape(z_shape), vs.reshape(z_shape)

    def depth_range(self) -> Tuple[float, float]:
        return float(self.depth_m[0]), float(self.depth_m[-1])

    def dz_mean(self) -> float:
        if self.depth_m.size < 2:
            return 0.0
        return float(np.mean(np.diff(self.depth_m)))

    def summary(self) -> str:
        z0, z1 = self.depth_range()
        return (
            f"VelocityModel(model_type={self.meta.get('model_type')}, "
            f"nz={self.depth_m.size}, z=[{z0:.3f}, {z1:.3f}] m, "
            f"dz_mean={self.dz_mean():.6f} m, "
            f"vp=[{float(self.vp_mps.min()):.3f}, {float(self.vp_mps.max()):.3f}] m/s, "
            f"vs=[{float(self.vs_mps.min()):.3f}, {float(self.vs_mps.max()):.3f}] m/s)"
        )
