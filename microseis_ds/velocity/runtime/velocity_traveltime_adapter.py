# microseis_ds/velocity/runtime/velocity_traveltime_adapter.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from microseis_ds.velocity.runtime.velocity_model import (
    EquivalentLayeredVelocityModel,
    VelocityModelError,
    load_equiv_layered_velocity_model,
)


JsonDict = Dict[str, Any]
PathLike = Union[str, Path]
ArrayLike = Union[float, int, np.ndarray, list]

MAIN_JSON_FILENAME = "velocity_model_equiv_layered.json"
DEPTH_FILENAME = "depth.npy"
VP_FILENAME = "vp.npy"
VS_FILENAME = "vs.npy"
ASSET_HASHES_FILENAME = "asset_hashes.json"
QC_TRAVELTIME_EQUIV_FILENAME = "qc_traveltime_equiv.json"
INTERFACE_NOTE_FILENAME = "interface_note.md"

TRAVELTIME_READY_REQUIRED_FILES: Tuple[str, ...] = (
    MAIN_JSON_FILENAME,
    DEPTH_FILENAME,
    VP_FILENAME,
    VS_FILENAME,
    ASSET_HASHES_FILENAME,
)


class TraveltimeReadyAdapterError(RuntimeError):
    """Raised when a traveltime_ready directory is not consumable."""


def _as_path(value: PathLike) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _load_json(path: Path) -> JsonDict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TraveltimeReadyAdapterError(f"Invalid JSON file: {path}; {exc}") from exc
    except OSError as exc:
        raise TraveltimeReadyAdapterError(f"Failed to read JSON file: {path}; {exc}") from exc
    if not isinstance(obj, dict):
        raise TraveltimeReadyAdapterError(f"JSON root must be an object: {path}")
    return obj


def _compute_sha256(path: Path) -> str:
    if not path.is_file():
        raise TraveltimeReadyAdapterError(f"cannot hash missing asset: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_1d_array(path: Path, name: str) -> np.ndarray:
    try:
        arr = np.load(path)
    except Exception as exc:
        raise TraveltimeReadyAdapterError(f"Failed to load {name}: {path}; {exc}") from exc
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise TraveltimeReadyAdapterError(f"{name} must be a 1D array, got shape={arr.shape}")
    if arr.size == 0:
        raise TraveltimeReadyAdapterError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise TraveltimeReadyAdapterError(f"{name} contains non-finite values")
    return arr


def _validate_depth_vp_vs(depth: np.ndarray, vp: np.ndarray, vs: np.ndarray) -> None:
    if not (len(depth) == len(vp) == len(vs)):
        raise TraveltimeReadyAdapterError(
            f"depth/vp/vs lengths must match; got depth={len(depth)}, vp={len(vp)}, vs={len(vs)}"
        )
    if len(depth) > 1 and not np.all(np.diff(depth) > 0):
        raise TraveltimeReadyAdapterError("depth must be strictly increasing")
    if not np.all(vp > 0):
        raise TraveltimeReadyAdapterError("vp must be strictly positive")
    if not np.all(vs > 0):
        raise TraveltimeReadyAdapterError("vs must be strictly positive")


def _verify_required_files(ready_dir: Path) -> None:
    missing = [name for name in TRAVELTIME_READY_REQUIRED_FILES if not (ready_dir / name).is_file()]
    if missing:
        raise TraveltimeReadyAdapterError(f"traveltime_ready missing required files: {missing}")


def _verify_asset_hashes(ready_dir: Path, asset_hashes: Mapping[str, Any]) -> None:
    if asset_hashes.get("algorithm") != "sha256":
        raise TraveltimeReadyAdapterError("asset_hashes.json algorithm must be sha256")
    items = asset_hashes.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise TraveltimeReadyAdapterError("asset_hashes.items must be a list")
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TraveltimeReadyAdapterError(f"asset_hashes.items[{idx}] must be an object")
        rel = str(item.get("path", ""))
        if not rel or rel.startswith("/") or ".." in Path(rel).parts or "\\" in rel:
            raise TraveltimeReadyAdapterError(f"invalid relative hash path in asset_hashes: {rel!r}")
        path = ready_dir / Path(*rel.split("/"))
        expected = str(item.get("sha256", ""))
        actual = _compute_sha256(path)
        if actual != expected:
            raise TraveltimeReadyAdapterError(f"sha256 mismatch for {rel}: expected={expected}, actual={actual}")
        if "size_bytes" in item and int(item["size_bytes"]) != int(path.stat().st_size):
            raise TraveltimeReadyAdapterError(f"size_bytes mismatch for {rel}")


@dataclass(frozen=True)
class TraveltimeReadyVelocityView:
    """Minimum traveltime-facing runtime view of a velocity traveltime_ready directory."""

    ready_dir: Path
    model: EquivalentLayeredVelocityModel
    depth_m: np.ndarray
    vp_mps: np.ndarray
    vs_mps: np.ndarray
    asset_hashes: JsonDict
    qc: Optional[JsonDict] = None

    def query_phase(self, z_m: ArrayLike, phase: str, *, extrapolate: bool = False) -> np.ndarray:
        """Query P or S velocity from the authoritative layered model."""
        return self.model.query_phase(z_m, phase, extrapolate=extrapolate)

    def query(self, z_m: ArrayLike, *, extrapolate: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Return (vp_mps, vs_mps) for depth z_m."""
        return self.model.query(z_m, extrapolate=extrapolate)

    def to_traveltime_dict(self) -> JsonDict:
        """Return a small serializable descriptor that downstream traveltime code can inspect."""
        return {
            "ready_dir": str(self.ready_dir),
            "main_json": str(self.ready_dir / MAIN_JSON_FILENAME),
            "depth_npy": str(self.ready_dir / DEPTH_FILENAME),
            "vp_npy": str(self.ready_dir / VP_FILENAME),
            "vs_npy": str(self.ready_dir / VS_FILENAME),
            "z_positive": self.model.meta.get("z_positive"),
            "phases": list(self.model.meta.get("phases", [])),
            "sample_count": int(self.depth_m.size),
            "depth_range_m": [float(np.min(self.depth_m)), float(np.max(self.depth_m))],
            "vp_range_mps": [float(np.min(self.vp_mps)), float(np.max(self.vp_mps))],
            "vs_range_mps": [float(np.min(self.vs_mps)), float(np.max(self.vs_mps))],
        }


def load_traveltime_ready_velocity(
    ready_dir: PathLike,
    *,
    verify_hashes: bool = True,
) -> TraveltimeReadyVelocityView:
    """Load and validate a velocity/traveltime_ready directory for downstream consumption."""
    root = _as_path(ready_dir)
    if not root.is_dir():
        raise TraveltimeReadyAdapterError(f"traveltime_ready directory not found: {root}")
    _verify_required_files(root)

    try:
        model = load_equiv_layered_velocity_model(root / MAIN_JSON_FILENAME)
    except VelocityModelError as exc:
        raise TraveltimeReadyAdapterError(f"invalid {MAIN_JSON_FILENAME}: {exc}") from exc

    depth = _load_1d_array(root / DEPTH_FILENAME, DEPTH_FILENAME)
    vp = _load_1d_array(root / VP_FILENAME, VP_FILENAME)
    vs = _load_1d_array(root / VS_FILENAME, VS_FILENAME)
    _validate_depth_vp_vs(depth, vp, vs)

    asset_hashes = _load_json(root / ASSET_HASHES_FILENAME)
    if verify_hashes:
        _verify_asset_hashes(root, asset_hashes)

    qc: Optional[JsonDict] = None
    qc_path = root / QC_TRAVELTIME_EQUIV_FILENAME
    if qc_path.is_file():
        qc = _load_json(qc_path)

    return TraveltimeReadyVelocityView(
        ready_dir=root,
        model=model,
        depth_m=depth,
        vp_mps=vp,
        vs_mps=vs,
        asset_hashes=asset_hashes,
        qc=qc,
    )


# Backward-friendly alias for callers that prefer "adapter" terminology.
load_velocity_traveltime_adapter = load_traveltime_ready_velocity
