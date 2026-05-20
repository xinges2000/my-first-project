# microseis_ds/velocity/build/build_traveltime_ready.py
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from microseis_ds.velocity.runtime.velocity_model import (
    EquivalentLayeredVelocityModel,
    VelocityModelError,
    load_equiv_layered_velocity_model,
)


JsonDict = Dict[str, Any]
PathLike = Union[str, Path]

SCHEMA_VERSION = "1.0.0"
MODULE_NAME_DEFAULT = "velocity"
EQUIV_LAYERED_DIRNAME = "equiv_layered"
TRAVELTIME_READY_DIRNAME = "traveltime_ready"
HASH_ALGORITHM = "sha256"

MAIN_JSON_FILENAME = "velocity_model_equiv_layered.json"
DEPTH_FILENAME = "depth.npy"
VP_FILENAME = "vp.npy"
VS_FILENAME = "vs.npy"
ASSET_HASHES_FILENAME = "asset_hashes.json"
INTERFACE_NOTE_FILENAME = "interface_note.md"
QC_TRAVELTIME_EQUIV_FILENAME = "qc_traveltime_equiv.json"

ARRAY_FILENAMES: Tuple[str, str, str] = (DEPTH_FILENAME, VP_FILENAME, VS_FILENAME)
CORE_READY_FILENAMES: Tuple[str, ...] = (
    MAIN_JSON_FILENAME,
    DEPTH_FILENAME,
    VP_FILENAME,
    VS_FILENAME,
    ASSET_HASHES_FILENAME,
    INTERFACE_NOTE_FILENAME,
    QC_TRAVELTIME_EQUIV_FILENAME,
)
HASHED_READY_FILENAMES: Tuple[str, ...] = (
    MAIN_JSON_FILENAME,
    DEPTH_FILENAME,
    VP_FILENAME,
    VS_FILENAME,
    INTERFACE_NOTE_FILENAME,
    QC_TRAVELTIME_EQUIV_FILENAME,
)


class TraveltimeReadyBuildError(RuntimeError):
    """Raised when traveltime_ready generation cannot produce a valid handoff."""


def _as_path(value: PathLike) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_velocity_dir(out_root: PathLike, module_name: str, dirname: str) -> Path:
    return _as_path(out_root) / module_name / dirname


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def _to_posix_relative(path: Path, base_dir: Path) -> str:
    p = _safe_resolve(path)
    b = _safe_resolve(base_dir)
    try:
        rel = p.relative_to(b)
    except ValueError as exc:
        raise TraveltimeReadyBuildError(
            f"ready asset is outside ready_dir: path={p}, ready_dir={b}"
        ) from exc
    return rel.as_posix()


def _load_json_file(path: Path) -> JsonDict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TraveltimeReadyBuildError(f"Invalid JSON file: {path}; {exc}") from exc
    except OSError as exc:
        raise TraveltimeReadyBuildError(f"Failed to read JSON file: {path}; {exc}") from exc
    if not isinstance(obj, dict):
        raise TraveltimeReadyBuildError(f"JSON root must be an object: {path}")
    return obj


def _write_json_file(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _compute_sha256(path: Path) -> str:
    if not path.is_file():
        raise TraveltimeReadyBuildError(f"cannot hash missing ready asset: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise TraveltimeReadyBuildError(f"source file does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _validate_array_triplet(depth: np.ndarray, vp: np.ndarray, vs: np.ndarray) -> None:
    if depth.ndim != 1 or vp.ndim != 1 or vs.ndim != 1:
        raise TraveltimeReadyBuildError(
            f"depth/vp/vs must all be 1D arrays; got depth={depth.shape}, vp={vp.shape}, vs={vs.shape}"
        )
    if not (len(depth) == len(vp) == len(vs)):
        raise TraveltimeReadyBuildError(
            f"depth/vp/vs lengths must match; got depth={len(depth)}, vp={len(vp)}, vs={len(vs)}"
        )
    if len(depth) == 0:
        raise TraveltimeReadyBuildError("depth/vp/vs arrays must not be empty")
    if not np.all(np.isfinite(depth)):
        raise TraveltimeReadyBuildError("depth array contains non-finite values")
    if not np.all(np.isfinite(vp)) or not np.all(vp > 0):
        raise TraveltimeReadyBuildError("vp array must be finite and strictly positive")
    if not np.all(np.isfinite(vs)) or not np.all(vs > 0):
        raise TraveltimeReadyBuildError("vs array must be finite and strictly positive")
    if len(depth) > 1 and not np.all(np.diff(depth) > 0):
        raise TraveltimeReadyBuildError("depth array must be strictly increasing")


def _load_source_arrays(equiv_layered_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.load(equiv_layered_dir / DEPTH_FILENAME)
    vp = np.load(equiv_layered_dir / VP_FILENAME)
    vs = np.load(equiv_layered_dir / VS_FILENAME)
    depth = np.asarray(depth, dtype=np.float64)
    vp = np.asarray(vp, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    _validate_array_triplet(depth, vp, vs)
    return depth, vp, vs


def _layer_boundaries(model: EquivalentLayeredVelocityModel) -> np.ndarray:
    values: List[float] = []
    for phase in ("P", "S"):
        for layer in model.get_layers(phase):
            values.append(float(layer.z_top_m))
            values.append(float(layer.z_bot_m))
    if not values:
        raise TraveltimeReadyBuildError("equiv layered model has no P/S layer boundaries")
    boundaries = np.array(sorted(set(values)), dtype=np.float64)
    if boundaries.ndim != 1 or boundaries.size == 0:
        raise TraveltimeReadyBuildError("failed to derive depth boundaries from P/S layers")
    if boundaries.size > 1 and not np.all(np.diff(boundaries) > 0):
        raise TraveltimeReadyBuildError("derived depth boundaries are not strictly increasing")
    return boundaries


def _derive_arrays_from_layers(
    model: EquivalentLayeredVelocityModel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = _layer_boundaries(model)
    vp, vs = model.query(depth, extrapolate=True)
    depth = np.asarray(depth, dtype=np.float64)
    vp = np.asarray(vp, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    _validate_array_triplet(depth, vp, vs)
    return depth, vp, vs


def _write_array_triplet(
    ready_dir: Path,
    depth: np.ndarray,
    vp: np.ndarray,
    vs: np.ndarray,
) -> None:
    ready_dir.mkdir(parents=True, exist_ok=True)
    np.save(ready_dir / DEPTH_FILENAME, np.asarray(depth, dtype=np.float64))
    np.save(ready_dir / VP_FILENAME, np.asarray(vp, dtype=np.float64))
    np.save(ready_dir / VS_FILENAME, np.asarray(vs, dtype=np.float64))


def _build_asset_hashes(ready_dir: Path, filenames: Sequence[str] = HASHED_READY_FILENAMES) -> JsonDict:
    items: List[JsonDict] = []
    for name in sorted(filenames):
        path = ready_dir / name
        if not path.is_file():
            raise TraveltimeReadyBuildError(f"required ready asset missing before hash build: {path}")
        items.append(
            {
                "path": _to_posix_relative(path, ready_dir),
                "sha256": _compute_sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": HASH_ALGORITHM,
        "count": len(items),
        "items": items,
        "generated_at": _utc_now_iso(),
    }


def _write_interface_note(
    *,
    ready_dir: Path,
    source_equiv_layered_dir: Path,
    array_source: str,
    missing_source_arrays: Sequence[str],
) -> None:
    missing_text = ", ".join(missing_source_arrays) if missing_source_arrays else "none"
    note = f"""# velocity traveltime_ready interface

This directory is the velocity module's formal traveltime-facing ready handoff.

Canonical main interface JSON:

- `{MAIN_JSON_FILENAME}`

Core sampled arrays:

- `{DEPTH_FILENAME}`
- `{VP_FILENAME}`
- `{VS_FILENAME}`

Interface evidence and integrity files:

- `{ASSET_HASHES_FILENAME}`
- `{QC_TRAVELTIME_EQUIV_FILENAME}`

Source equiv_layered directory:

- `{source_equiv_layered_dir}`

Array source mode:

- `{array_source}`

Missing source arrays that triggered derivation:

- `{missing_text}`

Boundary statement:

- This directory does not contain traveltime tables.
- It does not implement traveltime ray tracing or locator logic.
- It exposes a minimum, stable, P/S dual-phase velocity view for downstream traveltime consumption.
"""
    (ready_dir / INTERFACE_NOTE_FILENAME).write_text(note, encoding="utf-8")


def _build_qc_report(
    *,
    source_equiv_layered_dir: Path,
    ready_dir: Path,
    model: EquivalentLayeredVelocityModel,
    array_source: str,
    missing_source_arrays: Sequence[str],
    copied_source_qc: bool,
) -> JsonDict:
    depth = np.load(ready_dir / DEPTH_FILENAME)
    vp = np.load(ready_dir / VP_FILENAME)
    vs = np.load(ready_dir / VS_FILENAME)
    _validate_array_triplet(depth, vp, vs)

    warnings: List[str] = []
    if missing_source_arrays:
        warnings.append(
            "source equiv_layered directory did not contain a complete depth/vp/vs array triplet; "
            "traveltime_ready arrays were derived from velocity_model_equiv_layered.json layers"
        )
    if not copied_source_qc:
        warnings.append(
            "source qc_traveltime_equiv.json was not available; generated a lightweight Step 6 interface QC report"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "module": MODULE_NAME_DEFAULT,
        "check_name": "traveltime_ready_equiv_interface",
        "status": "warning" if warnings else "passed",
        "generated_at": _utc_now_iso(),
        "source_equiv_layered_dir": str(source_equiv_layered_dir),
        "ready_dir": str(ready_dir),
        "main_json": MAIN_JSON_FILENAME,
        "array_source": array_source,
        "missing_source_arrays": list(missing_source_arrays),
        "copied_source_qc": bool(copied_source_qc),
        "depth_range_m": {
            "min": float(np.min(depth)),
            "max": float(np.max(depth)),
        },
        "sample_count": int(depth.size),
        "vp_range_mps": {
            "min": float(np.min(vp)),
            "max": float(np.max(vp)),
        },
        "vs_range_mps": {
            "min": float(np.min(vs)),
            "max": float(np.max(vs)),
        },
        "layers": {
            "P": int(len(model.get_layers("P"))),
            "S": int(len(model.get_layers("S"))),
        },
        "required_files": list(CORE_READY_FILENAMES),
        "present_files": [name for name in CORE_READY_FILENAMES if (ready_dir / name).is_file()],
        "warnings": warnings,
        "errors": [],
    }


def _copy_or_generate_qc_report(
    *,
    source_equiv_layered_dir: Path,
    ready_dir: Path,
    model: EquivalentLayeredVelocityModel,
    array_source: str,
    missing_source_arrays: Sequence[str],
) -> bool:
    source_qc = source_equiv_layered_dir / QC_TRAVELTIME_EQUIV_FILENAME
    target_qc = ready_dir / QC_TRAVELTIME_EQUIV_FILENAME
    copied = False
    if source_qc.is_file():
        _copy_file(source_qc, target_qc)
        # Ensure the copied file is valid JSON. If not, fail early rather than
        # exporting an unreadable interface evidence file.
        _load_json_file(target_qc)
        copied = True
    else:
        report = _build_qc_report(
            source_equiv_layered_dir=source_equiv_layered_dir,
            ready_dir=ready_dir,
            model=model,
            array_source=array_source,
            missing_source_arrays=missing_source_arrays,
            copied_source_qc=False,
        )
        _write_json_file(target_qc, report)
    return copied


def _update_generated_qc_if_needed(
    *,
    source_equiv_layered_dir: Path,
    ready_dir: Path,
    model: EquivalentLayeredVelocityModel,
    array_source: str,
    missing_source_arrays: Sequence[str],
    copied_source_qc: bool,
) -> None:
    if copied_source_qc:
        return
    report = _build_qc_report(
        source_equiv_layered_dir=source_equiv_layered_dir,
        ready_dir=ready_dir,
        model=model,
        array_source=array_source,
        missing_source_arrays=missing_source_arrays,
        copied_source_qc=False,
    )
    _write_json_file(ready_dir / QC_TRAVELTIME_EQUIV_FILENAME, report)


def _verify_ready_outputs(ready_dir: Path) -> None:
    missing = [name for name in CORE_READY_FILENAMES if not (ready_dir / name).is_file()]
    if missing:
        raise TraveltimeReadyBuildError(f"traveltime_ready is missing required files: {missing}")
    depth = np.load(ready_dir / DEPTH_FILENAME)
    vp = np.load(ready_dir / VP_FILENAME)
    vs = np.load(ready_dir / VS_FILENAME)
    _validate_array_triplet(depth, vp, vs)
    _load_json_file(ready_dir / MAIN_JSON_FILENAME)
    _load_json_file(ready_dir / QC_TRAVELTIME_EQUIV_FILENAME)
    asset_hashes = _load_json_file(ready_dir / ASSET_HASHES_FILENAME)
    if asset_hashes.get("algorithm") != HASH_ALGORITHM:
        raise TraveltimeReadyBuildError("asset_hashes.json algorithm must be sha256")


def build_traveltime_ready(
    *,
    equiv_layered_dir: PathLike,
    out_dir: PathLike,
    overwrite: bool = True,
) -> JsonDict:
    """
    Build the velocity module's traveltime_ready interface directory.

    Strategy:
    - Always validate and copy velocity_model_equiv_layered.json as the main JSON.
    - If source depth.npy/vp.npy/vs.npy are all present, copy the full array triplet.
    - If any array is missing, derive a minimum depth/vp/vs triplet from P/S layers.
    - Generate interface_note.md, qc_traveltime_equiv.json, and asset_hashes.json.
    """
    source_dir = _as_path(equiv_layered_dir)
    ready_dir = _as_path(out_dir)
    if not source_dir.is_dir():
        raise TraveltimeReadyBuildError(f"equiv_layered_dir is not a directory: {source_dir}")

    source_json = source_dir / MAIN_JSON_FILENAME
    if not source_json.is_file():
        raise TraveltimeReadyBuildError(f"{MAIN_JSON_FILENAME} not found in equiv_layered_dir: {source_dir}")

    try:
        model = load_equiv_layered_velocity_model(source_json)
    except VelocityModelError as exc:
        raise TraveltimeReadyBuildError(f"invalid {MAIN_JSON_FILENAME}: {exc}") from exc

    ready_dir.mkdir(parents=True, exist_ok=True)
    if not ready_dir.is_dir():
        raise TraveltimeReadyBuildError(f"out_dir is not a directory: {ready_dir}")

    if not overwrite:
        existing = [name for name in CORE_READY_FILENAMES if (ready_dir / name).exists()]
        if existing:
            raise TraveltimeReadyBuildError(
                f"out_dir already contains traveltime_ready files and overwrite=False: {existing}"
            )

    _copy_file(source_json, ready_dir / MAIN_JSON_FILENAME)

    missing_arrays = [name for name in ARRAY_FILENAMES if not (source_dir / name).is_file()]
    if missing_arrays:
        depth, vp, vs = _derive_arrays_from_layers(model)
        array_source = "derived_from_layers"
    else:
        depth, vp, vs = _load_source_arrays(source_dir)
        array_source = "copied_from_equiv_layered"
    _write_array_triplet(ready_dir, depth, vp, vs)

    _write_interface_note(
        ready_dir=ready_dir,
        source_equiv_layered_dir=source_dir,
        array_source=array_source,
        missing_source_arrays=missing_arrays,
    )

    copied_qc = _copy_or_generate_qc_report(
        source_equiv_layered_dir=source_dir,
        ready_dir=ready_dir,
        model=model,
        array_source=array_source,
        missing_source_arrays=missing_arrays,
    )
    _update_generated_qc_if_needed(
        source_equiv_layered_dir=source_dir,
        ready_dir=ready_dir,
        model=model,
        array_source=array_source,
        missing_source_arrays=missing_arrays,
        copied_source_qc=copied_qc,
    )

    asset_hashes = _build_asset_hashes(ready_dir)
    _write_json_file(ready_dir / ASSET_HASHES_FILENAME, asset_hashes)

    _verify_ready_outputs(ready_dir)

    result = {
        "schema_version": SCHEMA_VERSION,
        "module": MODULE_NAME_DEFAULT,
        "status": "completed",
        "ready_dir": str(ready_dir),
        "source_equiv_layered_dir": str(source_dir),
        "main_json": str(ready_dir / MAIN_JSON_FILENAME),
        "array_source": array_source,
        "missing_source_arrays": missing_arrays,
        "copied_source_qc": copied_qc,
        "outputs": [str(ready_dir / name) for name in CORE_READY_FILENAMES],
        "asset_hashes_count": int(asset_hashes["count"]),
        "warnings": [] if not missing_arrays else [
            "source equiv_layered arrays were incomplete; arrays derived from P/S layers"
        ],
        "errors": [],
    }
    return result


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_traveltime_ready",
        description="Build velocity/traveltime_ready from velocity/equiv_layered outputs.",
    )
    p.add_argument("--out_root", default="prepared_project", help="Root output directory. Default: prepared_project")
    p.add_argument("--module_name", default=MODULE_NAME_DEFAULT, help="Module name under out_root. Default: velocity")
    p.add_argument(
        "--equiv_layered_dir",
        default=None,
        help="Source equiv_layered directory. Default: <out_root>/<module_name>/equiv_layered",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output traveltime_ready directory. Default: <out_root>/<module_name>/traveltime_ready",
    )
    p.add_argument("--no_overwrite", action="store_true", help="Fail if traveltime_ready files already exist.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    equiv_dir = Path(args.equiv_layered_dir) if args.equiv_layered_dir else _default_velocity_dir(
        args.out_root, args.module_name, EQUIV_LAYERED_DIRNAME
    )
    out_dir = Path(args.out_dir) if args.out_dir else _default_velocity_dir(
        args.out_root, args.module_name, TRAVELTIME_READY_DIRNAME
    )

    result = build_traveltime_ready(
        equiv_layered_dir=equiv_dir,
        out_dir=out_dir,
        overwrite=not bool(args.no_overwrite),
    )
    print("[OK] traveltime_ready generated")
    print(f"  source_dir : {Path(str(result['source_equiv_layered_dir'])).resolve()}")
    print(f"  ready_dir  : {Path(str(result['ready_dir'])).resolve()}")
    print(f"  main_json  : {Path(str(result['main_json'])).resolve()}")
    print(f"  arrays     : {result['array_source']}")
    print(f"  hash_count : {result['asset_hashes_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
