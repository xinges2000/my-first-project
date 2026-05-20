# microseis_ds/velocity/build/build_velocity_manifest.py
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


JsonDict = Dict[str, Any]
PathLike = Union[str, Path]

SCHEMA_VERSION = "1.0.0"
MODULE_NAME_DEFAULT = "velocity"
HASH_ALGORITHM = "sha256"

REF_ENGINEERING_DIRNAME = "ref_engineering"
EQUIV_LAYERED_DIRNAME = "equiv_layered"
META_DIRNAME = "meta"
MANIFEST_FILENAME = "velocity_manifest.json"
SUMMARY_FILENAME = "velocity_build_summary.json"


# Baseline handoff assets that velocity may read, carry, or use as provenance.
# Preferred candidates reflect the frozen baseline formal layout; later entries
# preserve compatibility with older flat-root outputs.
BASELINE_HANDOFF_ASSET_CANDIDATES: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("stations", ("stations.csv",), True),
    ("traceid_station", ("links/traceid_station.csv", "traceid_station.csv"), True),
    ("wellpaths_index", ("meta/wellpaths_index.json", "wellpaths_index.json", "wellpaths/wellpaths_index.json"), True),
    ("project_json", ("project.json",), True),
    ("baseline_manifest", ("manifest.json",), True),
    ("baseline_build_summary", ("meta/baseline_build_summary.json", "baseline_build_summary.json"), True),
    ("wellpath_build_summary", ("meta/wellpath_build_summary.json", "wellpath_build_summary.json"), False),
)

BASELINE_FORMAL_PRIMARY_RELATIVE_PATHS: Dict[str, str] = {
    "stations": "stations.csv",
    "traceid_station": "links/traceid_station.csv",
    "wellpaths_index": "meta/wellpaths_index.json",
    "project_json": "project.json",
    "baseline_manifest": "manifest.json",
    "baseline_build_summary": "meta/baseline_build_summary.json",
    "wellpath_build_summary": "meta/wellpath_build_summary.json",
}

BASELINE_DIRECT_WELLPATH_EXTENSIONS: Tuple[str, ...] = (".csv",)

# Core formal artifacts produced by Step 3 ref_engineering.
REF_ENGINEERING_ASSET_NAMES: Tuple[str, ...] = (
    "velocity_model.json",
    "depth.npy",
    "vp.npy",
    "vs.npy",
    "ref_tvd.csv",
    "ref_tvd_summary.json",
    "velocity_1d_engineering.csv",
)
REF_ENGINEERING_REQUIRED_ASSET_NAMES: Tuple[str, ...] = (
    "velocity_model.json",
    "depth.npy",
    "vp.npy",
    "vs.npy",
)

# Core formal artifacts produced by Step 4 equiv_layered / layerize path.
EQUIV_LAYERED_ASSET_NAMES: Tuple[str, ...] = (
    "velocity_model_equiv_layered.json",
    "depth.npy",
    "vp.npy",
    "vs.npy",
    "qc_traveltime_equiv.json",
)
EQUIV_LAYERED_REQUIRED_ASSET_NAMES: Tuple[str, ...] = (
    "velocity_model_equiv_layered.json",
    "depth.npy",
    "vp.npy",
    "vs.npy",
)

# Cache/build implementation directories that must never be treated as assets.
# The smoke-test output root is intentionally not listed here; real artifacts
# written under a caller-selected output root must remain collectable.
EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

# Self-referential outputs are excluded from core hash calculation.
EXCLUDED_FILENAMES = {
    MANIFEST_FILENAME,
    SUMMARY_FILENAME,
}

# Formal velocity assets that may be recursively collected under velocity output roots.
FORMAL_ASSET_SUFFIXES = {
    ".json",
    ".npy",
    ".csv",
}

# Formal delivery assets can be copied into a package directory by later steps.
DELIVERY_ASSET_SUFFIXES = {
    ".json",
    ".npy",
    ".csv",
    ".md",
    ".txt",
    ".zip",
}


@dataclass(frozen=True)
class VelocityAsset:
    """One collected formal velocity artifact."""

    path: Path
    relative_path: str
    group: str
    required: bool = False

    def exists(self) -> bool:
        return self.path.is_file()


class VelocityManifestError(RuntimeError):
    """Raised when manifest construction receives invalid paths or invalid state."""


def _as_path(value: PathLike) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_velocity_root(
    *,
    out_root: Optional[PathLike] = None,
    module_name: str = MODULE_NAME_DEFAULT,
    velocity_root: Optional[PathLike] = None,
) -> Path:
    if velocity_root is not None:
        return _as_path(velocity_root)
    if out_root is None:
        out_root = "prepared_project"
    return _as_path(out_root) / module_name


def _default_base_dir(
    *,
    out_root: Optional[PathLike],
    velocity_root: Path,
) -> Path:
    if out_root is not None:
        return _as_path(out_root)
    return velocity_root.parent


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def to_posix_relative_path(path: PathLike, base_dir: PathLike) -> str:
    """Return a stable POSIX-style relative path for manifest keys."""
    p = _safe_resolve(_as_path(path))
    base = _safe_resolve(_as_path(base_dir))
    try:
        rel = p.relative_to(base)
    except ValueError as exc:
        raise VelocityManifestError(f"Asset path is outside base_dir: path={p}, base_dir={base}") from exc
    return rel.as_posix()


def _has_excluded_parent(path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def _is_excluded_file(path: Path) -> bool:
    return path.name in EXCLUDED_FILENAMES or _has_excluded_parent(path)


def _is_formal_file(path: Path, suffixes: Iterable[str]) -> bool:
    return path.is_file() and path.suffix.lower() in set(suffixes) and not _is_excluded_file(path)


def _group_for_path(path: Path, velocity_root: Path) -> str:
    try:
        rel_parts = path.relative_to(velocity_root).parts
    except ValueError:
        rel_parts = path.parts

    if REF_ENGINEERING_DIRNAME in rel_parts:
        return "primary_outputs"
    if EQUIV_LAYERED_DIRNAME in rel_parts:
        return "equiv_layered_outputs"
    return "velocity_outputs"


def _dedupe_assets(assets: Iterable[VelocityAsset]) -> List[VelocityAsset]:
    by_rel: Dict[str, VelocityAsset] = {}
    for asset in assets:
        existing = by_rel.get(asset.relative_path)
        if existing is None or (asset.required and not existing.required):
            by_rel[asset.relative_path] = asset
    return [by_rel[key] for key in sorted(by_rel.keys())]


def _collect_named_assets(
    *,
    directory: Path,
    base_dir: Path,
    names: Sequence[str],
    group: str,
    required_names: Sequence[str] = (),
) -> List[VelocityAsset]:
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise VelocityManifestError(f"Expected asset directory, got file: {directory}")

    required_set = set(required_names)
    out: List[VelocityAsset] = []
    for name in names:
        p = directory / name
        if not p.is_file() or _is_excluded_file(p):
            continue
        out.append(
            VelocityAsset(
                path=p,
                relative_path=to_posix_relative_path(p, base_dir),
                group=group,
                required=name in required_set,
            )
        )
    return out


def _collect_recursive_velocity_assets(
    *,
    velocity_root: Path,
    base_dir: Path,
) -> List[VelocityAsset]:
    if not velocity_root.exists():
        return []
    if not velocity_root.is_dir():
        raise VelocityManifestError(f"velocity_root is not a directory: {velocity_root}")

    out: List[VelocityAsset] = []
    for p in sorted(velocity_root.rglob("*")):
        if not _is_formal_file(p, FORMAL_ASSET_SUFFIXES):
            continue
        out.append(
            VelocityAsset(
                path=p,
                relative_path=to_posix_relative_path(p, base_dir),
                group=_group_for_path(p, velocity_root),
                required=False,
            )
        )
    return out


def _collect_delivery_assets(
    *,
    delivery_dir: Optional[Path],
    base_dir: Path,
) -> List[VelocityAsset]:
    if delivery_dir is None or not delivery_dir.exists():
        return []
    if not delivery_dir.is_dir():
        raise VelocityManifestError(f"delivery_dir is not a directory: {delivery_dir}")

    out: List[VelocityAsset] = []
    for p in sorted(delivery_dir.rglob("*")):
        if not _is_formal_file(p, DELIVERY_ASSET_SUFFIXES):
            continue
        out.append(
            VelocityAsset(
                path=p,
                relative_path=to_posix_relative_path(p, base_dir),
                group="delivery_outputs",
                required=False,
            )
        )
    return out


def _collect_extra_assets(
    *,
    extra_asset_paths: Optional[Iterable[PathLike]],
    base_dir: Path,
) -> List[VelocityAsset]:
    if not extra_asset_paths:
        return []
    out: List[VelocityAsset] = []
    for item in extra_asset_paths:
        p = _as_path(item)
        if not _is_formal_file(p, FORMAL_ASSET_SUFFIXES | DELIVERY_ASSET_SUFFIXES):
            continue
        out.append(
            VelocityAsset(
                path=p,
                relative_path=to_posix_relative_path(p, base_dir),
                group="extra_outputs",
                required=False,
            )
        )
    return out


def collect_velocity_assets(
    *,
    velocity_root: PathLike,
    base_dir: Optional[PathLike] = None,
    delivery_dir: Optional[PathLike] = None,
    extra_asset_paths: Optional[Iterable[PathLike]] = None,
) -> List[VelocityAsset]:
    """
    Collect existing formal velocity artifacts.

    This function indexes only real files that already exist. It does not create
    model artifacts and does not enforce temporary logs, cache files, pytest
    caches, or self-referential manifest/summary files.
    """
    root = _as_path(velocity_root)
    base = _as_path(base_dir) if base_dir is not None else root.parent
    delivery = _as_path(delivery_dir) if delivery_dir is not None else None

    ref_dir = root / REF_ENGINEERING_DIRNAME
    equiv_dir = root / EQUIV_LAYERED_DIRNAME

    assets: List[VelocityAsset] = []
    assets.extend(
        _collect_named_assets(
            directory=ref_dir,
            base_dir=base,
            names=REF_ENGINEERING_ASSET_NAMES,
            group="primary_outputs",
            required_names=REF_ENGINEERING_REQUIRED_ASSET_NAMES,
        )
    )
    assets.extend(
        _collect_named_assets(
            directory=equiv_dir,
            base_dir=base,
            names=EQUIV_LAYERED_ASSET_NAMES,
            group="equiv_layered_outputs",
            required_names=EQUIV_LAYERED_REQUIRED_ASSET_NAMES,
        )
    )
    assets.extend(_collect_recursive_velocity_assets(velocity_root=root, base_dir=base))
    assets.extend(_collect_delivery_assets(delivery_dir=delivery, base_dir=base))
    assets.extend(_collect_extra_assets(extra_asset_paths=extra_asset_paths, base_dir=base))

    return _dedupe_assets(assets)


def compute_sha256(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 from file bytes without using mtime or platform metadata."""
    p = _as_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Cannot compute SHA256 for missing file: {p}")

    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_asset_hashes(assets: Iterable[VelocityAsset]) -> JsonDict:
    """Build a deterministic asset_hashes object for manifest and summary."""
    items: List[JsonDict] = []
    for asset in sorted(assets, key=lambda a: a.relative_path):
        if not asset.path.is_file():
            continue
        items.append(
            {
                "path": asset.relative_path,
                "sha256": compute_sha256(asset.path),
                "size_bytes": int(asset.path.stat().st_size),
            }
        )
    return {
        "algorithm": HASH_ALGORITHM,
        "count": int(len(items)),
        "items": items,
    }




def _existing_candidate(root: Path, candidates: Sequence[str]) -> Tuple[Optional[Path], Optional[str], List[str]]:
    checked = [str(root / rel) for rel in candidates]
    for rel in candidates:
        p = root / rel
        if p.is_file():
            return p, rel, checked
    return None, None, checked


def _relative_or_posix(path: Path, base_dir: Path) -> str:
    try:
        return to_posix_relative_path(path, base_dir)
    except VelocityManifestError:
        return path.as_posix()


def _hash_file_record(
    *,
    role: str,
    path: Path,
    base_dir: Path,
    required: bool,
    selected_relative_path: Optional[str] = None,
    source: str = "baseline_handoff",
) -> JsonDict:
    rel = _relative_or_posix(path, base_dir)
    return {
        "role": role,
        "path": rel,
        "selected_relative_path": selected_relative_path or rel,
        "sha256": compute_sha256(path),
        "size_bytes": int(path.stat().st_size),
        "required": bool(required),
        "source": source,
    }


def _normalized_source_inputs_mapping(source_inputs: Optional[Union[Mapping[str, Any], Sequence[Any]]]) -> Mapping[str, Any]:
    if isinstance(source_inputs, Mapping):
        return source_inputs
    return {}


def _resolve_source_input_path(raw: Any, *, roots: Sequence[Path]) -> Optional[Path]:
    if not isinstance(raw, str) or not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p if p.exists() else p
    for root in roots:
        candidate = root / p
        if candidate.exists():
            return candidate
    return roots[0] / p if roots else p


def _collect_direct_wellpath_sources(
    *,
    baseline_root: Path,
    base_dir: Path,
    source_inputs: Optional[Union[Mapping[str, Any], Sequence[Any]]],
) -> Tuple[List[JsonDict], JsonDict]:
    inputs = _normalized_source_inputs_mapping(source_inputs)
    raw_values: List[Any] = []
    for key in ("wellpath_path", "wellpath_csv", "wellpath_file", "wellpath_files"):
        value = inputs.get(key)
        if isinstance(value, list):
            raw_values.extend(value)
        elif value not in (None, ""):
            raw_values.append(value)

    roots = (base_dir, baseline_root, baseline_root.parent)
    records: List[JsonDict] = []
    seen: set[str] = set()
    for raw in raw_values:
        p = _resolve_source_input_path(raw, roots=roots)
        if p is None:
            continue
        candidates: List[Path] = []
        if p.is_dir():
            candidates.extend(sorted(x for x in p.glob("wellpath_*.csv") if x.is_file()))
        elif p.is_file() and p.name.startswith("wellpath_") and p.suffix.lower() in BASELINE_DIRECT_WELLPATH_EXTENSIONS:
            candidates.append(p)
        for item in candidates:
            key = str(item.resolve())
            if key in seen:
                continue
            seen.add(key)
            records.append(
                _hash_file_record(
                    role="wellpath_csv",
                    path=item,
                    base_dir=base_dir,
                    required=True,
                    selected_relative_path=_relative_or_posix(item, baseline_root),
                    source="baseline_handoff.direct_wellpath_source",
                )
            )

    return records, {
        "direct_wellpath_csv_read": bool(records),
        "source_input_keys_checked": ["wellpath_path", "wellpath_csv", "wellpath_file", "wellpath_files"],
        "boundary": (
            "wellpath_*.csv hashes are mandatory only when a concrete wellpath CSV "
            "is provided through velocity source_inputs; otherwise velocity handoff "
            "is traced through wellpaths_index.json."
        ),
    }


def build_baseline_handoff_evidence(
    *,
    baseline_root: Optional[PathLike],
    base_dir: PathLike,
    source_inputs: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    z_positive: str = "down",
) -> JsonDict:
    """Build structured baseline -> velocity provenance and SHA256 evidence.

    This does not mutate baseline outputs. It records velocity-side evidence for
    the baseline assets selected from the frozen formal layout first, with legacy
    flat-root fallback paths retained for compatibility.
    """
    if baseline_root is None:
        return {
            "status": "not_available",
            "reason": "baseline_root was not provided",
            "z_positive": z_positive,
            "depth_positive": z_positive,
            "asset_hashes": {"algorithm": HASH_ALGORITHM, "count": 0, "items": []},
            "warnings": [],
            "errors": [],
        }

    root = _as_path(baseline_root)
    base = _as_path(base_dir)
    warnings: List[str] = []
    errors: List[str] = []
    source_paths: JsonDict = {"baseline_root": _relative_or_posix(root, base) if root.exists() else root.as_posix()}
    selected_assets: List[JsonDict] = []
    selected_path_policy: JsonDict = {}

    if not root.is_dir():
        warnings.append(f"baseline_root does not exist; baseline handoff evidence not collected: {root}")
        return {
            "status": "not_available",
            "baseline_root": root.as_posix(),
            "z_positive": z_positive,
            "depth_positive": z_positive,
            "asset_hashes": {"algorithm": HASH_ALGORITHM, "count": 0, "items": []},
            "warnings": warnings,
            "errors": errors,
        }

    for role, candidates, required in BASELINE_HANDOFF_ASSET_CANDIDATES:
        path, selected_rel, checked = _existing_candidate(root, candidates)
        formal_rel = BASELINE_FORMAL_PRIMARY_RELATIVE_PATHS.get(role)
        policy = {
            "preferred_relative_path": formal_rel,
            "candidate_relative_paths": list(candidates),
            "checked_paths": checked,
            "selected_relative_path": selected_rel,
            "selection": "formal_primary" if selected_rel == formal_rel else ("legacy_fallback" if selected_rel else "missing"),
            "required": bool(required),
        }
        selected_path_policy[role] = policy
        if path is None:
            message = f"baseline handoff asset not found for role={role}; candidates={list(candidates)}"
            if required:
                errors.append(message)
            else:
                warnings.append(message)
            source_paths[f"{role}_path"] = ""
            source_paths[f"{role}_sha256"] = ""
            continue
        source_paths[f"{role}_path"] = _relative_or_posix(path, base)
        selected_assets.append(
            _hash_file_record(
                role=role,
                path=path,
                base_dir=base,
                required=required,
                selected_relative_path=selected_rel,
            )
        )
        source_paths[f"{role}_sha256"] = selected_assets[-1]["sha256"]

    direct_wellpath_records, wellpath_policy = _collect_direct_wellpath_sources(
        baseline_root=root,
        base_dir=base,
        source_inputs=source_inputs,
    )
    selected_assets.extend(direct_wellpath_records)

    items = sorted(selected_assets, key=lambda x: (str(x.get("path", "")), str(x.get("role", ""))))
    asset_hashes = {
        "algorithm": HASH_ALGORITHM,
        "count": int(len(items)),
        "items": items,
    }
    mandatory_roles = [role for role, _candidates, required in BASELINE_HANDOFF_ASSET_CANDIDATES if required]
    hashed_roles = {str(item.get("role")) for item in items}
    missing_mandatory_hash_roles = [role for role in mandatory_roles if role not in hashed_roles]
    if missing_mandatory_hash_roles:
        errors.append(f"missing mandatory baseline handoff hash roles: {missing_mandatory_hash_roles}")

    status = "failed" if errors else ("warning" if warnings else "passed")
    return {
        "schema_version": "baseline_handoff_v1",
        "status": status,
        "baseline_root": source_paths.get("baseline_root", root.as_posix()),
        "z_positive": z_positive,
        "depth_positive": z_positive,
        "source_paths": source_paths,
        "selected_path_policy": selected_path_policy,
        "asset_hashes": asset_hashes,
        "mandatory_hash_roles": mandatory_roles,
        "missing_mandatory_hash_roles": missing_mandatory_hash_roles,
        "direct_wellpath_policy": wellpath_policy,
        "warnings": warnings,
        "errors": errors,
        "boundary": {
            "baseline_generator_not_modified": True,
            "velocity_side_evidence_only": True,
            "formal_layout_preferred": True,
            "legacy_flat_root_fallback_supported": True,
        },
    }


def _paths_by_group(assets: Iterable[VelocityAsset], group: str) -> List[str]:
    return sorted(asset.relative_path for asset in assets if asset.group == group and asset.path.is_file())


def _all_asset_paths(assets: Iterable[VelocityAsset]) -> List[str]:
    return sorted(asset.relative_path for asset in assets if asset.path.is_file())


def _normalize_source_inputs(source_inputs: Optional[Union[Mapping[str, Any], Sequence[Any]]]) -> Any:
    if source_inputs is None:
        return {}
    if isinstance(source_inputs, Mapping):
        return dict(source_inputs)
    return [str(item) for item in source_inputs]


def _required_specs_for_existing_outputs(root: Path, base_dir: Path) -> List[Tuple[str, Path, str]]:
    specs: List[Tuple[str, Path, str]] = []

    ref_dir = root / REF_ENGINEERING_DIRNAME
    if ref_dir.is_dir():
        for name in REF_ENGINEERING_REQUIRED_ASSET_NAMES:
            specs.append(("ref_engineering", ref_dir / name, to_posix_relative_path(ref_dir / name, base_dir)))

    equiv_dir = root / EQUIV_LAYERED_DIRNAME
    if equiv_dir.is_dir():
        # Only require equiv_layered core assets when that output directory has
        # actually been produced for the current build/task.
        has_equiv_formal_assets = any(_is_formal_file(p, FORMAL_ASSET_SUFFIXES) for p in equiv_dir.rglob("*"))
        if has_equiv_formal_assets:
            for name in EQUIV_LAYERED_REQUIRED_ASSET_NAMES:
                specs.append(("equiv_layered", equiv_dir / name, to_posix_relative_path(equiv_dir / name, base_dir)))

    return specs


def _required_asset_status(root: Path, base_dir: Path) -> JsonDict:
    specs = _required_specs_for_existing_outputs(root, base_dir)
    items: List[JsonDict] = []
    generated: List[str] = []
    missing: List[str] = []

    for group, path, rel in specs:
        present = path.is_file() and not _is_excluded_file(path)
        item = {
            "group": group,
            "path": rel,
            "status": "present" if present else "missing",
        }
        items.append(item)
        if present:
            generated.append(rel)
        else:
            missing.append(rel)

    if not items:
        status = "not_applicable"
    elif missing:
        status = "missing"
    else:
        status = "complete"

    return {
        "status": status,
        "required_count": int(len(items)),
        "present_count": int(len(generated)),
        "missing_count": int(len(missing)),
        "present": sorted(generated),
        "missing": sorted(missing),
        "items": sorted(items, key=lambda x: x["path"]),
    }


def _contract_info() -> JsonDict:
    return {
        "input_contract": {
            "ref_engineering_log": "depth/vp/vs columns must be present, finite, strictly increasing in depth, and positive in velocity",
        },
        "output_contract": {
            "ref_engineering": list(REF_ENGINEERING_REQUIRED_ASSET_NAMES),
            "equiv_layered": list(EQUIV_LAYERED_REQUIRED_ASSET_NAMES),
            "hash": "asset hashes are SHA256 over file bytes, keyed by stable POSIX relative paths",
        },
    }


def _derive_status(
    *,
    requested_status: str,
    asset_count: int,
    required_status: Mapping[str, Any],
    errors: Sequence[str],
) -> str:
    if errors:
        return "failed"
    if asset_count <= 0:
        return "warning"
    if required_status.get("status") == "missing":
        return "warning"
    return requested_status or "completed"


def build_velocity_manifest_data(
    *,
    module_name: str = MODULE_NAME_DEFAULT,
    status: str = "completed",
    generator: str = "microseis_ds.velocity.build.build_velocity_manifest",
    generated_at: Optional[str] = None,
    assets: Sequence[VelocityAsset],
    asset_hashes: Mapping[str, Any],
    manifest_path: str,
    build_summary_path: str,
    source_inputs: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    baseline_handoff: Optional[Mapping[str, Any]] = None,
    warnings: Optional[Sequence[str]] = None,
    errors: Optional[Sequence[str]] = None,
) -> JsonDict:
    """Construct the velocity_manifest.json dictionary without writing files."""
    assets_list = list(assets)
    return {
        "schema_version": SCHEMA_VERSION,
        "module": module_name,
        "status": status,
        "generator": generator,
        "generated_at": generated_at or _utc_now_iso(),
        "primary_outputs": _paths_by_group(assets_list, "primary_outputs"),
        "equiv_layered_outputs": _paths_by_group(assets_list, "equiv_layered_outputs"),
        "velocity_outputs": _paths_by_group(assets_list, "velocity_outputs"),
        "delivery_outputs": _paths_by_group(assets_list, "delivery_outputs"),
        "extra_outputs": _paths_by_group(assets_list, "extra_outputs"),
        "generated_outputs": _all_asset_paths(assets_list),
        "meta_outputs": [build_summary_path],
        "asset_hashes": dict(asset_hashes),
        "build_summary_path": build_summary_path,
        "source_inputs": _normalize_source_inputs(source_inputs),
        "baseline_handoff": dict(baseline_handoff or {}),
        "warnings": list(warnings or []),
        "errors": list(errors or []),
    }


def build_velocity_summary_data(
    *,
    module_name: str = MODULE_NAME_DEFAULT,
    status: str = "completed",
    build_entry: str = "scripts/build_velocity.py",
    completed_at: Optional[str] = None,
    assets: Sequence[VelocityAsset],
    asset_hashes: Mapping[str, Any],
    manifest_path: str,
    required_asset_status: Mapping[str, Any],
    baseline_handoff: Optional[Mapping[str, Any]] = None,
    warnings: Optional[Sequence[str]] = None,
    errors: Optional[Sequence[str]] = None,
) -> JsonDict:
    """Construct the meta/velocity_build_summary.json dictionary without writing files."""
    assets_list = list(assets)
    contract = _contract_info()
    generated_outputs = {
        "primary_outputs": _paths_by_group(assets_list, "primary_outputs"),
        "equiv_layered_outputs": _paths_by_group(assets_list, "equiv_layered_outputs"),
        "velocity_outputs": _paths_by_group(assets_list, "velocity_outputs"),
        "delivery_outputs": _paths_by_group(assets_list, "delivery_outputs"),
        "extra_outputs": _paths_by_group(assets_list, "extra_outputs"),
    }
    asset_items = list(asset_hashes.get("items", []))
    asset_count = int(len(asset_items))

    return {
        "schema_version": SCHEMA_VERSION,
        "module": module_name,
        "status": status,
        "build_entry": build_entry,
        "generated_outputs": generated_outputs,
        "asset_count": asset_count,
        "asset_hashes_count": asset_count,
        "required_asset_status": dict(required_asset_status),
        "input_contract": contract["input_contract"],
        "output_contract": contract["output_contract"],
        "manifest_path": manifest_path,
        "baseline_handoff": dict(baseline_handoff or {}),
        "warnings": list(warnings or []),
        "errors": list(errors or []),
        "completed_at": completed_at or _utc_now_iso(),
    }


def write_json_stable(data: Mapping[str, Any], out_path: PathLike, *, sort_keys: bool = False) -> Path:
    """Write UTF-8 JSON with stable indentation and parent directory creation."""
    p = _as_path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(dict(data), ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )
    return p


def _file_from_manifest_path(base_dir: Path, rel_path: str) -> Path:
    if "\\" in rel_path:
        raise VelocityManifestError(f"Manifest asset path must use POSIX '/': {rel_path}")
    rel = Path(*rel_path.split("/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise VelocityManifestError(f"Manifest asset path must be a safe relative path: {rel_path}")
    return base_dir / rel


def _validate_manifest_summary_consistency(
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    base_dir: Path,
) -> None:
    asset_hashes = manifest.get("asset_hashes", {})
    if not isinstance(asset_hashes, Mapping):
        raise VelocityManifestError("manifest.asset_hashes must be an object")

    items = asset_hashes.get("items", [])
    if not isinstance(items, list):
        raise VelocityManifestError("manifest.asset_hashes.items must be a list")

    count = int(len(items))
    if int(asset_hashes.get("count", -1)) != count:
        raise VelocityManifestError("manifest.asset_hashes.count does not match number of items")
    if int(summary.get("asset_count", -1)) != count:
        raise VelocityManifestError("summary.asset_count does not match number of asset hashes")
    if int(summary.get("asset_hashes_count", -1)) != count:
        raise VelocityManifestError("summary.asset_hashes_count does not match number of asset hashes")

    paths = [str(item.get("path", "")) for item in items]
    if paths != sorted(paths):
        raise VelocityManifestError("manifest.asset_hashes.items must be sorted by path")
    if len(paths) != len(set(paths)):
        raise VelocityManifestError("manifest.asset_hashes.items contains duplicate paths")

    for item in items:
        path_value = str(item.get("path", ""))
        if not path_value:
            raise VelocityManifestError("asset hash item missing path")
        if Path(path_value).name in EXCLUDED_FILENAMES:
            raise VelocityManifestError(f"self-referential manifest/summary file was hashed: {path_value}")
        file_path = _file_from_manifest_path(base_dir, path_value)
        if not file_path.is_file():
            raise VelocityManifestError(f"asset hash path does not resolve to a real file: {path_value}")
        size_bytes = int(item.get("size_bytes", -1))
        if size_bytes != int(file_path.stat().st_size):
            raise VelocityManifestError(f"asset size mismatch for {path_value}")
        sha256 = str(item.get("sha256", ""))
        if sha256 != compute_sha256(file_path):
            raise VelocityManifestError(f"asset sha256 mismatch for {path_value}")


def build_velocity_manifest(
    *,
    out_root: Optional[PathLike] = "prepared_project",
    module_name: str = MODULE_NAME_DEFAULT,
    velocity_root: Optional[PathLike] = None,
    delivery_dir: Optional[PathLike] = None,
    source_inputs: Optional[Union[Mapping[str, Any], Sequence[Any]]] = None,
    baseline_root: Optional[PathLike] = None,
    z_positive: str = "down",
    status: str = "completed",
    generated_at: Optional[str] = None,
    build_entry: str = "scripts/build_velocity.py",
    write_files: bool = True,
    extra_asset_paths: Optional[Iterable[PathLike]] = None,
) -> JsonDict:
    """
    Build velocity manifest and build_summary data, optionally writing both files.

    This is the intended Step 5.2 integration point for scripts/build_velocity.py.
    It does not run engineering, layerize, package, runtime, or traveltime_ready
    logic. It only indexes existing formal velocity artifacts.
    """
    root = _resolve_velocity_root(out_root=out_root, module_name=module_name, velocity_root=velocity_root)
    base_dir = _default_base_dir(out_root=out_root, velocity_root=root)

    if delivery_dir is None and out_root is not None:
        default_delivery = _as_path(out_root) / "velocity_delivery"
        delivery_dir = default_delivery if default_delivery.exists() else None

    manifest_path = root / MANIFEST_FILENAME
    summary_path = root / META_DIRNAME / SUMMARY_FILENAME
    manifest_rel = to_posix_relative_path(manifest_path, base_dir)
    summary_rel = to_posix_relative_path(summary_path, base_dir)

    assets = collect_velocity_assets(
        velocity_root=root,
        base_dir=base_dir,
        delivery_dir=delivery_dir,
        extra_asset_paths=extra_asset_paths,
    )
    asset_hashes = build_asset_hashes(assets)
    required_status = _required_asset_status(root, base_dir)
    if baseline_root is None and out_root is not None:
        default_baseline_root = _as_path(out_root) / "baseline"
        baseline_root = default_baseline_root if default_baseline_root.exists() else None
    baseline_handoff = build_baseline_handoff_evidence(
        baseline_root=baseline_root,
        base_dir=base_dir,
        source_inputs=source_inputs,
        z_positive=z_positive,
    )

    warnings: List[str] = []
    errors: List[str] = []
    if not root.exists():
        warnings.append(f"velocity_root does not exist yet: {root}")
    if asset_hashes["count"] == 0:
        warnings.append("no formal velocity assets were collected")
    if required_status.get("missing_count", 0):
        warnings.append("one or more required assets for existing velocity outputs are missing")

    final_status = _derive_status(
        requested_status=status,
        asset_count=int(asset_hashes["count"]),
        required_status=required_status,
        errors=errors,
    )

    manifest = build_velocity_manifest_data(
        module_name=module_name,
        status=final_status,
        generated_at=generated_at,
        assets=assets,
        asset_hashes=asset_hashes,
        manifest_path=manifest_rel,
        build_summary_path=summary_rel,
        source_inputs=source_inputs,
        baseline_handoff=baseline_handoff,
        warnings=warnings,
        errors=errors,
    )
    summary = build_velocity_summary_data(
        module_name=module_name,
        status=final_status,
        build_entry=build_entry,
        completed_at=generated_at,
        assets=assets,
        asset_hashes=asset_hashes,
        manifest_path=manifest_rel,
        required_asset_status=required_status,
        baseline_handoff=baseline_handoff,
        warnings=warnings,
        errors=errors,
    )

    _validate_manifest_summary_consistency(manifest=manifest, summary=summary, base_dir=base_dir)

    if write_files:
        write_json_stable(manifest, manifest_path)
        write_json_stable(summary, summary_path)

    return {
        "manifest": manifest,
        "summary": summary,
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "asset_hashes": asset_hashes,
        "asset_count": int(asset_hashes["count"]),
        "baseline_handoff": baseline_handoff,
    }


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_velocity_manifest",
        description=(
            "Build velocity_manifest.json and meta/velocity_build_summary.json "
            "from existing formal velocity artifacts."
        ),
    )
    parser.add_argument("--out_root", default="prepared_project", help="Output root. Default: prepared_project")
    parser.add_argument("--module_name", default=MODULE_NAME_DEFAULT, help="Module name. Default: velocity")
    parser.add_argument("--velocity_root", default=None, help="Optional explicit velocity module output directory.")
    parser.add_argument("--delivery_dir", default=None, help="Optional delivery bundle directory to include.")
    parser.add_argument("--baseline_root", default=None, help="Optional baseline handoff root for provenance/hash evidence.")
    parser.add_argument("--z_positive", default="down", help="Depth/Z positive direction recorded for baseline handoff evidence. Default: down")
    parser.add_argument("--status", default="completed", help="Build status recorded in manifest/summary.")
    parser.add_argument("--generated_at", default=None, help="Optional fixed ISO timestamp for reproducible tests.")
    parser.add_argument("--no_write", action="store_true", help="Build data but do not write JSON files.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    result = build_velocity_manifest(
        out_root=args.out_root,
        module_name=args.module_name,
        velocity_root=args.velocity_root,
        delivery_dir=args.delivery_dir,
        baseline_root=args.baseline_root,
        z_positive=args.z_positive,
        status=args.status,
        generated_at=args.generated_at,
        write_files=not bool(args.no_write),
    )
    print("[OK] velocity manifest data generated")
    print(f"  manifest_path : {result['manifest_path']}")
    print(f"  summary_path  : {result['summary_path']}")
    print(f"  asset_count   : {result['asset_count']}")
    return 0


__all__ = [
    "VelocityAsset",
    "VelocityManifestError",
    "collect_velocity_assets",
    "compute_sha256",
    "build_asset_hashes",
    "build_baseline_handoff_evidence",
    "build_velocity_manifest_data",
    "build_velocity_summary_data",
    "write_json_stable",
    "build_velocity_manifest",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
