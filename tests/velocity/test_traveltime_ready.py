from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from microseis_ds.velocity.build.build_traveltime_ready import (
    ASSET_HASHES_FILENAME,
    DEPTH_FILENAME,
    INTERFACE_NOTE_FILENAME,
    MAIN_JSON_FILENAME,
    QC_TRAVELTIME_EQUIV_FILENAME,
    VP_FILENAME,
    VS_FILENAME,
    TraveltimeReadyBuildError,
    build_traveltime_ready,
)
from microseis_ds.velocity.runtime.velocity_traveltime_adapter import (
    TraveltimeReadyAdapterError,
    load_traveltime_ready_velocity,
)


READY_REQUIRED_FILES = (
    MAIN_JSON_FILENAME,
    DEPTH_FILENAME,
    VP_FILENAME,
    VS_FILENAME,
    ASSET_HASHES_FILENAME,
    INTERFACE_NOTE_FILENAME,
    QC_TRAVELTIME_EQUIV_FILENAME,
)


def _copy_normal_equiv_json(src_assets_root: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    src = src_assets_root / "normal" / MAIN_JSON_FILENAME
    dst = dst_dir / MAIN_JSON_FILENAME
    shutil.copy2(src, dst)
    return dst


def _copy_negative_equiv_json(src_assets_root: Path, name: str, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    src = src_assets_root / "negative" / name
    dst = dst_dir / MAIN_JSON_FILENAME
    shutil.copy2(src, dst)
    return dst


def _load_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"expected JSON file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_items_by_path(asset_hashes: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    items = asset_hashes.get("items")
    assert isinstance(items, list)
    paths = [str(item.get("path", "")) for item in items]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert all("\\" not in path for path in paths)
    return {str(item["path"]): item for item in items}


def _assert_ready_file_set(ready_dir: Path) -> None:
    for name in READY_REQUIRED_FILES:
        assert (ready_dir / name).is_file(), f"traveltime_ready missing {name}"


def test_traveltime_ready_build_derives_arrays_when_equiv_arrays_are_missing(
    test_run_root: Path,
    velocity_test_assets_root: Path,
) -> None:
    source_dir = test_run_root / "velocity" / "equiv_layered"
    ready_dir = test_run_root / "velocity" / "traveltime_ready"
    _copy_normal_equiv_json(velocity_test_assets_root, source_dir)

    result = build_traveltime_ready(equiv_layered_dir=source_dir, out_dir=ready_dir)

    assert result["status"] == "completed"
    assert result["array_source"] == "derived_from_layers"
    assert set(result["missing_source_arrays"]) == {DEPTH_FILENAME, VP_FILENAME, VS_FILENAME}
    _assert_ready_file_set(ready_dir)

    depth = np.load(ready_dir / DEPTH_FILENAME)
    vp = np.load(ready_dir / VP_FILENAME)
    vs = np.load(ready_dir / VS_FILENAME)

    assert np.allclose(depth, [0.0, 100.0, 200.0])
    assert np.allclose(vp, [3000.0, 3500.0, 3500.0])
    assert np.allclose(vs, [1700.0, 1900.0, 1900.0])

    note = (ready_dir / INTERFACE_NOTE_FILENAME).read_text(encoding="utf-8")
    assert "velocity traveltime_ready interface" in note
    assert "does not contain traveltime tables" in note

    qc = _load_json(ready_dir / QC_TRAVELTIME_EQUIV_FILENAME)
    assert qc["module"] == "velocity"
    assert qc["check_name"] == "traveltime_ready_equiv_interface"
    assert qc["array_source"] == "derived_from_layers"
    assert qc["warnings"]


def test_traveltime_ready_build_copies_complete_equiv_arrays_without_derivation(
    test_run_root: Path,
    velocity_test_assets_root: Path,
) -> None:
    source_dir = test_run_root / "velocity" / "equiv_layered"
    ready_dir = test_run_root / "velocity" / "traveltime_ready"
    _copy_normal_equiv_json(velocity_test_assets_root, source_dir)

    depth = np.array([0.0, 50.0, 100.0, 200.0], dtype=np.float64)
    vp = np.array([3000.0, 3200.0, 3500.0, 3500.0], dtype=np.float64)
    vs = np.array([1700.0, 1800.0, 1900.0, 1900.0], dtype=np.float64)
    np.save(source_dir / DEPTH_FILENAME, depth)
    np.save(source_dir / VP_FILENAME, vp)
    np.save(source_dir / VS_FILENAME, vs)
    (source_dir / QC_TRAVELTIME_EQUIV_FILENAME).write_text(
        json.dumps({"schema_version": "1.0.0", "status": "passed", "source": "unit-test"}),
        encoding="utf-8",
    )

    result = build_traveltime_ready(equiv_layered_dir=source_dir, out_dir=ready_dir)

    assert result["array_source"] == "copied_from_equiv_layered"
    assert result["missing_source_arrays"] == []
    assert result["copied_source_qc"] is True
    assert np.allclose(np.load(ready_dir / DEPTH_FILENAME), depth)
    assert np.allclose(np.load(ready_dir / VP_FILENAME), vp)
    assert np.allclose(np.load(ready_dir / VS_FILENAME), vs)
    assert _load_json(ready_dir / QC_TRAVELTIME_EQUIV_FILENAME)["source"] == "unit-test"


def test_traveltime_ready_asset_hashes_are_stable_and_recomputable(
    test_run_root: Path,
    velocity_test_assets_root: Path,
) -> None:
    source_dir = test_run_root / "velocity" / "equiv_layered"
    ready_dir = test_run_root / "velocity" / "traveltime_ready"
    _copy_normal_equiv_json(velocity_test_assets_root, source_dir)

    build_traveltime_ready(equiv_layered_dir=source_dir, out_dir=ready_dir)

    asset_hashes = _load_json(ready_dir / ASSET_HASHES_FILENAME)
    assert asset_hashes["algorithm"] == "sha256"
    hash_items = _hash_items_by_path(asset_hashes)

    expected_hashed_files = {
        MAIN_JSON_FILENAME,
        DEPTH_FILENAME,
        VP_FILENAME,
        VS_FILENAME,
        INTERFACE_NOTE_FILENAME,
        QC_TRAVELTIME_EQUIV_FILENAME,
    }
    assert set(hash_items) == expected_hashed_files
    assert int(asset_hashes["count"]) == len(expected_hashed_files)

    for relative_path, item in hash_items.items():
        asset_path = ready_dir / relative_path
        assert int(item["size_bytes"]) == asset_path.stat().st_size
        assert str(item["sha256"]) == _sha256(asset_path)


def test_traveltime_ready_adapter_consumes_ready_directory(
    test_run_root: Path,
    velocity_test_assets_root: Path,
) -> None:
    source_dir = test_run_root / "velocity" / "equiv_layered"
    ready_dir = test_run_root / "velocity" / "traveltime_ready"
    _copy_normal_equiv_json(velocity_test_assets_root, source_dir)
    build_traveltime_ready(equiv_layered_dir=source_dir, out_dir=ready_dir)

    view = load_traveltime_ready_velocity(ready_dir)

    assert view.ready_dir == ready_dir
    assert view.model.meta["model_type"] == "1d_layered_equiv_traveltime"
    assert view.depth_m.shape == view.vp_mps.shape == view.vs_mps.shape
    assert view.asset_hashes["algorithm"] == "sha256"
    assert view.qc is not None

    vp, vs = view.query([0.0, 100.0, 200.0])
    assert np.allclose(vp, [3000.0, 3500.0, 3500.0])
    assert np.allclose(vs, [1700.0, 1900.0, 1900.0])

    descriptor = view.to_traveltime_dict()
    assert descriptor["main_json"].endswith(MAIN_JSON_FILENAME)
    assert descriptor["z_positive"] == "down"
    assert descriptor["phases"] == ["P", "S"]
    assert descriptor["sample_count"] == 3


def test_traveltime_ready_build_fails_when_main_json_is_missing(test_run_root: Path) -> None:
    source_dir = test_run_root / "velocity" / "equiv_layered"
    ready_dir = test_run_root / "velocity" / "traveltime_ready"
    source_dir.mkdir(parents=True)

    with pytest.raises(TraveltimeReadyBuildError, match=MAIN_JSON_FILENAME):
        build_traveltime_ready(equiv_layered_dir=source_dir, out_dir=ready_dir)

    assert not ready_dir.exists() or not (ready_dir / MAIN_JSON_FILENAME).exists()


def test_traveltime_ready_build_fails_on_invalid_equiv_contract(
    test_run_root: Path,
    velocity_test_assets_root: Path,
) -> None:
    source_dir = test_run_root / "velocity" / "equiv_layered"
    ready_dir = test_run_root / "velocity" / "traveltime_ready"
    _copy_negative_equiv_json(
        velocity_test_assets_root,
        "velocity_model_equiv_layered_missing_P.json",
        source_dir,
    )

    with pytest.raises(TraveltimeReadyBuildError, match=r"layers\.P"):
        build_traveltime_ready(equiv_layered_dir=source_dir, out_dir=ready_dir)


def test_traveltime_ready_adapter_rejects_hash_mismatch(
    test_run_root: Path,
    velocity_test_assets_root: Path,
) -> None:
    source_dir = test_run_root / "velocity" / "equiv_layered"
    ready_dir = test_run_root / "velocity" / "traveltime_ready"
    _copy_normal_equiv_json(velocity_test_assets_root, source_dir)
    build_traveltime_ready(equiv_layered_dir=source_dir, out_dir=ready_dir)

    (ready_dir / INTERFACE_NOTE_FILENAME).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(TraveltimeReadyAdapterError, match="sha256 mismatch"):
        load_traveltime_ready_velocity(ready_dir)


def test_traveltime_ready_build_does_not_modify_source_equiv_layered_directory(
    test_run_root: Path,
    velocity_test_assets_root: Path,
) -> None:
    source_dir = test_run_root / "velocity" / "equiv_layered"
    ready_dir = test_run_root / "velocity" / "traveltime_ready"
    source_json = _copy_normal_equiv_json(velocity_test_assets_root, source_dir)
    before = sorted(path.name for path in source_dir.iterdir())
    source_hash_before = _sha256(source_json)

    build_traveltime_ready(equiv_layered_dir=source_dir, out_dir=ready_dir)

    after = sorted(path.name for path in source_dir.iterdir())
    source_hash_after = _sha256(source_json)
    assert after == before == [MAIN_JSON_FILENAME]
    assert source_hash_after == source_hash_before
