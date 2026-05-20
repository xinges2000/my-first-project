from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pytest

from scripts import build_velocity


FORMAL_ASSET_NAMES = (
    "velocity_model.json",
    "depth.npy",
    "vp.npy",
    "vs.npy",
)

CORE_MANIFEST_ASSET_PATHS = (
    "velocity/ref_engineering/depth.npy",
    "velocity/ref_engineering/velocity_model.json",
    "velocity/ref_engineering/vp.npy",
    "velocity/ref_engineering/vs.npy",
)

MANIFEST_REL_PATH = "velocity/velocity_manifest.json"
SUMMARY_REL_PATH = "velocity/meta/velocity_build_summary.json"


def _ref_engineering_dir(test_run_root: Path) -> Path:
    return test_run_root / "velocity" / "ref_engineering"


def _manifest_path(test_run_root: Path) -> Path:
    return test_run_root / "velocity" / "velocity_manifest.json"


def _summary_path(test_run_root: Path) -> Path:
    return test_run_root / "velocity" / "meta" / "velocity_build_summary.json"


def _load_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"expected JSON file to exist: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _path_from_manifest_relative(test_run_root: Path, relative_path: str) -> Path:
    assert relative_path, "manifest asset path should not be empty"
    assert "\\" not in relative_path, f"manifest path must use POSIX '/': {relative_path}"
    rel = Path(*relative_path.split("/"))
    assert not rel.is_absolute(), f"manifest path must be relative: {relative_path}"
    assert ".." not in rel.parts, f"manifest path must not escape output root: {relative_path}"
    return test_run_root / rel


def _compute_sha256(path: Path) -> str:
    assert path.is_file(), f"cannot hash missing asset: {path}"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_items_by_path(asset_hashes: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    items = asset_hashes.get("items")
    assert isinstance(items, list), "asset_hashes.items should be a list"

    paths = [str(item.get("path", "")) for item in items]
    assert paths == sorted(paths), "asset_hashes.items should be sorted by stable relative path"
    assert len(paths) == len(set(paths)), "asset_hashes.items should not contain duplicate paths"

    return {str(item["path"]): item for item in items}


def _run_ref_engineering_build(input_csv: Path, test_run_root: Path) -> Path:
    """
    Run the public velocity build entry in the narrow Step 3.4 scope.

    The test intentionally uses task=engineering only, so it verifies the
    ref_engineering public build path without triggering export_ref_tvd,
    layerize, package, QC, runtime, or traveltime_ready behavior.
    """
    rc = build_velocity.main(
        [
            "--task",
            "engineering",
            "--out_root",
            str(test_run_root),
            "--module_name",
            "velocity",
            "--log_csv",
            str(input_csv),
        ]
    )
    assert rc == 0
    return _ref_engineering_dir(test_run_root)


def _assert_formal_assets_exist(ref_dir: Path) -> None:
    for name in FORMAL_ASSET_NAMES:
        assert (ref_dir / name).is_file(), f"expected formal asset to exist: {name}"


def _assert_formal_assets_absent(ref_dir: Path) -> None:
    for name in FORMAL_ASSET_NAMES:
        assert not (ref_dir / name).exists(), (
            f"formal asset should not be generated for invalid input: {name}"
        )


def _assert_message_contains_all(message: str, parts: Iterable[str]) -> None:
    for part in parts:
        assert part in message, f"expected error message to contain {part!r}; got: {message!r}"


def _assert_ref_engineering_fails(
    input_csv: Path,
    test_run_root: Path,
    expected_message_parts: Iterable[str],
) -> None:
    ref_dir = _ref_engineering_dir(test_run_root)

    with pytest.raises(ValueError) as exc_info:
        build_velocity.main(
            [
                "--task",
                "engineering",
                "--out_root",
                str(test_run_root),
                "--module_name",
                "velocity",
                "--log_csv",
                str(input_csv),
            ]
        )

    _assert_message_contains_all(str(exc_info.value), expected_message_parts)
    _assert_formal_assets_absent(ref_dir)


def test_ref_engineering_build_from_ps_log_writes_dual_phase_assets_and_metadata(
    test_run_root: Path,
    velocity_normal_inputs,
) -> None:
    input_csv = velocity_normal_inputs.ref_engineering.engineering_log_csv

    ref_dir = _run_ref_engineering_build(input_csv=input_csv, test_run_root=test_run_root)

    _assert_formal_assets_exist(ref_dir)

    depth = np.load(ref_dir / "depth.npy")
    vp = np.load(ref_dir / "vp.npy")
    vs = np.load(ref_dir / "vs.npy")
    model = json.loads((ref_dir / "velocity_model.json").read_text(encoding="utf-8"))

    assert len(depth) == len(vp) == len(vs)
    assert len(depth) >= 2
    assert np.all(np.diff(depth) > 0)
    assert np.all(vp > 0)
    assert np.all(vs > 0)

    assert model["fields"]["vp"] is True
    assert model["fields"]["vs"] is True
    assert model["files"]["vp"] == "vp.npy"
    assert model["files"]["vs"] == "vs.npy"
    assert model["grid"]["shape"]["nz"] == len(depth)

    vs_source = str(model["provenance"]["vs_source"])
    assert "from column" in vs_source
    assert "computed" not in vs_source


def test_ref_engineering_build_writes_manifest_summary_hash_with_stable_paths(
    test_run_root: Path,
    velocity_normal_inputs,
) -> None:
    input_csv = velocity_normal_inputs.ref_engineering.engineering_log_csv

    _run_ref_engineering_build(input_csv=input_csv, test_run_root=test_run_root)

    manifest_path = _manifest_path(test_run_root)
    summary_path = _summary_path(test_run_root)
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)

    assert manifest["module"] == "velocity"
    assert summary["module"] == "velocity"
    assert manifest["status"] == summary["status"] == "completed"
    assert manifest["build_summary_path"] == SUMMARY_REL_PATH
    assert summary["manifest_path"] == MANIFEST_REL_PATH
    assert manifest["meta_outputs"] == [SUMMARY_REL_PATH]

    asset_hashes = manifest["asset_hashes"]
    assert asset_hashes["algorithm"] == "sha256"
    hash_items = _hash_items_by_path(asset_hashes)

    assert int(asset_hashes["count"]) == len(hash_items)
    assert int(summary["asset_count"]) == len(hash_items)
    assert int(summary["asset_hashes_count"]) == len(hash_items)

    expected_core_paths = set(CORE_MANIFEST_ASSET_PATHS)
    assert expected_core_paths.issubset(set(hash_items))
    assert expected_core_paths.issubset(set(manifest["primary_outputs"]))
    assert expected_core_paths.issubset(set(manifest["generated_outputs"]))
    assert expected_core_paths.issubset(set(summary["generated_outputs"]["primary_outputs"]))

    assert MANIFEST_REL_PATH not in hash_items
    assert SUMMARY_REL_PATH not in hash_items

    for relative_path in CORE_MANIFEST_ASSET_PATHS:
        item = hash_items[relative_path]
        asset_path = _path_from_manifest_relative(test_run_root, relative_path)
        assert int(item["size_bytes"]) == asset_path.stat().st_size
        assert str(item["sha256"]) == _compute_sha256(asset_path)

    required_status = summary["required_asset_status"]
    assert required_status["status"] == "complete"
    assert int(required_status["required_count"]) == len(CORE_MANIFEST_ASSET_PATHS)
    assert int(required_status["present_count"]) == len(CORE_MANIFEST_ASSET_PATHS)
    assert int(required_status["missing_count"]) == 0
    assert set(required_status["present"]) == expected_core_paths
    assert required_status["missing"] == []

    for item in required_status["items"]:
        assert item["group"] == "ref_engineering"
        assert item["path"] in expected_core_paths
        assert item["status"] == "present"

    assert manifest["warnings"] == []
    assert manifest["errors"] == []
    assert summary["warnings"] == []
    assert summary["errors"] == []


def test_ref_engineering_missing_depth_column_fails(
    test_run_root: Path,
    velocity_negative_inputs,
) -> None:
    input_csv = velocity_negative_inputs.ref_engineering.missing_depth.engineering_log_csv

    _assert_ref_engineering_fails(
        input_csv=input_csv,
        test_run_root=test_run_root,
        expected_message_parts=("depth column not found",),
    )


def test_ref_engineering_missing_vp_column_fails(
    test_run_root: Path,
    velocity_negative_inputs,
) -> None:
    input_csv = velocity_negative_inputs.ref_engineering.missing_vp.engineering_log_csv

    _assert_ref_engineering_fails(
        input_csv=input_csv,
        test_run_root=test_run_root,
        expected_message_parts=("vp column not found",),
    )


def test_ref_engineering_missing_vs_column_fails(
    test_run_root: Path,
    velocity_negative_inputs,
) -> None:
    input_csv = velocity_negative_inputs.ref_engineering.missing_vs.engineering_log_csv

    _assert_ref_engineering_fails(
        input_csv=input_csv,
        test_run_root=test_run_root,
        expected_message_parts=("vs column not found",),
    )


def test_ref_engineering_nonmonotonic_depth_fails_without_reordering(
    test_run_root: Path,
    velocity_negative_inputs,
) -> None:
    input_csv = velocity_negative_inputs.ref_engineering.nonmonotonic_depth.engineering_log_csv

    _assert_ref_engineering_fails(
        input_csv=input_csv,
        test_run_root=test_run_root,
        expected_message_parts=("depth", "strictly increasing"),
    )


def test_ref_engineering_nonpositive_vp_fails(
    test_run_root: Path,
    velocity_test_assets_root: Path,
) -> None:
    input_csv = velocity_test_assets_root / "negative" / "engineering_log_nonpositive_vp.csv"
    assert input_csv.is_file(), "Step 3.4 nonpositive_vp sample should exist"

    _assert_ref_engineering_fails(
        input_csv=input_csv,
        test_run_root=test_run_root,
        expected_message_parts=("vp", "> 0"),
    )


def test_ref_engineering_nonpositive_vs_fails(
    test_run_root: Path,
    velocity_test_assets_root: Path,
) -> None:
    input_csv = velocity_test_assets_root / "negative" / "engineering_log_nonpositive_vs.csv"
    assert input_csv.is_file(), "Step 3.4 nonpositive_vs sample should exist"

    _assert_ref_engineering_fails(
        input_csv=input_csv,
        test_run_root=test_run_root,
        expected_message_parts=("vs", "> 0"),
    )


def _write_text_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json_file(path: Path, obj: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _make_formal_baseline_handoff_assets(run_root: Path) -> Path:
    baseline_root = run_root / "baseline"
    _write_text_file(
        baseline_root / "stations.csv",
        "station_id,X_m,Y_m,Z_m\nSTA001,1000.0,2000.0,10.0\nSTA002,1100.0,2100.0,11.0\n",
    )
    _write_text_file(
        baseline_root / "links" / "traceid_station.csv",
        "inline_3D,station_id\n1001,STA001\n1002,STA002\n",
    )
    _write_json_file(
        baseline_root / "meta" / "wellpaths_index.json",
        {"items": [], "row_count": 0, "mapping_policy": "unit-test"},
    )
    _write_json_file(
        baseline_root / "project.json",
        {
            "project_name": "unit_step8_build_manifest",
            "coordinate_system": {"name": "CGCS2000 / Gauss-Kruger", "unit": "m"},
        },
    )
    _write_json_file(
        baseline_root / "manifest.json",
        {
            "project_name": "unit_step8_build_manifest",
            "assets": {
                "stations": "stations.csv",
                "traceid_station": "links/traceid_station.csv",
                "wellpaths_index": "meta/wellpaths_index.json",
            },
        },
    )
    _write_json_file(baseline_root / "meta" / "baseline_build_summary.json", {"status": "passed"})
    _write_json_file(baseline_root / "meta" / "wellpath_build_summary.json", {"status": "passed", "records": []})
    return baseline_root


def test_build_velocity_manifest_records_baseline_handoff_hash_provenance_and_z_positive(
    test_run_root: Path,
    velocity_normal_inputs,
) -> None:
    baseline_root = _make_formal_baseline_handoff_assets(test_run_root)
    input_csv = velocity_normal_inputs.ref_engineering.engineering_log_csv

    rc = build_velocity.main(
        [
            "--task",
            "engineering",
            "--out_root",
            str(test_run_root),
            "--module_name",
            "velocity",
            "--log_csv",
            str(input_csv),
            "--baseline_root",
            str(baseline_root),
            "--z_positive",
            "down",
        ]
    )
    assert rc == 0

    manifest = _load_json(_manifest_path(test_run_root))
    summary = _load_json(_summary_path(test_run_root))
    handoff = manifest["baseline_handoff"]

    assert summary["baseline_handoff"] == handoff
    assert handoff["status"] == "passed"
    assert handoff["z_positive"] == "down"
    assert handoff["depth_positive"] == "down"
    assert handoff["boundary"]["velocity_side_evidence_only"] is True
    assert handoff["selected_path_policy"]["traceid_station"]["selection"] == "formal_primary"
    assert handoff["selected_path_policy"]["wellpaths_index"]["selection"] == "formal_primary"
    assert handoff["source_paths"]["traceid_station_path"].endswith("baseline/links/traceid_station.csv")
    assert handoff["source_paths"]["wellpaths_index_path"].endswith("baseline/meta/wellpaths_index.json")

    asset_hashes = handoff["asset_hashes"]
    assert asset_hashes["algorithm"] == "sha256"
    items = {str(item["role"]): item for item in asset_hashes["items"]}
    for role in (
        "stations",
        "traceid_station",
        "wellpaths_index",
        "project_json",
        "baseline_manifest",
        "baseline_build_summary",
    ):
        assert role in items
        asset_path = _path_from_manifest_relative(test_run_root, str(items[role]["path"]))
        assert str(items[role]["sha256"]) == _compute_sha256(asset_path)

    assert handoff["missing_mandatory_hash_roles"] == []
    assert handoff["direct_wellpath_policy"]["direct_wellpath_csv_read"] is False
