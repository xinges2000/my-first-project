from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd
import pytest


def _ns(mapping: dict[str, Any]) -> SimpleNamespace:
    """Recursively convert a mapping into a SimpleNamespace."""
    converted: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            converted[key] = _ns(value)
        else:
            converted[key] = value
    return SimpleNamespace(**converted)


def _tests_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _tests_dir().parent


def _asset_path(*parts: str) -> Path:
    return _repo_root().joinpath("tests", "assets", "baseline", *parts)


def _velocity_asset_path(*parts: str) -> Path:
    return _repo_root().joinpath("tests", "assets", "velocity", *parts)


@pytest.fixture(scope="session", autouse=True)
def _force_mpl_agg() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session", autouse=True)
def _ensure_repo_root_on_sys_path() -> None:
    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# =========================
# 环境层 fixtures
# =========================

@pytest.fixture(scope="session")
def test_assets_root() -> Path:
    """Static baseline asset root committed in the repository."""
    root = _asset_path()
    if not root.exists():
        raise FileNotFoundError(
            f"Baseline test asset root does not exist: {root}. "
            "Expected fixed repository layout: tests/assets/baseline/"
        )
    return root


@pytest.fixture(scope="function")
def test_run_root(tmp_path: Path) -> Path:
    """Per-test isolated run root."""
    run_root = tmp_path / "prepared_project"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


@pytest.fixture(scope="function")
def baseline_module_out_dir(test_run_root: Path) -> Path:
    """Per-test baseline module output directory."""
    path = test_run_root / "baseline"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="function")
def baseline_qc_out_dir(test_run_root: Path) -> Path:
    """
    Per-test QC output directory for baseline module.

    Route B: this fixture represents runtime-generated QC outputs, not
    pre-baked sample artifacts under tests/assets/baseline/qc.
    """
    path = test_run_root / "qc" / "baseline"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def velocity_test_assets_root() -> Path:
    """Static velocity asset root committed in the repository."""
    root = _velocity_asset_path()
    if not root.exists():
        raise FileNotFoundError(
            f"Velocity test asset root does not exist: {root}. "
            "Expected fixed repository layout: tests/assets/velocity/"
        )
    return root


@pytest.fixture(scope="function")
def velocity_module_out_dir(test_run_root: Path) -> Path:
    """Per-test velocity module output directory."""
    path = test_run_root / "velocity"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="function")
def velocity_qc_out_dir(test_run_root: Path) -> Path:
    """Per-test QC output directory for velocity module."""
    path = test_run_root / "qc" / "velocity"
    path.mkdir(parents=True, exist_ok=True)
    return path


# =========================
# 样例层 fixtures
# =========================

@pytest.fixture(scope="session")
def baseline_sample_catalog(test_assets_root: Path) -> SimpleNamespace:
    """
    Stable input sample catalog for Step 6.

    Route B keeps the automated mainline focused on input assets:
    - normal: normal builder inputs
    - negative: negative builder inputs

    The on-repo qc/ directory may still exist as reference evidence, but it is
    not treated as a primary automated input fixture here.
    """
    catalog = {
        "root": test_assets_root,
        "normal": {
            "single_well": {
                "gps_txt": test_assets_root / "normal" / "2026-01-28_10_46_GPS-h345.txt",
                "wellheads_csv": test_assets_root / "normal" / "W204H85-3-wellheads-Zone18.csv",
                "survey_csvs": [
                    test_assets_root / "normal" / "W204H85-3_survey.csv",
                ],
                "survey_well_ids": ["W204H85-3"],
            },
            "multi_well": {
                "gps_txt": test_assets_root / "normal" / "2026-01-28_10_46_GPS-h345.txt",
                "wellheads_csv": test_assets_root / "normal" / "W204H85-wellheads-4well-Zone18.csv",
                "survey_csvs": [
                    test_assets_root / "normal" / "W204H85-3_survey.csv",
                    test_assets_root / "normal" / "W204H85-4_survey.csv",
                ],
                "survey_well_ids": ["W204H85-3", "W204H85-4"],
            },
        },
        "negative": {
            "build_project": {
                "duplicate_station_id": {
                    "gps_txt": test_assets_root / "negative" / "2026-01-28_10_46_GPS-h345-duplicate_station_id.txt",
                },
                "missing_z": {
                    "gps_txt": test_assets_root / "negative" / "2026-01-28_10_46_GPS-h345-missing-Z.txt",
                },
            },
            "build_wellpath": {
                "invalid_survey": {
                    "survey_csv": test_assets_root / "negative" / "W204H85-3_survey-feifa.csv",
                    "well_id": "W204H85-3",
                },
            },
        },
    }
    return _ns(catalog)


@pytest.fixture(scope="session")
def baseline_normal_inputs(baseline_sample_catalog: SimpleNamespace) -> SimpleNamespace:
    """Normal-path input samples anchored under tests/assets/baseline/normal."""
    return baseline_sample_catalog.normal


@pytest.fixture(scope="session")
def baseline_negative_inputs(baseline_sample_catalog: SimpleNamespace) -> SimpleNamespace:
    """Negative-path input samples anchored under tests/assets/baseline/negative."""
    return baseline_sample_catalog.negative


@pytest.fixture(scope="session")
def velocity_sample_catalog(velocity_test_assets_root: Path) -> SimpleNamespace:
    """
    Minimal velocity input sample catalog for Step 2 wiring.

    This fixture only exposes stable input paths. Business pass/fail semantics
    remain in later velocity tests and are intentionally not encoded here.
    """
    catalog = {
        "root": velocity_test_assets_root,
        "normal": {
            "ref_engineering": {
                "engineering_log_csv": (
                    velocity_test_assets_root / "normal" / "engineering_log_ps_valid.csv"
                ),
            },
        },
        "negative": {
            "ref_engineering": {
                "missing_depth": {
                    "engineering_log_csv": (
                        velocity_test_assets_root / "negative" / "engineering_log_missing_depth.csv"
                    ),
                },
                "missing_vp": {
                    "engineering_log_csv": (
                        velocity_test_assets_root / "negative" / "engineering_log_missing_vp.csv"
                    ),
                },
                "missing_vs": {
                    "engineering_log_csv": (
                        velocity_test_assets_root / "negative" / "engineering_log_missing_vs.csv"
                    ),
                },
                "nonpositive_velocity": {
                    "engineering_log_csv": (
                        velocity_test_assets_root / "negative" / "engineering_log_nonpositive_velocity.csv"
                    ),
                },
                "nonmonotonic_depth": {
                    "engineering_log_csv": (
                        velocity_test_assets_root / "negative" / "engineering_log_nonmonotonic_depth.csv"
                    ),
                },
            },
        },
    }
    return _ns(catalog)


@pytest.fixture(scope="session")
def velocity_normal_inputs(velocity_sample_catalog: SimpleNamespace) -> SimpleNamespace:
    """Normal-path input samples anchored under tests/assets/velocity/normal."""
    return velocity_sample_catalog.normal


@pytest.fixture(scope="session")
def velocity_negative_inputs(velocity_sample_catalog: SimpleNamespace) -> SimpleNamespace:
    """Negative-path input samples anchored under tests/assets/velocity/negative."""
    return velocity_sample_catalog.negative


# =========================
# 读取层 fixtures
# =========================

@pytest.fixture(scope="session")
def read_json_file() -> Callable[[Path], dict[str, Any]]:
    def _read(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    return _read


@pytest.fixture(scope="session")
def read_csv_file() -> Callable[[Path], pd.DataFrame]:
    def _read(path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return _read
