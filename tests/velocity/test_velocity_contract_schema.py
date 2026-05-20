from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from microseis_ds.common.schema import (
    SchemaValidationError,
    TRAVELTIME_READY_REQUIRED_FILES,
    VELOCITY_EQUIV_LAYER_REQUIRED_KEYS,
    VELOCITY_LAYER_CSV_COLUMNS,
    assert_traveltime_ready_contract,
    assert_velocity_equiv_layered_contract,
    assert_velocity_layer_table_contract,
)
from microseis_ds.common.types import (
    VELOCITY_EQUIV_MODEL_TYPE,
    VELOCITY_REQUIRED_PHASES,
    VELOCITY_V1_Z_POSITIVE,
)


def _valid_equiv_contract() -> dict[str, object]:
    return {
        "model_type": VELOCITY_EQUIV_MODEL_TYPE,
        "z_positive": VELOCITY_V1_Z_POSITIVE,
        "units": {
            "length": "m",
            "velocity": "m/s",
        },
        "phases": list(VELOCITY_REQUIRED_PHASES),
        "layers": {
            "P": [
                {
                    "layer_id": 1,
                    "z_top_m": 0.0,
                    "z_bot_m": 100.0,
                    "v_mps": 2500.0,
                }
            ],
            "S": [
                {
                    "layer_id": 1,
                    "z_top_m": 0.0,
                    "z_bot_m": 100.0,
                    "v_mps": 1400.0,
                }
            ],
        },
    }


def test_velocity_v1_frozen_constants_match_required_contract() -> None:
    assert VELOCITY_EQUIV_MODEL_TYPE == "1d_layered_equiv_traveltime"
    assert VELOCITY_V1_Z_POSITIVE == "down"
    assert tuple(VELOCITY_REQUIRED_PHASES) == ("P", "S")
    assert VELOCITY_LAYER_CSV_COLUMNS == ["depth_top", "depth_bottom", "vp", "vs"]
    assert tuple(VELOCITY_EQUIV_LAYER_REQUIRED_KEYS) == (
        "layer_id",
        "z_top_m",
        "z_bot_m",
        "v_mps",
    )
    assert tuple(TRAVELTIME_READY_REQUIRED_FILES) == (
        "velocity_model_equiv_layered.json",
        "depth.npy",
        "vp.npy",
        "vs.npy",
        "asset_hashes.json",
    )


def test_equiv_contract_accepts_valid_ps_dual_phase_model_without_layer_phase_field() -> None:
    model = _valid_equiv_contract()

    assert "phase" not in model["layers"]["P"][0]
    assert "phase" not in model["layers"]["S"][0]
    assert_velocity_equiv_layered_contract(model)


@pytest.mark.parametrize("missing_phase", ["P", "S"])
def test_equiv_contract_rejects_missing_phase_in_top_level_phases(missing_phase: str) -> None:
    model = _valid_equiv_contract()
    model["phases"] = [phase for phase in model["phases"] if phase != missing_phase]

    with pytest.raises(SchemaValidationError):
        assert_velocity_equiv_layered_contract(model)


@pytest.mark.parametrize("missing_phase", ["P", "S"])
def test_equiv_contract_rejects_missing_phase_layer_list(missing_phase: str) -> None:
    model = _valid_equiv_contract()
    del model["layers"][missing_phase]

    with pytest.raises(SchemaValidationError):
        assert_velocity_equiv_layered_contract(model)


def test_equiv_contract_rejects_single_velocity_fallback() -> None:
    model = {
        "model_type": VELOCITY_EQUIV_MODEL_TYPE,
        "z_positive": VELOCITY_V1_Z_POSITIVE,
        "units": {
            "length": "m",
            "velocity": "m/s",
        },
        "phases": list(VELOCITY_REQUIRED_PHASES),
        "velocity": 2500.0,
    }

    with pytest.raises(SchemaValidationError):
        assert_velocity_equiv_layered_contract(model)


@pytest.mark.parametrize("missing_key", ["layer_id", "z_top_m", "z_bot_m", "v_mps"])
def test_equiv_contract_requires_minimal_layer_fields(missing_key: str) -> None:
    model = _valid_equiv_contract()
    del model["layers"]["P"][0][missing_key]

    with pytest.raises(SchemaValidationError):
        assert_velocity_equiv_layered_contract(model)


def test_equiv_contract_rejects_invalid_layer_bounds() -> None:
    model = _valid_equiv_contract()
    model["layers"]["P"][0]["z_bot_m"] = 0.0

    with pytest.raises(SchemaValidationError):
        assert_velocity_equiv_layered_contract(model)


@pytest.mark.parametrize("invalid_velocity", [0.0, -1.0])
def test_equiv_contract_rejects_nonpositive_velocity(invalid_velocity: float) -> None:
    model = _valid_equiv_contract()
    model["layers"]["S"][0]["v_mps"] = invalid_velocity

    with pytest.raises(SchemaValidationError):
        assert_velocity_equiv_layered_contract(model)


def test_equiv_contract_rejects_invalid_model_type() -> None:
    model = _valid_equiv_contract()
    model["model_type"] = "1d_layered"

    with pytest.raises(SchemaValidationError):
        assert_velocity_equiv_layered_contract(model)


def test_equiv_contract_rejects_invalid_z_positive() -> None:
    model = _valid_equiv_contract()
    model["z_positive"] = "up"

    with pytest.raises(SchemaValidationError):
        assert_velocity_equiv_layered_contract(model)


def test_velocity_layer_table_contract_accepts_required_columns() -> None:
    df = pd.DataFrame(
        {
            "depth_top": [0.0, 100.0],
            "depth_bottom": [100.0, 200.0],
            "vp": [2500.0, 2700.0],
            "vs": [1400.0, 1500.0],
        }
    )

    assert_velocity_layer_table_contract(df)


@pytest.mark.parametrize("missing_column", ["depth_top", "depth_bottom", "vp", "vs"])
def test_velocity_layer_table_contract_rejects_missing_required_columns(missing_column: str) -> None:
    df = pd.DataFrame(
        {
            "depth_top": [0.0],
            "depth_bottom": [100.0],
            "vp": [2500.0],
            "vs": [1400.0],
        }
    ).drop(columns=[missing_column])

    with pytest.raises(SchemaValidationError):
        assert_velocity_layer_table_contract(df)


def test_velocity_layer_table_contract_rejects_invalid_bounds() -> None:
    df = pd.DataFrame(
        {
            "depth_top": [100.0],
            "depth_bottom": [100.0],
            "vp": [2500.0],
            "vs": [1400.0],
        }
    )

    with pytest.raises(SchemaValidationError):
        assert_velocity_layer_table_contract(df)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("vp", 0.0),
        ("vs", -1.0),
    ],
)
def test_velocity_layer_table_contract_rejects_nonpositive_velocity(
    column: str,
    invalid_value: float,
) -> None:
    df = pd.DataFrame(
        {
            "depth_top": [0.0],
            "depth_bottom": [100.0],
            "vp": [2500.0],
            "vs": [1400.0],
        }
    )
    df.loc[0, column] = invalid_value

    with pytest.raises(SchemaValidationError):
        assert_velocity_layer_table_contract(df)


def test_traveltime_ready_contract_accepts_minimal_complete_directory(tmp_path) -> None:
    ready_dir = tmp_path / "traveltime_ready"
    ready_dir.mkdir()

    for name in TRAVELTIME_READY_REQUIRED_FILES:
        (ready_dir / name).touch()

    assert_traveltime_ready_contract(ready_dir)


@pytest.mark.parametrize("missing_file", list(TRAVELTIME_READY_REQUIRED_FILES))
def test_traveltime_ready_contract_rejects_missing_required_file(tmp_path, missing_file: str) -> None:
    ready_dir = tmp_path / "traveltime_ready"
    ready_dir.mkdir()

    for name in TRAVELTIME_READY_REQUIRED_FILES:
        if name != missing_file:
            (ready_dir / name).touch()

    with pytest.raises(SchemaValidationError):
        assert_traveltime_ready_contract(ready_dir)
