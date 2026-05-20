from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microseis_ds.velocity.runtime.velocity_model import (
    EquivalentLayeredVelocityModel,
    VelocityModelError,
    load_equiv_layered_velocity_model,
)


def _velocity_assets_root() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "velocity"


def _normal_equiv_dir() -> Path:
    return _velocity_assets_root() / "normal"


def _normal_equiv_json() -> Path:
    return _normal_equiv_dir() / "velocity_model_equiv_layered.json"


def _negative_equiv_json(name: str) -> Path:
    return _velocity_assets_root() / "negative" / name


def test_runtime_loads_equiv_layered_json_from_directory_and_file_path() -> None:
    """Normal path: the Step 3 handoff file can be loaded by directory or file path."""
    model_from_dir = load_equiv_layered_velocity_model(_normal_equiv_dir())
    model_from_file = load_equiv_layered_velocity_model(_normal_equiv_json())

    assert isinstance(model_from_dir, EquivalentLayeredVelocityModel)
    assert isinstance(model_from_file, EquivalentLayeredVelocityModel)
    assert model_from_dir.meta["model_type"] == "1d_layered_equiv_traveltime"
    assert model_from_dir.meta["z_positive"] == "down"
    assert model_from_dir.meta["phases"] == ["P", "S"]

    assert model_from_dir.depth_range("P") == (0.0, 200.0)
    assert model_from_dir.depth_range("S") == (0.0, 200.0)


def test_runtime_exposes_p_and_s_layers_with_contract_fields() -> None:
    """P/S phase layer access is explicit and preserves required layer fields."""
    model = load_equiv_layered_velocity_model(_normal_equiv_json())

    p_layers = model.get_layers("P")
    s_layers = model.get_layers("S")

    assert len(p_layers) == 2
    assert len(s_layers) == 2

    assert p_layers[0].layer_id == 0
    assert p_layers[0].z_top_m == 0.0
    assert p_layers[0].z_bot_m == 100.0
    assert p_layers[0].v_mps == 3000.0

    assert s_layers[1].layer_id == 1
    assert s_layers[1].z_top_m == 100.0
    assert s_layers[1].z_bot_m == 200.0
    assert s_layers[1].v_mps == 1900.0

    layer_dict = model.to_layer_dict()
    assert set(layer_dict.keys()) == {"P", "S"}
    assert set(layer_dict["P"][0].keys()) == {"layer_id", "z_top_m", "z_bot_m", "v_mps"}


def test_runtime_queries_phase_and_joint_vp_vs_by_depth() -> None:
    """Runtime exposes minimum traveltime-facing velocity query behavior."""
    model = load_equiv_layered_velocity_model(_normal_equiv_json())

    z = np.array([0.0, 99.999, 100.0, 200.0], dtype=np.float64)
    vp = model.query_phase(z, "P")
    vs = model.query_phase(z, "S")
    joint_vp, joint_vs = model.query(z)

    assert np.allclose(vp, [3000.0, 3000.0, 3500.0, 3500.0])
    assert np.allclose(vs, [1700.0, 1700.0, 1900.0, 1900.0])
    assert np.allclose(joint_vp, vp)
    assert np.allclose(joint_vs, vs)


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("velocity_model_equiv_layered_missing_layer_id.json", "layer_id"),
        ("velocity_model_equiv_layered_missing_z_top_m.json", "z_top_m"),
        ("velocity_model_equiv_layered_missing_z_bot_m.json", "z_bot_m"),
        ("velocity_model_equiv_layered_missing_v_mps.json", "v_mps"),
        ("velocity_model_equiv_layered_missing_z_positive.json", "z_positive"),
        ("velocity_model_equiv_layered_missing_phases.json", "phases"),
    ],
)
def test_runtime_fails_on_missing_required_contract_fields(filename: str, message: str) -> None:
    """Contract fields must fail hard instead of being silently filled."""
    with pytest.raises(VelocityModelError, match=message):
        load_equiv_layered_velocity_model(_negative_equiv_json(filename))


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("velocity_model_equiv_layered_invalid_interval.json", "z_bot_m"),
        ("velocity_model_equiv_layered_nonpositive_v_mps.json", "v_mps"),
        ("velocity_model_equiv_layered_gap_in_P_layers.json", "contiguous"),
    ],
)
def test_runtime_fails_on_invalid_layer_geometry_or_velocity(filename: str, message: str) -> None:
    """Layer geometry and velocities are minimum runtime validity checks."""
    with pytest.raises(VelocityModelError, match=message):
        load_equiv_layered_velocity_model(_negative_equiv_json(filename))


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("velocity_model_equiv_layered_missing_P.json", r"layers\.P"),
        ("velocity_model_equiv_layered_missing_S.json", r"layers\.S"),
    ],
)
def test_runtime_fails_when_required_phase_layers_are_missing(filename: str, message: str) -> None:
    """Both P and S phase layer sequences are required for velocity V1."""
    with pytest.raises(VelocityModelError, match=message):
        load_equiv_layered_velocity_model(_negative_equiv_json(filename))


def test_runtime_rejects_invalid_phase_queries() -> None:
    """Invalid phase names fail with a clear runtime error."""
    model = load_equiv_layered_velocity_model(_normal_equiv_json())

    with pytest.raises(VelocityModelError, match="Unsupported phase"):
        model.get_layers("X")

    with pytest.raises(VelocityModelError, match="Unsupported phase"):
        model.query_phase([10.0], "X")


def test_runtime_rejects_out_of_range_depth_without_silent_clipping() -> None:
    """Out-of-range depth query fails unless the caller explicitly asks for extrapolation."""
    model = load_equiv_layered_velocity_model(_normal_equiv_json())

    with pytest.raises(VelocityModelError, match="out of range"):
        model.query_phase([-1.0], "P")

    with pytest.raises(VelocityModelError, match="out of range"):
        model.query([201.0])

    vp, vs = model.query([-1.0, 201.0], extrapolate=True)
    assert np.allclose(vp, [3000.0, 3500.0])
    assert np.allclose(vs, [1700.0, 1900.0])


def test_runtime_errors_when_equiv_json_filename_is_not_available(tmp_path: Path) -> None:
    """Directory handoff requires the stable Step 3 filename."""
    with pytest.raises(VelocityModelError, match="velocity_model_equiv_layered.json not found"):
        load_equiv_layered_velocity_model(tmp_path)
