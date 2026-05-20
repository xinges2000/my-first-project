from __future__ import annotations

import json
import math
from dataclasses import MISSING, dataclass, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import pandas as pd
from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator

from .types import (
    Detection,
    EventHypothesis,
    Pick,
    Station,
    TRAVELTIME_READY_REQUIRED_FILES,
    VELOCITY_EQUIV_MODEL_TYPE,
    VELOCITY_REQUIRED_PHASES,
    VELOCITY_V1_Z_POSITIVE,
    VelocityLayerTableRow,
    WellpathPoint,
)


SCHEMA_REGISTRY: Dict[str, str] = {
    "velocity_model": "velocity_model.schema.json",
}

SUPPORTED_VERSIONS: Dict[str, Sequence[str]] = {}


@dataclass(frozen=True)
class SchemaError:
    schema_name: str
    message: str
    json_path: str = ""
    validator: str = ""
    validator_value: Any = None
    instance: Any = None
    context: Tuple["SchemaError", ...] = ()

    def one_line(self) -> str:
        p = self.json_path or "/"
        v = f" [{self.validator}]" if self.validator else ""
        return f"{self.schema_name}: {p}{v} {self.message}"


class SchemaValidationError(ValueError):
    def __init__(self, schema_name: str, errors: List[SchemaError]):
        self.schema_name = schema_name
        self.errors = errors
        super().__init__(self._build_msg())

    def _build_msg(self) -> str:
        head = f"Schema validation failed: {self.schema_name} ({len(self.errors)} errors)"
        lines = [head] + ["  - " + e.one_line() for e in self.errors[:20]]
        if len(self.errors) > 20:
            lines.append(f"  ... ({len(self.errors) - 20} more)")
        return "\n".join(lines)


def _schemas_dir() -> Path:
    return Path(__file__).resolve().parent / "schemas"


def get_schema_path(schema_name: str) -> Path:
    if schema_name not in SCHEMA_REGISTRY:
        raise KeyError(f"Unknown schema_name={schema_name!r}. Known: {list(SCHEMA_REGISTRY.keys())}")
    return _schemas_dir() / SCHEMA_REGISTRY[schema_name]


@lru_cache(maxsize=64)
def load_schema(schema_name: str) -> Dict[str, Any]:
    path = get_schema_path(schema_name)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=64)
def _get_validator(schema_name: str) -> Validator:
    schema = load_schema(schema_name)
    return Draft202012Validator(schema)


def assert_supported_version(schema_name: str, version: Optional[str]) -> None:
    allowed = SUPPORTED_VERSIONS.get(schema_name)
    if not allowed:
        return
    if version is None:
        raise ValueError(f"{schema_name}: missing 'version' but supported versions are: {list(allowed)}")
    if version not in allowed:
        raise ValueError(f"{schema_name}: unsupported version {version!r}; supported: {list(allowed)}")


def _to_schema_error(schema_name: str, err) -> SchemaError:
    path = "/" + "/".join(str(x) for x in list(err.absolute_path)) if err.absolute_path else "/"
    ctx = tuple(_to_schema_error(schema_name, c) for c in getattr(err, "context", []) or [])
    return SchemaError(
        schema_name=schema_name,
        message=str(err.message),
        json_path=path,
        validator=str(getattr(err, "validator", "") or ""),
        validator_value=getattr(err, "validator_value", None),
        instance=getattr(err, "instance", None),
        context=ctx,
    )


def validate(schema_name: str, data: Dict[str, Any], *, strict: bool = True) -> List[SchemaError]:
    if isinstance(data, dict):
        assert_supported_version(schema_name, data.get("version"))

    v = _get_validator(schema_name)
    errors = sorted(v.iter_errors(data), key=lambda e: list(e.absolute_path))
    return [_to_schema_error(schema_name, e) for e in errors]


def validate_or_raise(schema_name: str, data: Dict[str, Any], *, strict: bool = True) -> None:
    errs = validate(schema_name, data, strict=strict)
    if errs:
        raise SchemaValidationError(schema_name, errs)


CSVContract = Dict[str, object]


def _dataclass_csv_contract(cls: Type[Any]) -> Tuple[List[str], Dict[str, object]]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")

    cols: List[str] = []
    defaults: Dict[str, object] = {}

    for f in fields(cls):
        cols.append(f.name)
        if f.default is not MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
            defaults[f.name] = f.default_factory()  # type: ignore[misc]

    return cols, defaults


def _make_csv_contract(
    cls: Type[Any],
    *,
    required_columns: Optional[Sequence[str]] = None,
    aliases: Optional[Mapping[str, str]] = None,
) -> CSVContract:
    columns, defaults = _dataclass_csv_contract(cls)
    return {
        "columns": columns,
        "defaults": defaults,
        "required_columns": list(required_columns) if required_columns is not None else [c for c in columns if c not in defaults],
        "aliases": dict(aliases or {}),
    }


STATIONS_CONTRACT = _make_csv_contract(
    Station,
    required_columns=["station_id", "X_m", "Y_m", "Z_m"],
)

DETECTIONS_CONTRACT = _make_csv_contract(Detection)
PICKS_CONTRACT = _make_csv_contract(Pick)
EVENTS_CONTRACT = _make_csv_contract(EventHypothesis)


VELOCITY_LAYER_TABLE_CONTRACT = _make_csv_contract(
    VelocityLayerTableRow,
    required_columns=["depth_top", "depth_bottom", "vp", "vs"],
)

WELLPATH_ALIASES: Dict[str, str] = {
    "md_m": "MD",
    "incl_deg": "Inc_deg",
    "azimuth_deg": "Az_deg",
    "tvd_m": "TVD_m",
    "x_m": "X_m",
    "y_m": "Y_m",
    "z_m": "Z_m",
    "n_gk_m": "N_gk_m",
    "e_gk_m": "E_gk_m",
}

WELLPATH_CONTRACT = _make_csv_contract(
    WellpathPoint,
    required_columns=["MD", "TVD_m", "X_m", "Y_m", "Z_m"],
    aliases=WELLPATH_ALIASES,
)

STATIONS_CSV_COLUMNS = list(STATIONS_CONTRACT["columns"])
STATIONS_CSV_DEFAULTS = dict(STATIONS_CONTRACT["defaults"])

WELLPATH_CSV_COLUMNS = list(WELLPATH_CONTRACT["columns"])
WELLPATH_CSV_DEFAULTS = dict(WELLPATH_CONTRACT["defaults"])

DETECTIONS_CSV_COLUMNS = list(DETECTIONS_CONTRACT["columns"])
DETECTIONS_CSV_DEFAULTS = dict(DETECTIONS_CONTRACT["defaults"])

PICKS_CSV_COLUMNS = list(PICKS_CONTRACT["columns"])
PICKS_CSV_DEFAULTS = dict(PICKS_CONTRACT["defaults"])

EVENTS_CSV_COLUMNS = list(EVENTS_CONTRACT["columns"])
EVENTS_CSV_DEFAULTS = dict(EVENTS_CONTRACT["defaults"])


VELOCITY_LAYER_CSV_COLUMNS = list(VELOCITY_LAYER_TABLE_CONTRACT["columns"])
VELOCITY_LAYER_CSV_DEFAULTS = dict(VELOCITY_LAYER_TABLE_CONTRACT["defaults"])


CSV_CONTRACTS: Dict[str, CSVContract] = {
    "stations": STATIONS_CONTRACT,
    "wellpath": WELLPATH_CONTRACT,
    "detections": DETECTIONS_CONTRACT,
    "picks": PICKS_CONTRACT,
    "events": EVENTS_CONTRACT,
    "velocity_layer_table": VELOCITY_LAYER_TABLE_CONTRACT,
}


def get_csv_contract(name: str) -> CSVContract:
    try:
        return CSV_CONTRACTS[name]
    except KeyError as e:
        raise KeyError(f"Unknown csv contract {name!r}. Known: {list(CSV_CONTRACTS.keys())}") from e


def required_columns_from_contract(
    columns: Sequence[str],
    defaults: Mapping[str, object],
    required_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    if required_columns is not None:
        return list(required_columns)
    return [c for c in columns if c not in defaults]


def normalize_csv_columns(
    df: pd.DataFrame,
    contract_name: str,
    *,
    strict_alias_conflict: bool = True,
) -> pd.DataFrame:
    contract = get_csv_contract(contract_name)
    aliases = dict(contract.get("aliases", {}))
    if not aliases:
        return df.copy()

    rename_map: Dict[str, str] = {}

    for src, dst in aliases.items():
        if src in df.columns and src != dst:
            if strict_alias_conflict and dst in df.columns:
                raise ValueError(
                    f"CSV alias conflict in contract={contract_name!r}: "
                    f"both alias column {src!r} and canonical column {dst!r} exist"
                )
            rename_map[src] = dst

    return df.rename(columns=rename_map)


def check_csv_contract(df: pd.DataFrame, contract_name: str) -> Dict[str, object]:
    contract = get_csv_contract(contract_name)
    df_norm = normalize_csv_columns(df, contract_name)

    columns = list(contract["columns"])
    defaults = dict(contract["defaults"])
    required = required_columns_from_contract(
        columns,
        defaults,
        contract.get("required_columns"),
    )

    actual = list(df_norm.columns)

    missing_required = [c for c in required if c not in actual]
    missing_optional = [c for c in columns if c not in actual and c in defaults]
    extra_columns = [c for c in actual if c not in columns]
    null_in_required = [c for c in required if c in df_norm.columns and df_norm[c].isna().any()]

    return {
        "contract_name": contract_name,
        "columns": columns,
        "required_columns": required,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "extra_columns": extra_columns,
        "null_in_required": null_in_required,
    }


def ensure_csv_columns(
    df: pd.DataFrame,
    columns: List[str],
    *,
    defaults: Mapping[str, object] | None = None,
    allow_extra: bool = True,
) -> pd.DataFrame:
    if defaults is None:
        defaults = {}

    out = df.copy()

    for c in columns:
        if c not in out.columns:
            out[c] = defaults.get(c, None)

    if allow_extra:
        extra = [c for c in out.columns if c not in columns]
        out = out[columns + extra]
    else:
        out = out[columns]

    return out


def ensure_csv_contract(
    df: pd.DataFrame,
    contract_name: str,
    *,
    allow_extra: bool = True,
) -> pd.DataFrame:
    contract = get_csv_contract(contract_name)
    df_norm = normalize_csv_columns(df, contract_name)
    return ensure_csv_columns(
        df_norm,
        list(contract["columns"]),
        defaults=dict(contract["defaults"]),
        allow_extra=allow_extra,
    )


def save_csv_contract(
    df: pd.DataFrame,
    csv_path: str | Path,
    columns: List[str],
    *,
    defaults: Mapping[str, object] | None = None,
    allow_extra: bool = True,
    float_format: str = "%.6f",
) -> Path:
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = ensure_csv_columns(df, columns, defaults=defaults, allow_extra=allow_extra)
    out.to_csv(p, index=False, float_format=float_format)
    return p


def save_csv_by_contract(
    df: pd.DataFrame,
    csv_path: str | Path,
    contract_name: str,
    *,
    allow_extra: bool = True,
    float_format: str = "%.6f",
) -> Path:
    contract = get_csv_contract(contract_name)
    df_norm = normalize_csv_columns(df, contract_name)
    return save_csv_contract(
        df_norm,
        csv_path,
        list(contract["columns"]),
        defaults=dict(contract["defaults"]),
        allow_extra=allow_extra,
        float_format=float_format,
    )


# =========================
# velocity V1 contract helpers
# =========================

VELOCITY_EQUIV_LAYERED_SCHEMA_NAME = "velocity_equiv_layered"
VELOCITY_LAYER_TABLE_SCHEMA_NAME = "velocity_layer_table"
TRAVELTIME_READY_SCHEMA_NAME = "traveltime_ready"

VELOCITY_EQUIV_LAYERED_REQUIRED_KEYS: Tuple[str, ...] = (
    "model_type",
    "z_positive",
    "units",
    "phases",
    "layers",
)
VELOCITY_EQUIV_LAYER_REQUIRED_KEYS: Tuple[str, ...] = (
    "layer_id",
    "z_top_m",
    "z_bot_m",
    "v_mps",
)
VELOCITY_REQUIRED_UNIT_ITEMS: Dict[str, str] = {
    "length": "m",
    "velocity": "m/s",
}


def _add_schema_error(
    errors: List[SchemaError],
    schema_name: str,
    message: str,
    *,
    json_path: str = "",
) -> None:
    errors.append(SchemaError(schema_name=schema_name, message=message, json_path=json_path))


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def check_velocity_equiv_layered_contract(obj: Any) -> Dict[str, object]:
    """
    Check the velocity V1 P/S dual-phase equivalent layered contract.

    This contract intentionally lives outside the legacy ``velocity_model``
    jsonschema path because the V1 ``velocity_model_equiv_layered.json`` format
    is a dedicated handoff artifact for ``traveltime_ready``.
    """
    errors: List[SchemaError] = []
    warnings: List[str] = []

    if not isinstance(obj, Mapping):
        _add_schema_error(
            errors,
            VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
            f"Expected mapping, got {type(obj)!r}",
            json_path="/",
        )
        return {
            "contract_name": VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
            "valid": False,
            "errors": errors,
            "warnings": warnings,
        }

    missing_required = [k for k in VELOCITY_EQUIV_LAYERED_REQUIRED_KEYS if k not in obj]
    for key in missing_required:
        _add_schema_error(
            errors,
            VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
            f"Missing required key {key!r}",
            json_path=f"/{key}",
        )

    if obj.get("model_type") != VELOCITY_EQUIV_MODEL_TYPE:
        _add_schema_error(
            errors,
            VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
            f"model_type must be {VELOCITY_EQUIV_MODEL_TYPE!r}",
            json_path="/model_type",
        )

    if obj.get("z_positive") != VELOCITY_V1_Z_POSITIVE:
        _add_schema_error(
            errors,
            VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
            f"z_positive must be {VELOCITY_V1_Z_POSITIVE!r} for velocity V1",
            json_path="/z_positive",
        )

    units = obj.get("units")
    if not isinstance(units, Mapping):
        _add_schema_error(
            errors,
            VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
            "units must be a mapping",
            json_path="/units",
        )
    else:
        for key, expected in VELOCITY_REQUIRED_UNIT_ITEMS.items():
            if units.get(key) != expected:
                _add_schema_error(
                    errors,
                    VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                    f"units.{key} must be {expected!r}",
                    json_path=f"/units/{key}",
                )

    phases = obj.get("phases")
    if not _is_non_string_sequence(phases):
        _add_schema_error(
            errors,
            VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
            "phases must be a non-string sequence",
            json_path="/phases",
        )
    else:
        phase_set = set(phases)
        missing_phases = [phase for phase in VELOCITY_REQUIRED_PHASES if phase not in phase_set]
        if missing_phases:
            _add_schema_error(
                errors,
                VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                f"phases must include {list(VELOCITY_REQUIRED_PHASES)!r}; missing {missing_phases!r}",
                json_path="/phases",
            )

    layers = obj.get("layers")
    if not isinstance(layers, Mapping):
        if any(k in obj for k in ("velocity", "v", "velocity_mps")):
            _add_schema_error(
                errors,
                VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                "single-velocity fallback is not allowed; require layers.P and layers.S",
                json_path="/layers",
            )
        else:
            _add_schema_error(
                errors,
                VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                "layers must be a mapping with keys 'P' and 'S'",
                json_path="/layers",
            )
    else:
        for phase in VELOCITY_REQUIRED_PHASES:
            phase_layers = layers.get(phase)
            if phase_layers is None:
                _add_schema_error(
                    errors,
                    VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                    f"Missing required phase layer list {phase!r}",
                    json_path=f"/layers/{phase}",
                )
                continue
            if not _is_non_string_sequence(phase_layers) or len(phase_layers) == 0:
                _add_schema_error(
                    errors,
                    VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                    f"layers.{phase} must be a non-empty sequence",
                    json_path=f"/layers/{phase}",
                )
                continue

            for idx, layer in enumerate(phase_layers):
                layer_path = f"/layers/{phase}/{idx}"
                if not isinstance(layer, Mapping):
                    _add_schema_error(
                        errors,
                        VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                        "layer item must be a mapping",
                        json_path=layer_path,
                    )
                    continue

                for key in VELOCITY_EQUIV_LAYER_REQUIRED_KEYS:
                    if key not in layer:
                        _add_schema_error(
                            errors,
                            VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                            f"Missing required layer key {key!r}",
                            json_path=f"{layer_path}/{key}",
                        )

                if "layer_id" in layer and layer.get("layer_id") in (None, ""):
                    _add_schema_error(
                        errors,
                        VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                        "layer_id must be non-empty",
                        json_path=f"{layer_path}/layer_id",
                    )

                z_top = _safe_float(layer.get("z_top_m"))
                z_bot = _safe_float(layer.get("z_bot_m"))
                v_mps = _safe_float(layer.get("v_mps"))

                if "z_top_m" in layer and z_top is None:
                    _add_schema_error(
                        errors,
                        VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                        "z_top_m must be a finite number",
                        json_path=f"{layer_path}/z_top_m",
                    )
                if "z_bot_m" in layer and z_bot is None:
                    _add_schema_error(
                        errors,
                        VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                        "z_bot_m must be a finite number",
                        json_path=f"{layer_path}/z_bot_m",
                    )
                if z_top is not None and z_bot is not None and z_bot <= z_top:
                    _add_schema_error(
                        errors,
                        VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                        "z_bot_m must be greater than z_top_m",
                        json_path=layer_path,
                    )

                if "v_mps" in layer and v_mps is None:
                    _add_schema_error(
                        errors,
                        VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                        "v_mps must be a finite number",
                        json_path=f"{layer_path}/v_mps",
                    )
                elif v_mps is not None and v_mps <= 0:
                    _add_schema_error(
                        errors,
                        VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
                        "v_mps must be > 0",
                        json_path=f"{layer_path}/v_mps",
                    )

    return {
        "contract_name": VELOCITY_EQUIV_LAYERED_SCHEMA_NAME,
        "valid": not errors,
        "missing_required": missing_required,
        "errors": errors,
        "warnings": warnings,
    }


def assert_velocity_equiv_layered_contract(obj: Any) -> None:
    report = check_velocity_equiv_layered_contract(obj)
    errors = list(report["errors"])
    if errors:
        raise SchemaValidationError(VELOCITY_EQUIV_LAYERED_SCHEMA_NAME, errors)


def check_velocity_layer_table_contract(df: pd.DataFrame) -> Dict[str, object]:
    report = check_csv_contract(df, "velocity_layer_table")
    errors: List[SchemaError] = []
    warnings: List[str] = []

    for col in report["missing_required"]:
        _add_schema_error(
            errors,
            VELOCITY_LAYER_TABLE_SCHEMA_NAME,
            f"Missing required column {col!r}",
            json_path=f"/{col}",
        )
    for col in report["null_in_required"]:
        _add_schema_error(
            errors,
            VELOCITY_LAYER_TABLE_SCHEMA_NAME,
            f"Required column {col!r} contains null values",
            json_path=f"/{col}",
        )

    numeric_cols = [c for c in VELOCITY_LAYER_CSV_COLUMNS if c in df.columns]
    numeric_df = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        numeric_df[col] = numeric_series
        invalid_mask = numeric_series.isna() | ~numeric_series.map(math.isfinite)
        if invalid_mask.any():
            _add_schema_error(
                errors,
                VELOCITY_LAYER_TABLE_SCHEMA_NAME,
                f"Column {col!r} contains non-numeric or non-finite values at rows {list(df.index[invalid_mask])!r}",
                json_path=f"/{col}",
            )

    required_for_semantics = set(VELOCITY_LAYER_CSV_COLUMNS)
    if required_for_semantics.issubset(numeric_df.columns):
        finite_mask = numeric_df[list(VELOCITY_LAYER_CSV_COLUMNS)].applymap(math.isfinite).all(axis=1)
        valid_rows = numeric_df.loc[finite_mask]

        invalid_bounds = valid_rows.index[valid_rows["depth_bottom"] <= valid_rows["depth_top"]]
        if len(invalid_bounds) > 0:
            _add_schema_error(
                errors,
                VELOCITY_LAYER_TABLE_SCHEMA_NAME,
                f"depth_bottom must be greater than depth_top at rows {list(invalid_bounds)!r}",
                json_path="/depth_bottom",
            )

        invalid_vp = valid_rows.index[valid_rows["vp"] <= 0]
        if len(invalid_vp) > 0:
            _add_schema_error(
                errors,
                VELOCITY_LAYER_TABLE_SCHEMA_NAME,
                f"vp must be > 0 at rows {list(invalid_vp)!r}",
                json_path="/vp",
            )

        invalid_vs = valid_rows.index[valid_rows["vs"] <= 0]
        if len(invalid_vs) > 0:
            _add_schema_error(
                errors,
                VELOCITY_LAYER_TABLE_SCHEMA_NAME,
                f"vs must be > 0 at rows {list(invalid_vs)!r}",
                json_path="/vs",
            )

        vs_not_less_than_vp = valid_rows.index[valid_rows["vs"] >= valid_rows["vp"]]
        if len(vs_not_less_than_vp) > 0:
            warnings.append(
                f"vs is usually expected to be less than vp; check rows {list(vs_not_less_than_vp)!r}"
            )

    return {
        **report,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def assert_velocity_layer_table_contract(df: pd.DataFrame) -> None:
    report = check_velocity_layer_table_contract(df)
    errors = list(report["errors"])
    if errors:
        raise SchemaValidationError(VELOCITY_LAYER_TABLE_SCHEMA_NAME, errors)


def check_traveltime_ready_contract(path: str | Path) -> Dict[str, object]:
    root = Path(path)
    errors: List[SchemaError] = []
    missing_required_files: List[str] = []

    if not root.exists():
        _add_schema_error(
            errors,
            TRAVELTIME_READY_SCHEMA_NAME,
            f"traveltime_ready directory does not exist: {root}",
            json_path="/",
        )
    elif not root.is_dir():
        _add_schema_error(
            errors,
            TRAVELTIME_READY_SCHEMA_NAME,
            f"traveltime_ready path is not a directory: {root}",
            json_path="/",
        )
    else:
        missing_required_files = [
            name for name in TRAVELTIME_READY_REQUIRED_FILES if not (root / name).is_file()
        ]
        for name in missing_required_files:
            _add_schema_error(
                errors,
                TRAVELTIME_READY_SCHEMA_NAME,
                f"Missing required file {name!r}",
                json_path=f"/{name}",
            )

    return {
        "contract_name": TRAVELTIME_READY_SCHEMA_NAME,
        "valid": not errors,
        "required_files": list(TRAVELTIME_READY_REQUIRED_FILES),
        "missing_required_files": missing_required_files,
        "errors": errors,
    }


def assert_traveltime_ready_contract(path: str | Path) -> None:
    report = check_traveltime_ready_contract(path)
    errors = list(report["errors"])
    if errors:
        raise SchemaValidationError(TRAVELTIME_READY_SCHEMA_NAME, errors)


ExtraValidator = Callable[[Any], List[SchemaError]]
EXTRA_VALIDATORS: Dict[str, ExtraValidator] = {}


def register_extra_validator(schema_name: str):
    def _wrap(fn: ExtraValidator) -> ExtraValidator:
        EXTRA_VALIDATORS[schema_name] = fn
        return fn
    return _wrap


def validate_bundle(schema_name: str, obj: Any, *, strict: bool = True) -> List[SchemaError]:
    errs: List[SchemaError] = []

    if isinstance(obj, dict):
        errs.extend(validate(schema_name, obj, strict=strict))
    else:
        errs.append(SchemaError(schema_name, f"Expected dict for schema validation, got {type(obj)}"))
        return errs

    extra = EXTRA_VALIDATORS.get(schema_name)
    if extra is not None:
        errs.extend(extra(obj))
    return errs


def validate_bundle_or_raise(schema_name: str, obj: Any, *, strict: bool = True) -> None:
    errs = validate_bundle(schema_name, obj, strict=strict)
    if errs:
        raise SchemaValidationError(schema_name, errs)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def _load_npy(path: Path):
    import numpy as np
    return np.load(path)


def _expected_shape(model_type: str, grid: Dict[str, Any]) -> Tuple[int, ...]:
    shape = grid.get("shape", {})
    nx = int(shape.get("nx", 1))
    ny = int(shape.get("ny", 1))
    nz = int(shape.get("nz", 1))
    if model_type == "1d_layered":
        return (nz,)
    return (nz, ny, nx)


@register_extra_validator("velocity_model")
def _extra_validate_velocity_model(vm: Dict[str, Any]) -> List[SchemaError]:
    errs: List[SchemaError] = []
    schema_name = "velocity_model"

    base_dir = vm.get("_base_dir")
    if base_dir is None:
        errs.append(SchemaError(schema_name, "Missing vm['_base_dir'] for file checks; skip file validation."))
        return errs

    base = Path(base_dir)
    files = vm.get("files", {})
    fmt = files.get("format", "npy")
    dtype_decl = str(files.get("dtype", "float32"))
    model_type = str(vm.get("model_type", ""))
    grid = vm.get("grid", {})

    field_paths: List[Tuple[str, Any]] = [
        ("vp", files.get("vp")),
        ("vs", files.get("vs")),
        ("rho", files.get("rho")),
    ]

    if fmt != "npy":
        errs.append(SchemaError(schema_name, f"Extra validator currently supports format='npy' only, got {fmt!r}"))
        return errs

    exp_shape = _expected_shape(model_type, grid)

    dtype_map = {
        "float32": "float32",
        "float64": "float64",
    }
    if dtype_decl not in dtype_map:
        errs.append(SchemaError(schema_name, f"Unsupported dtype in files.dtype: {dtype_decl!r}"))
        return errs

    for field, rel in field_paths:
        if rel in (None, "", False):
            continue
        p = base / str(rel)
        if not p.exists():
            errs.append(SchemaError(schema_name, f"Missing file for field '{field}': {p}", json_path="/files"))
            continue

        try:
            arr = _load_npy(p)
        except Exception as e:
            errs.append(SchemaError(schema_name, f"Failed to read npy for '{field}': {e}", json_path="/files"))
            continue

        if tuple(arr.shape) != tuple(exp_shape):
            errs.append(
                SchemaError(
                    schema_name,
                    f"Shape mismatch for '{field}': got {arr.shape}, expected {exp_shape} (convention: 1D=(nz,), 3D=(nz,ny,nx))",
                    json_path="/grid/shape",
                )
            )

        if str(arr.dtype) != dtype_decl:
            errs.append(
                SchemaError(
                    schema_name,
                    f"Dtype mismatch for '{field}': got {arr.dtype}, expected {dtype_decl}",
                    json_path="/files/dtype",
                )
            )

        if field in ("vp", "vs"):
            try:
                mn = float(arr.min())
                mx = float(arr.max())
                if not (math.isfinite(mn) and math.isfinite(mx)):
                    errs.append(SchemaError(schema_name, f"Non-finite values in '{field}'", json_path=f"/files/{field}"))
                if mn <= 0:
                    errs.append(SchemaError(schema_name, f"Non-positive min in '{field}': {mn}", json_path=f"/files/{field}"))
                if field == "vp" and mx > 12000:
                    errs.append(SchemaError(schema_name, f"Unusually large Vp max={mx} m/s", json_path=f"/files/{field}"))
                if field == "vs" and mx > 7000:
                    errs.append(SchemaError(schema_name, f"Unusually large Vs max={mx} m/s", json_path=f"/files/{field}"))
            except Exception:
                errs.append(SchemaError(schema_name, f"Cannot compute min/max for '{field}'", json_path=f"/files/{field}"))

    return errs


def validate_velocity_model_bundle(velocity_model_json: str | Path, *, strict: bool = True) -> List[SchemaError]:
    p = Path(velocity_model_json)
    vm = _read_json(p)
    vm["_base_dir"] = str(p.parent)
    return validate_bundle("velocity_model", vm, strict=strict)


def validate_velocity_model_bundle_or_raise(velocity_model_json: str | Path, *, strict: bool = True) -> None:
    errs = validate_velocity_model_bundle(velocity_model_json, strict=strict)
    if errs:
        raise SchemaValidationError("velocity_model", errs)
