from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, Literal

import numpy as np


PhaseType = Literal["P", "S"]
CoordFrame = Literal["proj_xy_m"]  # 工程投影米制（主）


@dataclass(frozen=True)
class Station:
    """
    台站基础 contract（baseline 主坐标约定）

    主坐标：
        - X_m = Northing (m)
        - Y_m = Easting  (m)
        - Z_m = Elevation / engineering Z (m)

    说明：
        - Step 1 contract 收口后，stations.csv 的主必需列为：
          station_id, X_m, Y_m, Z_m
        - 其中 Z_m 虽然保留 dataclass 默认值 0.0，
          但在 schema contract 中被显式提升为 required column

    兼容字段：
        - E_gk_m / N_gk_m：保留用于兼容既有高斯-克吕格表达
        - lon_deg / lat_deg：保留用于展示/回查
        - inline_3D：保留用于 3D 采集线 / 施工侧辅助字段
    """
    station_id: str
    X_m: float
    Y_m: float
    Z_m: float = 0.0
    E_gk_m: Optional[float] = None
    N_gk_m: Optional[float] = None
    lon_deg: Optional[float] = None
    lat_deg: Optional[float] = None
    n_traces: int = 0
    components_present: str = ""
    inline_3D: Optional[float] = None


@dataclass(frozen=True)
class WellpathPoint:
    """
    井轨迹基础 contract（按当前 baseline 真实 CSV 样例收口）

    当前 canonical CSV 列顺序：
        MD, Inc_deg, Az_deg, dE_m, dN_m, TVD_m, X_m, Y_m, Z_m, N_gk_m, E_gk_m

    字段语义：
        - MD: measured depth，单位 m
        - Inc_deg / Az_deg: 井斜 / 方位角，单位 degree
        - dE_m / dN_m: 相对井口的 Easting / Northing 偏移，单位 m
        - TVD_m: true vertical depth，向下为正，单位 m
        - X_m / Y_m: 工程主坐标，约定 X=Northing, Y=Easting
        - Z_m: 工程 Z；在当前样例中表现为“高程型 Z（up positive）”
        - N_gk_m / E_gk_m: 高斯-克吕格兼容列；当前样例中与 X_m / Y_m 数值一致

    说明：
        1. common 层在 Step 1 只负责收口列 contract，不在此处强行把 Z 正方向统一成 down positive
        2. 若后续 build/runtime 需要统一 z_positive="down"，应在读取/转换阶段显式完成
    """
    MD: float
    Inc_deg: float = 0.0
    Az_deg: float = 0.0
    dE_m: float = 0.0
    dN_m: float = 0.0
    TVD_m: Optional[float] = None
    X_m: Optional[float] = None
    Y_m: Optional[float] = None
    Z_m: Optional[float] = None
    N_gk_m: Optional[float] = None
    E_gk_m: Optional[float] = None


Wellpath = Tuple[WellpathPoint, ...]


# =========================
# velocity V1 contract primitives
# =========================

VelocityDepthAxis = Literal["TVD_m", "MD_m", "Z_m"]
VelocityZPositive = Literal["down", "up"]
VelocityModelType = Literal["1d_continuous_engineering", "1d_layered_equiv_traveltime"]
VelocityModelRole = Literal["ref_engineering", "equiv_layered", "traveltime_ready"]

VELOCITY_EQUIV_MODEL_TYPE: VelocityModelType = "1d_layered_equiv_traveltime"
VELOCITY_V1_Z_POSITIVE: VelocityZPositive = "down"
VELOCITY_REQUIRED_PHASES: Tuple[PhaseType, ...] = ("P", "S")
TRAVELTIME_READY_REQUIRED_FILES: Tuple[str, ...] = (
    "velocity_model_equiv_layered.json",
    "depth.npy",
    "vp.npy",
    "vs.npy",
    "asset_hashes.json",
)


@dataclass(frozen=True)
class VelocityLayerTableRow:
    """Readable table contract for one 1D equivalent velocity layer row."""
    depth_top: float
    depth_bottom: float
    vp: float
    vs: float


@dataclass(frozen=True)
class VelocityLayerContract:
    """
    Minimal layer-level contract used by velocity V1 JSON artifacts.

    The phase is expressed by the parent path ``layers.P`` / ``layers.S``.
    Therefore each layer item does not require a duplicated ``phase`` field.
    """
    layer_id: int
    z_top_m: float
    z_bot_m: float
    v_mps: float


@dataclass(frozen=True)
class VelocityEquivLayeredContract:
    """Top-level P/S dual-phase contract for the V1 equivalent layered model."""
    model_type: VelocityModelType
    z_positive: VelocityZPositive
    phases: Tuple[PhaseType, ...]
    units: Dict[str, str]
    layers: Dict[PhaseType, Tuple[VelocityLayerContract, ...]]


@dataclass(frozen=True)
class TraveltimeReadyContract:
    """Minimal file-set contract exposed by velocity V1 to traveltime."""
    required_files: Tuple[str, ...]


@dataclass(frozen=True)
class WaveformTrace:
    station_id: str
    channel: str
    fs_hz: float
    t0_s: float
    data: np.ndarray  # (n,)


@dataclass(frozen=True)
class Detection:
    station_id: str
    t_start_s: float
    t_end_s: float
    score: float


@dataclass(frozen=True)
class Pick:
    station_id: str
    phase: PhaseType
    t_pick_s: float
    uncertainty_s: float = 0.02
    snr: Optional[float] = None
    method: str = "baseline"


@dataclass(frozen=True)
class EventHypothesis:
    event_id: str
    t0_s: float
    X_m: float
    Y_m: float
    Z_m: float
    rms_s: Optional[float] = None
    n_picks: int = 0
    quality: Optional[str] = None


@dataclass(frozen=True)
class ProjectGeometry:
    coord_frame: CoordFrame
    stations: Tuple[Station, ...]
    proj_meta: Optional[Dict[str, object]] = None


class Detector(Protocol):
    def detect(self, traces: Sequence[WaveformTrace]) -> List[Detection]:
        ...


class Picker(Protocol):
    def pick(self, traces: Sequence[WaveformTrace], detections: Sequence[Detection]) -> List[Pick]:
        ...


class Associator(Protocol):
    def associate(self, picks: Sequence[Pick]) -> Dict[str, List[Pick]]:
        ...


class VelocityModel(Protocol):
    def travel_time_s(
        self,
        src_xyz_m: Tuple[float, float, float],
        sta_xyz_m: Tuple[float, float, float],
        phase: PhaseType,
    ) -> float:
        ...


class Locator(Protocol):
    def locate(
        self,
        event_id: str,
        picks: Sequence[Pick],
        geom: ProjectGeometry,
        vel: VelocityModel,
        x0: Tuple[float, float, float, float] | None = None,  # (t0, X, Y, Z)
    ) -> EventHypothesis:
        ...
