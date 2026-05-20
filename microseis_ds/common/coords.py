# microseis_ds/coords.py
# ----------------------- sxe start
# 1  这个程序正确判断了目前的坐标系是中国工程数据中常用的
#       高斯-克吕格带号拼接坐标
#    明确了判断依据，而不是EPSG 3857系统，也不是UTM坐标系统
#
# 2  此文件中包含了下列函数，进行各种坐标系与EPSG 3857系统的转换
# wm3857_to_lonlat
# lonlat_to_wm3857
# 以及 detect_coord_type() / wellhead_to_gk() 的 3857 分支
# 如果后期将要叠加在线底图（高德/天地图/OSM）：可以参考这部分，实现
# “外部 GIS 展示需求”，做一个转换模块
# ----------------------- sxe end
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Literal

try:
    from pyproj import CRS, Transformer
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pyproj is required. Install with: pip install pyproj\n"
        f"Import error: {e}"
    )

# ----------------------------
# China sanity bounds (rough)
# ----------------------------
# Mainland China approx bbox (lon, lat)
CHINA_LON_MIN, CHINA_LON_MAX = 73.0, 136.0
CHINA_LAT_MIN, CHINA_LAT_MAX = 18.0, 54.0


def _in_china_bbox(lon: float, lat: float) -> bool:
    return (CHINA_LON_MIN <= lon <= CHINA_LON_MAX) and (CHINA_LAT_MIN <= lat <= CHINA_LAT_MAX)


def _assert_in_china(lon: float, lat: float, ctx: str) -> None:
    if not _in_china_bbox(lon, lat):
        raise ValueError(
            f"[COORD ERROR] {ctx}: lon/lat out of China bbox: lon={lon:.6f}, lat={lat:.6f}. "
            f"Expected lon in [{CHINA_LON_MIN},{CHINA_LON_MAX}], lat in [{CHINA_LAT_MIN},{CHINA_LAT_MAX}]. "
            "This usually means wrong CRS / wrong units / X-Y swapped / or wrong GK zone interpretation."
        )


# ----------------------------
# CRS helpers
# ----------------------------
def make_transformer(src: str | CRS, dst: str | CRS) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


# EPSG:3857 <-> lon/lat
_T_3857_TO_4326 = make_transformer("EPSG:3857", "EPSG:4326")
_T_4326_TO_3857 = make_transformer("EPSG:4326", "EPSG:3857")


def wm3857_to_lonlat(x: float, y: float, *, validate_china: bool = True, ctx: str = "") -> Tuple[float, float]:
    lon, lat = _T_3857_TO_4326.transform(x, y)
    lon = float(lon)
    lat = float(lat)
    if validate_china:
        _assert_in_china(lon, lat, ctx or "EPSG:3857->lon/lat")
    return lon, lat


def lonlat_to_wm3857(lon: float, lat: float) -> Tuple[float, float]:
    x, y = _T_4326_TO_3857.transform(lon, lat)
    return float(x), float(y)


# ----------------------------
# GK config (CGCS2000 + Gauss-Kruger)
# ----------------------------
@dataclass(frozen=True)
class GKConfig:
    """
    CGCS2000 + Gauss-Kruger (Transverse Mercator).
    Use PROJ string so you don't need to guess EPSG.

    Typical parameters:
      - lon_0: central meridian (deg). (e.g., 105, 108, 111...)
      - x_0: false easting, usually 500000
      - y_0: false northing, usually 0
      - k: scale factor, usually 1.0
      - ellps: GRS80 for CGCS2000
    """
    lon_0: float
    x_0: float = 500000.0
    y_0: float = 0.0
    k: float = 1.0

    def proj4(self) -> str:
        return (
            f"+proj=tmerc +lat_0=0 +lon_0={self.lon_0} "
            f"+k={self.k} +x_0={self.x_0} +y_0={self.y_0} "
            f"+ellps=GRS80 +units=m +no_defs"
        )


def lonlat_to_gk(lon: float, lat: float, gk: GKConfig) -> Tuple[float, float]:
    t = make_transformer("EPSG:4326", CRS.from_proj4(gk.proj4()))
    e, n = t.transform(lon, lat)
    return float(e), float(n)


def gk_to_lonlat(e: float, n: float, gk: GKConfig, *, validate_china: bool = False, ctx: str = "") -> Tuple[float, float]:
    t = make_transformer(CRS.from_proj4(gk.proj4()), "EPSG:4326")
    lon, lat = t.transform(e, n)
    lon = float(lon)
    lat = float(lat)
    if validate_china:
        _assert_in_china(lon, lat, ctx or "GK->lon/lat")
    return lon, lat


# ----------------------------
# Auto recommend GK lon0 (3° / 6°)
# ----------------------------
def recommend_gk_lon0(lon_deg: float) -> Dict[str, float]:
    """
    Recommend Gauss–Krüger central meridian (lon0) for both 3° and 6° bands.

    IMPORTANT: There are TWO common "6° GK lon0" conventions in Chinese engineering data:

    (A) EPSG / standard 6° GK (recommended for strict geodesy & interoperability):
        - Zone definition: zone = floor(lon / 6) + 1    (1..60 for lon in [0,360))
        - Central meridian: lon0 = zone*6 - 3
        - Equivalent closed form (for lon in [0,180]): lon0 = 6*floor(lon/6) + 3
        - This yields lon0 in {..., 99, 105, 111, ...}  (i.e., 6k ± 3)

        Example: lon=104.86 -> floor(104.86/6)=17 -> lon0=6*17+3=105

    (B) Engineering "zone*6" convention (seen in some field tables / vendors):
        - Central meridian: lon0 = zone*6
        - This yields lon0 in {..., 96, 102, 108, ...}  (i.e., multiples of 6)
        - Example: zone=18 -> lon0=108
        - NOTE: This is NOT EPSG standard; only use if your vendor/project explicitly uses it.

    For 3° GK, the widely used central meridians are multiples of 3 (e.g., 105, 108, 111...).
    We provide a simple rounding-based recommendation that matches typical practice.

    Returns:
      {
        "3deg": <recommended lon0 for 3° GK>,
        "6deg_epsg": <recommended lon0 for 6° GK (EPSG/standard)>,
        "6deg_engineering": <recommended lon0 for 6° GK (zone*6 convention)>
      }
    """
    import math

    if not (0.0 <= lon_deg <= 180.0):
        raise ValueError(f"Invalid longitude: {lon_deg}")

    # 3° GK: central meridian at multiples of 3 degrees (typical practice).
    lon0_3deg = 3.0 * round(lon_deg / 3.0)

    # 6° GK (EPSG/standard): lon0 in {..., 99, 105, 111, ...}
    # zone = floor(lon/6) + 1  -> lon0 = zone*6 - 3 -> lon0 = 6*floor(lon/6) + 3
    lon0_6deg_epsg = 6.0 * math.floor(lon_deg / 6.0) + 3.0

    # 6° GK (engineering "zone*6"): lon0 in {..., 96, 102, 108, ...}
    # This is simply snapping to nearest multiple of 6; keep for reference/diagnostics.
    lon0_6deg_engineering = 6.0 * round(lon_deg / 6.0)

    return {
        "3deg": lon0_3deg,
        "6deg_epsg": lon0_6deg_epsg,
        "6deg_engineering": lon0_6deg_engineering,
    }

# ============================================================
# GK 带号拼接坐标（中国工程常见坑）自动识别与处理
# ============================================================

CoordType = Literal["GK_WITH_ZONE_6DEG", "WEB_MERCATOR_3857"]


def _looks_like_gk_zone_concat(val: float) -> bool:
    """
    Heuristic: GK 带号拼接通常是 zone*1e6 + easting，
    在中国常见为 1.1e7 ~ 2.4e7 量级（zone 11~24）。
    """
    a = abs(val)
    return 1.0e7 <= a <= 2.6e7


def split_gk_zone_concat(
    X: float,
    Y: float,
    *,
    zone_on: Literal["X", "Y"] = "Y",
) -> Tuple[int, float, float]:
    """
    Split GK coordinate where zone is concatenated into one axis:
      concat = zone*1e6 + easting

    Returns:
      zone (int), easting (m), northing (m)

    Common in Chinese engineering tables:
      X ~ northing (3e6)
      Y ~ zone*1e6 + easting (1.8e7)
    """
    concat = Y if zone_on == "Y" else X
    other = X if zone_on == "Y" else Y

    zone = int(abs(concat) // 1_000_000)
    easting = float(concat - zone * 1_000_000)  # keep sign convention of input
    northing = float(other)

    return zone, easting, northing


def zone_to_lon0_6deg(zone: int) -> float:
    """
    For 6° GK, central meridian is commonly lon0 = zone * 6° - 3 (engineering convention).
    Example: zone=18 -> lon0=108°E
    """
    return float(zone * 6 - 3)


def _gk_inverse_from_zone(
    zone: int,
    easting: float,
    northing: float,
    *,
    validate_china: bool = True,
) -> Tuple[float, float, GKConfig]:
    """
    Interpret input as 6° GK with zone concatenation, then inverse-project to lon/lat.
    """
    lon0 = zone_to_lon0_6deg(zone)
    gk_tmp = GKConfig(lon_0=lon0, x_0=500000.0, y_0=0.0, k=1.0)
    lon, lat = gk_to_lonlat(easting, northing, gk_tmp, validate_china=validate_china, ctx="GK(zone)->lon/lat")
    return lon, lat, gk_tmp


def detect_coord_type(
    X: float,
    Y: float,
    *,
    validate_china: bool = True,
) -> CoordType:
    """
    Detect whether (X,Y) is likely:
      - GK_WITH_ZONE_6DEG (zone concatenated into one axis, usually Y)
      - WEB_MERCATOR_3857 (possibly swapped, but detection here is conservative)

    This function is conservative: if it cannot confidently detect GK-with-zone,
    it returns WEB_MERCATOR_3857 and you can still run strict conversion which may raise error.
    """
    # Strong signal for GK-with-zone: one axis is ~1e7-2e7 and the other is ~1e6-5e6
    if (_looks_like_gk_zone_concat(Y) and (1.0e6 <= abs(X) <= 6.0e6)) or (
        _looks_like_gk_zone_concat(X) and (1.0e6 <= abs(Y) <= 6.0e6)
    ):
        # Try to inverse-project and validate in China bbox
        for zone_on in ("Y", "X"):
            try:
                zone, e, n = split_gk_zone_concat(X, Y, zone_on=zone_on)  # may be wrong if zone_on wrong
                if not (1 <= zone <= 60):
                    continue
                # easting in GK should be roughly within [-1e6, 1e6], typical around 500k
                if not (-2.0e6 <= e <= 2.0e6):
                    continue
                _gk_inverse_from_zone(zone, e, n, validate_china=validate_china)
                return "GK_WITH_ZONE_6DEG"
            except Exception:
                continue

    return "WEB_MERCATOR_3857"


# ----------------------------
# Unified Wellhead conversion
# ----------------------------
@dataclass(frozen=True)
class WellheadResult:
    detected_type: CoordType

    # if detected GK-with-zone
    gk_zone: Optional[int] = None
    gk_zone_on: Optional[Literal["X", "Y"]] = None
    gk_e_in: Optional[float] = None
    gk_n_in: Optional[float] = None
    gk_lon0_from_zone: Optional[float] = None

    # if detected 3857 (standardized)
    wm_x: Optional[float] = None
    wm_y: Optional[float] = None
    swapped_xy_3857: Optional[bool] = None

    # lon/lat after inversion (for traceability / auto recommend)
    lon: float = 0.0
    lat: float = 0.0

    # Canonical GK coordinates (output, using provided gk_target)
    gk_e: float = 0.0
    gk_n: float = 0.0

    # recommended lon0 for 3° / 6° based on lon
    recommended_lon0: Optional[Dict[str, float]] = None


def _wellhead_from_3857(
    X: float,
    Y: float,
    *,
    validate_china: bool = True,
) -> Tuple[float, float, float, float, bool]:
    """
    Try interpret as EPSG:3857 with possible X/Y swap. Return:
      lon, lat, wm_x, wm_y, swapped
    """
    cand = []
    lon1, lat1 = _T_3857_TO_4326.transform(X, Y)
    cand.append((False, float(lon1), float(lat1), float(X), float(Y)))
    lon2, lat2 = _T_3857_TO_4326.transform(Y, X)
    cand.append((True, float(lon2), float(lat2), float(Y), float(X)))

    ok = [(sw, lon, lat, x, y) for (sw, lon, lat, x, y) in cand if _in_china_bbox(lon, lat)]
    if validate_china:
        if len(ok) == 0:
            raise ValueError(
                f"[COORD ERROR] EPSG:3857->lon/lat failed for both axis orders.\n"
                f"  input X={X}, Y={Y}\n"
                f"  try1 (x=X,y=Y) => lon={cand[0][1]:.6f}, lat={cand[0][2]:.6f}\n"
                f"  try2 (x=Y,y=X) => lon={cand[1][1]:.6f}, lat={cand[1][2]:.6f}\n"
                "Neither falls in China bbox. Likely wrong CRS / wrong units / or not 3857."
            )
        if len(ok) == 2:
            raise ValueError(
                f"[COORD ERROR] Ambiguous EPSG:3857 axis order: both produce lon/lat in China.\n"
                f"  input X={X}, Y={Y}\n"
                f"  try1 lon/lat=({cand[0][1]:.6f},{cand[0][2]:.6f})\n"
                f"  try2 lon/lat=({cand[1][1]:.6f},{cand[1][2]:.6f})\n"
                "Please confirm axis meaning from vendor."
            )

    swapped, lon, lat, wm_x, wm_y = ok[0] if len(ok) == 1 else (cand[1] if abs(Y) > abs(X) else cand[0])
    if validate_china:
        _assert_in_china(lon, lat, "EPSG:3857 wellhead")
    return lon, lat, wm_x, wm_y, swapped


def _wellhead_from_gk_zone(
    X: float,
    Y: float,
    *,
    validate_china: bool = True,
) -> Tuple[float, float, int, Literal["X", "Y"], float, float, float]:
    """
    Try interpret as GK-with-zone-concat (6° band). Return:
      lon, lat, zone, zone_on, easting, northing, lon0
    """
    # Try zone on Y first (most common in China engineering tables)
    trials = ["Y", "X"]
    errors = []
    for zone_on in trials:
        try:
            zone, e, n = split_gk_zone_concat(X, Y, zone_on=zone_on)  # zone*1e6 + e
            if not (1 <= zone <= 60):
                raise ValueError(f"zone out of range: {zone}")
            if not (-2.0e6 <= e <= 2.0e6):
                raise ValueError(f"easting abnormal: {e}")
            lon, lat, gk_tmp = _gk_inverse_from_zone(zone, e, n, validate_china=validate_china)
            return lon, lat, zone, zone_on, e, n, gk_tmp.lon_0
        except Exception as ex:
            errors.append((zone_on, str(ex)))

    raise ValueError(
        "[COORD ERROR] GK-with-zone detection failed. Tried zone_on=Y and zone_on=X.\n"
        + "\n".join([f"  {z}: {msg}" for z, msg in errors])
    )


def wellhead_to_gk(
    X: float,
    Y: float,
    gk_target: GKConfig,
    *,
    validate_china: bool = True,
    prefer: Optional[CoordType] = None,
) -> WellheadResult:
    """
    Unified entry:
      - auto detect & convert wellhead coordinates into Canonical GK (gk_target).
      - supports:
          (1) GK_WITH_ZONE_6DEG: common Chinese engineering "zone concatenation"
          (2) WEB_MERCATOR_3857: display/map coordinates (only if truly 3857)

    Behavior:
      - tries the preferred type first if provided
      - otherwise auto-detect, then strict-validate
      - always returns lon/lat traceability + recommended lon0 (3°/6°) from lon
    """
    tried = []
    types_to_try = [prefer] if prefer else [detect_coord_type(X, Y, validate_china=validate_china)]
    # Also try the other type as fallback (but still strict), to avoid false negative
    if types_to_try[0] == "GK_WITH_ZONE_6DEG":
        types_to_try.append("WEB_MERCATOR_3857")
    else:
        types_to_try.append("GK_WITH_ZONE_6DEG")

    last_err: Optional[Exception] = None

    for t in types_to_try:
        tried.append(t)
        try:
            if t == "GK_WITH_ZONE_6DEG":
                lon, lat, zone, zone_on, e_in, n_in, lon0 = _wellhead_from_gk_zone(X, Y, validate_china=validate_china)
                gk_e, gk_n = lonlat_to_gk(lon, lat, gk_target)
                return WellheadResult(
                    detected_type="GK_WITH_ZONE_6DEG",
                    gk_zone=zone,
                    gk_zone_on=zone_on,
                    gk_e_in=e_in,
                    gk_n_in=n_in,
                    gk_lon0_from_zone=lon0,
                    lon=lon,
                    lat=lat,
                    gk_e=gk_e,
                    gk_n=gk_n,
                    recommended_lon0=recommend_gk_lon0(lon),
                )

            if t == "WEB_MERCATOR_3857":
                lon, lat, wm_x, wm_y, swapped = _wellhead_from_3857(X, Y, validate_china=validate_china)
                gk_e, gk_n = lonlat_to_gk(lon, lat, gk_target)
                return WellheadResult(
                    detected_type="WEB_MERCATOR_3857",
                    wm_x=wm_x,
                    wm_y=wm_y,
                    swapped_xy_3857=swapped,
                    lon=lon,
                    lat=lat,
                    gk_e=gk_e,
                    gk_n=gk_n,
                    recommended_lon0=recommend_gk_lon0(lon),
                )

            raise ValueError(f"Unknown type: {t}")

        except Exception as ex:
            last_err = ex
            continue

    raise ValueError(
        f"[COORD ERROR] wellhead_to_gk failed. Tried types={tried}. Last error: {last_err}"
    )
