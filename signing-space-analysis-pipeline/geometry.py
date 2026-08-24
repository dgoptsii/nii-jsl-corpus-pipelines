"""Signing-space geometry: normalisation, yaw correction, region classification.

The coordinate pipeline for one frame:

1. **Shoulder normalisation**: shoulder midpoint to the origin, shoulder line
   rotated onto the x-axis (removing camera roll), scaled by shoulder width, so
   everything downstream is in shoulder widths.
2. **Yaw correction**: a torso turned away from the lens compresses the
   horizontal axis; the turn is estimated from the shoulder depth difference
   and undone by stretching x by 1/cos(yaw).
3. **Handedness mirroring**: for a left-hander, negate x so their dominant-hand
   space lands where a right-hander's does.
4. **Region classification**: anatomical reference lines from the normalised
   pose and face place each stored hand point in one region.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import (
    CANVAS_X_MAX,
    CANVAS_X_MIN,
    CANVAS_Y_MAX,
    CANVAS_Y_MIN,
    CENTER_SIDE_EXTRA,
    CHIN,
    CORE_REGION_KEYS,
    FACE_TOP,
    FALLBACK_FACE_TOP_Y,
    HAND_INDEX,
    LEFT_SHOULDER,
    LOWER_PERIPHERY_TORSO_FRACTION,
    MIN_COS_FOR_UNFORESHORTEN,
    MIN_SHOULDER_VISIBILITY,
    REGION_COUNT_HAND_POINTS,
    RIGHT_SHOULDER,
    SIDE_PERIPHERY_WIDTH,
    TORSO_HEIGHT_IN_SHOULDER_WIDTHS,
    YAW_GAIN,
    YAW_MAX_DEG,
    YAW_MEAN_WINDOW,
    YAW_MEDIAN_WINDOW,
    YAW_MIN_SIGNAL_DEG,
    YAW_MIN_VALID_FRACTION,
    YAW_SIGN_LEFT,
    YAW_SIGN_RIGHT,
)


# ===========================================================================
# NORMALISATION
# ===========================================================================

def shoulders_ok(pose_xyv: Optional[np.ndarray]) -> bool:
    """Both shoulders visible enough for the frame's normalisation to be trusted."""
    if pose_xyv is None or pose_xyv.shape[0] <= max(LEFT_SHOULDER, RIGHT_SHOULDER):
        return False
    return (
        float(pose_xyv[LEFT_SHOULDER, 2]) >= MIN_SHOULDER_VISIBILITY
        and float(pose_xyv[RIGHT_SHOULDER, 2]) >= MIN_SHOULDER_VISIBILITY
    )


def normalize_points(
    points_xy: Optional[np.ndarray],
    pose_xyv: Optional[np.ndarray],
    yaw_rad: float = 0.0,
    mirror_x: bool = False,
) -> Optional[np.ndarray]:
    """Map image coordinates into shoulder-normalised signing space.

    ``yaw_rad`` un-foreshortens the horizontal axis; ``mirror_x`` flips it for a
    left-handed signer. Both are applied after the shoulders have been levelled,
    so the shoulder line stays on the x-axis exactly as the region model assumes.
    """
    if points_xy is None or pose_xyv is None:
        return None
    if pose_xyv.shape[0] <= max(LEFT_SHOULDER, RIGHT_SHOULDER):
        return None

    # The signer's LEFT shoulder appears on the image right, and vice versa.
    image_left = pose_xyv[RIGHT_SHOULDER, :2].astype(np.float32)
    image_right = pose_xyv[LEFT_SHOULDER, :2].astype(np.float32)

    center = (image_left + image_right) / 2.0
    shoulder_vec = image_right - image_left
    scale = float(np.linalg.norm(shoulder_vec)) or 1.0

    angle = math.atan2(float(shoulder_vec[1]), float(shoulder_vec[0]))
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    out = ((points_xy.astype(np.float32) - center) @ rotation.T) / scale

    if yaw_rad and math.isfinite(yaw_rad):
        out[:, 0] = out[:, 0] / max(math.cos(yaw_rad), MIN_COS_FOR_UNFORESHORTEN)

    if mirror_x:
        out[:, 0] = -out[:, 0]

    return out


# ===========================================================================
# YAW ESTIMATION
# ===========================================================================

def side_sign(side: str) -> float:
    """The two signers face opposite side-cameras, so they rotate opposite ways."""
    return YAW_SIGN_LEFT if str(side).lower() == "left" else YAW_SIGN_RIGHT


def raw_yaw_for_frame(
    pose_xyz: Optional[np.ndarray],
    pose_xyv: Optional[np.ndarray],
    side: str,
) -> float:
    """Per-frame yaw in radians from the shoulder depth difference, or NaN.

    Magnitude comes from the depth difference; direction comes from which side of
    the recording the signer was cropped from, which is deterministic in a fixed
    camera rig and far more reliable than the sign of MediaPipe's noisy z.
    """
    if pose_xyz is None or pose_xyv is None or not shoulders_ok(pose_xyv):
        return float("nan")

    left = pose_xyz[LEFT_SHOULDER]
    right = pose_xyz[RIGHT_SHOULDER]
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        return float("nan")

    dx = abs(float(left[0] - right[0]))
    dz = abs(float(left[2] - right[2]))
    magnitude = abs(math.atan2(YAW_GAIN * dz, dx + 1e-6))

    return side_sign(side) * magnitude


def _interpolate_nans(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64).copy()
    good = np.isfinite(out)
    if not good.any():
        return np.zeros_like(out)
    if good.all():
        return out
    index = np.arange(len(out))
    out[~good] = np.interp(index[~good], index[good], out[good])
    return out


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    window += window % 2 == 0
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(values))])


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    window += window % 2 == 0
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def build_yaw_series(
    pose_xyz_all: np.ndarray,
    pose_xyv_all: np.ndarray,
    side: str,
    enabled: bool = True,
) -> Dict[str, object]:
    """Smoothed per-frame yaw for a clip, with an explicit fallback to 2D.

    Returns ``{"yaw": array, "valid": mask, "used_3d": bool, "reason": str}``.
    When the depth signal is unusable the yaw is all zeros, i.e. plain shoulder
    normalisation - the space is never sheared on a turn we could not measure.
    """
    n = len(pose_xyv_all)
    zeros = np.zeros(n, dtype=np.float64)

    if not enabled or n == 0:
        return {"yaw": zeros, "valid": np.zeros(n, bool), "used_3d": False,
                "reason": "yaw correction disabled" if not enabled else "empty clip"}

    raw = np.array(
        [raw_yaw_for_frame(pose_xyz_all[i], pose_xyv_all[i], side) for i in range(n)],
        dtype=np.float64,
    )
    valid = np.isfinite(raw)

    if valid.mean() < YAW_MIN_VALID_FRACTION:
        return {"yaw": zeros, "valid": valid, "used_3d": False,
                "reason": "too few frames with visible shoulders"}

    smooth = _moving_average(
        _median_filter(_interpolate_nans(raw), YAW_MEDIAN_WINDOW), YAW_MEAN_WINDOW
    )
    limit = math.radians(YAW_MAX_DEG)
    smooth = np.clip(smooth, -limit, limit)

    if YAW_MIN_SIGNAL_DEG > 0 and math.degrees(np.max(np.abs(smooth))) < YAW_MIN_SIGNAL_DEG:
        return {"yaw": zeros, "valid": valid, "used_3d": False,
                "reason": "depth signal below the noise floor"}

    return {"yaw": smooth, "valid": valid, "used_3d": True, "reason": ""}


# ===========================================================================
# REFERENCE LINES AND REGIONS
# ===========================================================================

def reference_lines(
    pose_norm: np.ndarray,
    face_norm: Optional[np.ndarray],
) -> Dict[str, float]:
    """Anatomical horizontal and vertical lines, in shoulder-normalised units."""
    left = pose_norm[LEFT_SHOULDER]
    right = pose_norm[RIGHT_SHOULDER]

    shoulder_left_x = float(min(left[0], right[0]))
    shoulder_right_x = float(max(left[0], right[0]))
    shoulder_y = 0.0

    has_face = face_norm is not None and len(face_norm) > max(CHIN, FACE_TOP)

    # If the chin was not detected, the upper band ends at the shoulder line
    # rather than at an invented chin.
    chin_y = shoulder_y
    if has_face and np.all(np.isfinite(face_norm[CHIN])):
        candidate = float(face_norm[CHIN, 1])
        if -1.2 <= candidate <= 0.25:
            chin_y = candidate

    face_top_y = FALLBACK_FACE_TOP_Y
    if has_face and np.all(np.isfinite(face_norm[FACE_TOP])):
        candidate = float(face_norm[FACE_TOP, 1])
        if -1.8 <= candidate <= -0.15:
            face_top_y = candidate
    if face_top_y >= chin_y:
        face_top_y = chin_y - 0.55

    shoulder_width = max(shoulder_right_x - shoulder_left_x, 0.50)
    torso_bottom_y = shoulder_y + TORSO_HEIGHT_IN_SHOULDER_WIDTHS * shoulder_width

    # Normalisation puts the signer's LEFT shoulder at +x, so the maximum-x
    # shoulder is the signer's left and the minimum-x one is the signer's right.
    signer_right_shoulder_x = shoulder_left_x     # minimum x
    signer_left_shoulder_x = shoulder_right_x     # maximum x

    return {
        "center_right_x": signer_right_shoulder_x - CENTER_SIDE_EXTRA,
        "center_left_x": signer_left_shoulder_x + CENTER_SIDE_EXTRA,
        "periphery_right_x": signer_right_shoulder_x - SIDE_PERIPHERY_WIDTH,
        "periphery_left_x": signer_left_shoulder_x + SIDE_PERIPHERY_WIDTH,
        "face_top_y": face_top_y,
        "chin_y": chin_y,
        "shoulder_y": shoulder_y,
        "mid_torso_y": shoulder_y + 0.5 * TORSO_HEIGHT_IN_SHOULDER_WIDTHS * shoulder_width,
        "hip_y": torso_bottom_y,
        "lower_bottom_y": torso_bottom_y + LOWER_PERIPHERY_TORSO_FRACTION * shoulder_width,
    }


def core_rectangles(lines: Dict[str, float]) -> Dict[str, Tuple[float, float, float, float]]:
    """Rectangles inside the periphery bounds, as (x1, y1, x2, y2).

    Region names are in the SIGNER's frame: ``p_right`` is the space beside the
    signer's right shoulder. Because normalisation puts the signer's left
    shoulder at +x, the signer's right side is the NEGATIVE-x half - which is
    what appears on the left of the image. Getting this backwards would silently
    swap every lateral statistic, so the sides are named explicitly here.
    """
    signer_right_x = lines["center_right_x"]        # negative x
    signer_left_x = lines["center_left_x"]          # positive x
    periphery_right_x = lines["periphery_right_x"]  # further negative
    periphery_left_x = lines["periphery_left_x"]    # further positive
    top, chin = lines["face_top_y"], lines["chin_y"]
    mid, hip, bottom = lines["mid_torso_y"], lines["hip_y"], lines["lower_bottom_y"]

    return {
        "p_upper_right": (periphery_right_x, top, signer_right_x, chin),
        "p_upper_center": (signer_right_x, top, signer_left_x, chin),
        "p_upper_left": (signer_left_x, top, periphery_left_x, chin),
        "upper_torso": (signer_right_x, chin, signer_left_x, mid),
        "lower_torso": (signer_right_x, mid, signer_left_x, hip),
        # The side bands are split at the mid-torso line, so a periphery cell
        # covers the same vertical extent as the central cell beside it and as
        # the extreme-periphery cell outside it.
        "p_right_upper_torso": (periphery_right_x, chin, signer_right_x, mid),
        "p_right_lower_torso": (periphery_right_x, mid, signer_right_x, hip),
        "p_left_upper_torso": (signer_left_x, chin, periphery_left_x, mid),
        "p_left_lower_torso": (signer_left_x, mid, periphery_left_x, hip),
        "p_lower_right": (periphery_right_x, hip, signer_right_x, bottom),
        "p_lower_center": (signer_right_x, hip, signer_left_x, bottom),
        "p_lower_left": (signer_left_x, hip, periphery_left_x, bottom),
        "periphery_bounds": (periphery_right_x, top, periphery_left_x, bottom),
    }


def extreme_rectangles(lines: Dict[str, float]) -> Dict[str, Tuple[float, float, float, float]]:
    """Rectangles outside the periphery bounds, in the signer's frame."""
    signer_right_x = lines["center_right_x"]
    signer_left_x = lines["center_left_x"]
    periphery_right_x = lines["periphery_right_x"]
    periphery_left_x = lines["periphery_left_x"]
    top, chin = lines["face_top_y"], lines["chin_y"]
    mid, hip, bottom = lines["mid_torso_y"], lines["hip_y"], lines["lower_bottom_y"]

    return {
        "ep_upper_right": (CANVAS_X_MIN, CANVAS_Y_MIN, signer_right_x, top),
        "ep_upper_center": (signer_right_x, CANVAS_Y_MIN, signer_left_x, top),
        "ep_upper_left": (signer_left_x, CANVAS_Y_MIN, CANVAS_X_MAX, top),
        "ep_right_upper": (CANVAS_X_MIN, top, periphery_right_x, chin),
        "ep_right_upper_torso": (CANVAS_X_MIN, chin, periphery_right_x, mid),
        "ep_right_lower_torso": (CANVAS_X_MIN, mid, periphery_right_x, hip),
        "ep_right_lower": (CANVAS_X_MIN, hip, periphery_right_x, bottom),
        "ep_left_upper": (periphery_left_x, top, CANVAS_X_MAX, chin),
        "ep_left_upper_torso": (periphery_left_x, chin, CANVAS_X_MAX, mid),
        "ep_left_lower_torso": (periphery_left_x, mid, CANVAS_X_MAX, hip),
        "ep_left_lower": (periphery_left_x, hip, CANVAS_X_MAX, bottom),
        "ep_lower_right": (CANVAS_X_MIN, bottom, signer_right_x, CANVAS_Y_MAX),
        "ep_lower_center": (signer_right_x, bottom, signer_left_x, CANVAS_Y_MAX),
        "ep_lower_left": (signer_left_x, bottom, CANVAS_X_MAX, CANVAS_Y_MAX),
    }


def _in_rect(point: np.ndarray, rect: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= float(point[0]) <= x2 and y1 <= float(point[1]) <= y2


def extreme_sector(point: np.ndarray, lines: Dict[str, float]) -> str:
    """Which extreme-periphery sector a point outside the bounds falls into.

    Names are in the signer's frame, so the negative-x half is "right".
    """
    x, y = float(point[0]), float(point[1])
    signer_right_x = lines["center_right_x"]
    signer_left_x = lines["center_left_x"]
    periphery_right_x = lines["periphery_right_x"]
    periphery_left_x = lines["periphery_left_x"]
    top, chin = lines["face_top_y"], lines["chin_y"]
    mid, hip, bottom = lines["mid_torso_y"], lines["hip_y"], lines["lower_bottom_y"]

    if y < top:
        if x < signer_right_x:
            return "ep_upper_right"
        return "ep_upper_left" if x > signer_left_x else "ep_upper_center"
    if y > bottom:
        if x < signer_right_x:
            return "ep_lower_right"
        return "ep_lower_left" if x > signer_left_x else "ep_lower_center"
    if x < periphery_right_x:
        if y < chin:
            return "ep_right_upper"
        if y < mid:
            return "ep_right_upper_torso"
        return "ep_right_lower_torso" if y < hip else "ep_right_lower"
    if x > periphery_left_x:
        if y < chin:
            return "ep_left_upper"
        if y < mid:
            return "ep_left_upper_torso"
        return "ep_left_lower_torso" if y < hip else "ep_left_lower"

    # Inside the bounds but matching no core rectangle: the band above the
    # shoulders between chin and face top.
    return "p_upper_center"


def classify_point(
    point: np.ndarray,
    rects: Dict[str, Tuple[float, float, float, float]],
    lines: Dict[str, float],
) -> str:
    """Assign one normalised point to exactly one region."""
    if not np.all(np.isfinite(point)):
        return "missing"

    for name in CORE_REGION_KEYS:
        if _in_rect(point, rects[name]):
            return name

    return extreme_sector(point, lines)


def counted_hand_indices() -> List[int]:
    """Indices into the stored hand array that are counted into regions."""
    return [HAND_INDEX[name] for name in REGION_COUNT_HAND_POINTS
            if name in HAND_INDEX]


def classify_hand(
    hand_norm: Optional[np.ndarray],
    rects: Dict[str, Tuple[float, float, float, float]],
    lines: Dict[str, float],
) -> Dict[str, int]:
    """Count the stored hand points falling into each region."""
    counts: Dict[str, int] = {}
    if hand_norm is None or len(hand_norm) == 0:
        return counts

    for index in counted_hand_indices():
        if index >= len(hand_norm):
            continue
        region = classify_point(hand_norm[index], rects, lines)
        counts[region] = counts.get(region, 0) + 1

    return counts


def points_inside_frame(hand_xy: Optional[np.ndarray]) -> int:
    """How many stored hand points lie within (a small margin of) the frame."""
    if hand_xy is None:
        return 0
    inside = (
        (hand_xy[:, 0] >= -0.05) & (hand_xy[:, 0] <= 1.05)
        & (hand_xy[:, 1] >= -0.05) & (hand_xy[:, 1] <= 1.05)
    )
    return int(inside.sum())
