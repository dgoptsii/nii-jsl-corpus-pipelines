"""Turn stored landmarks into per-frame signing-space region counts.

This is where handedness is resolved: landmarks are normalised, yaw undone, and
for a left-hander the horizontal axis negated so their signing space lands
where a right-hander's does. The hands are then written out by ROLE
(``dominant_*`` / ``non_dominant_*``) rather than by side, so left- and
right-handers pool into one distribution. ``handedness`` is kept on every row,
so the original left/right identity remains recoverable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import HAND_ROLES, REGION_KEYS, YAW_ENABLED
from geometry import (
    build_yaw_series,
    classify_hand,
    core_rectangles,
    normalize_points,
    points_inside_frame,
    reference_lines,
    shoulders_ok,
)
from landmarks import ClipLandmarks

BASE_COLUMNS = [
    "frame_index", "shoulders_ok", "yaw_deg", "yaw_used_3d",
    "dominant_present", "non_dominant_present",
    "dominant_source", "non_dominant_source", "face_source",
    "dominant_points_in_frame", "non_dominant_points_in_frame",
]


def hands_by_role(clip: ClipLandmarks, handedness: str) -> Dict[str, np.ndarray]:
    """Map MediaPipe's anatomical hands onto dominant / non-dominant.

    MediaPipe reports the signer's anatomical left and right hands, so for a
    left-handed signer the dominant hand is the LEFT one.
    """
    if str(handedness).lower() == "left":
        return {"dominant": clip.left_hand_xy, "non_dominant": clip.right_hand_xy}
    return {"dominant": clip.right_hand_xy, "non_dominant": clip.left_hand_xy}


def sources_by_role(clip: ClipLandmarks, handedness: str) -> Dict[str, List[str]]:
    if str(handedness).lower() == "left":
        return {"dominant": clip.left_hand_source, "non_dominant": clip.right_hand_source}
    return {"dominant": clip.right_hand_source, "non_dominant": clip.left_hand_source}


def region_count_columns() -> List[str]:
    return [f"{role}_{region}" for role in HAND_ROLES for region in REGION_KEYS]


def compute_region_counts(
    clip: ClipLandmarks,
    side: str,
    handedness: str = "right",
    yaw_enabled: bool = YAW_ENABLED,
) -> pd.DataFrame:
    """One row per frame: region counts for each hand role, plus diagnostics."""
    mirror = str(handedness).lower() == "left"
    yaw = build_yaw_series(clip.pose_xyz, clip.pose_xyv, side, enabled=yaw_enabled)
    yaw_series = yaw["yaw"]

    hands = hands_by_role(clip, handedness)
    sources = sources_by_role(clip, handedness)
    columns = BASE_COLUMNS + region_count_columns()
    rows: List[Dict[str, object]] = []

    for index in range(len(clip)):
        pose = clip.pose_xyv[index]
        row: Dict[str, object] = {column: 0 for column in columns}
        row["frame_index"] = index
        row["yaw_deg"] = round(float(np.degrees(yaw_series[index])), 3)
        row["yaw_used_3d"] = bool(yaw["used_3d"])
        row["face_source"] = clip.face_source[index] if index < len(clip.face_source) else ""

        for role in HAND_ROLES:
            hand = hands[role][index]
            row[f"{role}_points_in_frame"] = points_inside_frame(hand)
            row[f"{role}_source"] = (
                sources[role][index] if index < len(sources[role]) else "missing"
            )
            row[f"{role}_present"] = False

        ok = shoulders_ok(pose)
        row["shoulders_ok"] = bool(ok)
        if not ok:
            rows.append(row)
            continue

        angle = float(yaw_series[index])
        pose_norm = normalize_points(pose[:, :2], pose, angle, mirror)
        face_norm = normalize_points(clip.face_xy[index], pose, angle, mirror)
        lines = reference_lines(pose_norm, face_norm)
        rects = core_rectangles(lines)

        for role in HAND_ROLES:
            hand_norm = normalize_points(hands[role][index], pose, angle, mirror)
            if hand_norm is None or not np.all(np.isfinite(hand_norm)):
                continue
            counts = classify_hand(hand_norm, rects, lines)
            for region, value in counts.items():
                row[f"{role}_{region}"] = int(value)
            row[f"{role}_present"] = bool(
                sum(v for r, v in counts.items() if r != "missing") > 0
            )

        rows.append(row)

    frame = pd.DataFrame(rows, columns=columns)
    frame.attrs["yaw_used_3d"] = bool(yaw["used_3d"])
    frame.attrs["yaw_reason"] = str(yaw["reason"])
    return frame
