"""Optional diagnostic renders: annotated example videos and YOLO mask images.

Nothing here is needed to produce a table or a figure. It answers "is the
pipeline seeing what I think it is seeing?", which no summary statistic can: a
hand tracked onto the wrong signer, a jumped chin anchor and a sheared yaw
estimate all produce plausible numbers.

Both are off by default because they are slow and large:
:func:`render_clip_debug_video` (written by step 4) draws the clip beside the
normalised space with every counted point coloured by its region;
:func:`save_person_mask_debug` (written by step 3) draws YOLO's target and
non-target silhouettes and the masked frame MediaPipe received.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import (
    CANVAS_X_MAX,
    CANVAS_X_MIN,
    CANVAS_Y_MAX,
    CANVAS_Y_MIN,
    CHIN,
    CORE_REGION_KEYS,
    EXTREME_REGION_KEYS,
    FACE_TOP,
    HAND_ROLES,
    LEFT_SHOULDER,
    REGION_LABELS,
    RIGHT_SHOULDER,
)
from geometry import (
    classify_hand,
    classify_point,
    core_rectangles,
    counted_hand_indices,
    extreme_rectangles,
    normalize_points,
    reference_lines,
    shoulders_ok,
)

# Canvas holding the normalised signing space, in pixels.
CANVAS_W = 980
CANVAS_H = 900
CANVAS_MARGIN = 80
#: Reserved at the bottom for the counts panel, so the grid never sits under it.
PANEL_HEIGHT = 150
BACKGROUND = (18, 18, 18)

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Skeleton links, as indices into the STORED landmark arrays.
POSE_PAIRS = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]   # shoulders, arms
HAND_PAIRS = [(0, 2), (0, 3), (0, 4), (0, 5)]           # wrist to each knuckle

#: One colour per region, in BGR. Carried over from the reference implementation
#: so a render from either pipeline is directly comparable.
REGION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "upper_torso": (80, 190, 255),
    "lower_torso": (120, 130, 255),
    "p_upper_left": (255, 170, 70),
    "p_upper_center": (255, 200, 90),
    "p_upper_right": (255, 230, 110),
    "p_left_upper_torso": (255, 110, 180),
    "p_left_lower_torso": (255, 150, 200),
    "p_right_upper_torso": (190, 110, 255),
    "p_right_lower_torso": (210, 150, 255),
    "p_lower_left": (130, 220, 255),
    "p_lower_center": (130, 255, 190),
    "p_lower_right": (170, 255, 130),
    "ep_upper_left": (90, 90, 220),
    "ep_upper_center": (110, 110, 240),
    "ep_upper_right": (130, 130, 255),
    "ep_left_upper": (80, 140, 220),
    "ep_left_upper_torso": (80, 170, 220),
    "ep_left_lower_torso": (80, 200, 220),
    "ep_left_lower": (80, 230, 220),
    "ep_right_upper": (220, 120, 80),
    "ep_right_upper_torso": (220, 150, 80),
    "ep_right_lower_torso": (220, 180, 80),
    "ep_right_lower": (220, 210, 80),
    "ep_lower_left": (120, 80, 160),
    "ep_lower_center": (150, 80, 180),
    "ep_lower_right": (180, 80, 200),
    "missing": (180, 180, 180),
}

ROLE_COLORS = {"dominant": (120, 255, 120), "non_dominant": (255, 120, 120)}


# ===========================================================================
# CANVAS
# ===========================================================================

def make_mapper(width: int = CANVAS_W, height: int = CANVAS_H,
                margin: int = CANVAS_MARGIN):
    """Normalised signing-space coordinates -> canvas pixels.

    The x axis is **not** flipped, and that is deliberate. Normalisation puts the
    signer's left shoulder at +x, and the signer's left shoulder is the one that
    appears on the right of the image - so plotting x straight through already
    gives the facing-the-signer view, matching the video panel drawn beside it.
    Flipping here would misalign the two halves of the frame, which is precisely
    the error this render exists to catch.

    (The matplotlib body maps in ``figures.py`` do mirror, because their
    rectangles are hand-authored in a 0..1 layout rather than measured.)
    """
    span_x = CANVAS_X_MAX - CANVAS_X_MIN
    span_y = CANVAS_Y_MAX - CANVAS_Y_MIN
    scale = min(
        (width - 2 * margin) / span_x,
        (height - margin - PANEL_HEIGHT) / span_y,
    )
    offset_x = (width - span_x * scale) / 2.0     # centred; the panel eats the bottom

    def mapper(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        out = np.empty_like(points)
        out[:, 0] = offset_x + (points[:, 0] - CANVAS_X_MIN) * scale
        out[:, 1] = margin + (points[:, 1] - CANVAS_Y_MIN) * scale
        return out

    return mapper


def _rect_polygon(rect: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _fitted_label(canvas, text: str, cx: int, cy: int, box_width: int, colour) -> None:
    """Centre a label in its cell, shrinking it until it fits.

    Narrow cells - the periphery columns - are the ones whose names are longest,
    so a fixed size spills across the neighbouring region and makes the render
    harder to read than no label at all.
    """
    for size in (0.40, 0.34, 0.29, 0.24):
        (text_w, text_h), _ = cv2.getTextSize(text, FONT, size, 1)
        if text_w <= box_width - 6 or size == 0.24:
            cv2.putText(canvas, text, (cx - text_w // 2, cy + text_h // 2),
                        FONT, size, colour, 1, cv2.LINE_AA)
            return


def _draw_region(canvas, mapper, rect, colour, label,
                 alpha: float = 0.12, thickness: int = 2) -> None:
    polygon = mapper(_rect_polygon(rect)).astype(np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [polygon], colour)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    cv2.polylines(canvas, [polygon], True, colour, thickness, lineType=cv2.LINE_AA)

    cx = int(np.mean(polygon[:, 0]))
    cy = int(np.mean(polygon[:, 1]))
    box_width = int(polygon[:, 0].max() - polygon[:, 0].min())
    _fitted_label(canvas, label, cx, cy, box_width, colour)


def _draw_links(image, points_px: np.ndarray, pairs: Sequence[Tuple[int, int]],
                colour, thickness: int = 2) -> None:
    for a, b in pairs:
        if a >= len(points_px) or b >= len(points_px):
            continue
        first, second = points_px[a], points_px[b]
        if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
            continue
        cv2.line(image, (int(first[0]), int(first[1])), (int(second[0]), int(second[1])),
                 colour, thickness, cv2.LINE_AA)


def _draw_counted_points(canvas, mapper, hand_norm, rects, lines,
                         ) -> None:
    """Draw the points that are actually counted, coloured by their region."""
    if hand_norm is None or len(hand_norm) == 0:
        return

    points_px = mapper(hand_norm)
    for index in counted_hand_indices():
        if index >= len(hand_norm):
            continue
        x, y = points_px[index]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        region = classify_point(hand_norm[index], rects, lines)
        colour = REGION_COLORS.get(region, (200, 200, 200))
        cv2.circle(canvas, (int(round(x)), int(round(y))), 6, colour, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 7, (255, 255, 255), 1, cv2.LINE_AA)


PANEL_WIDTH = CANVAS_W - 32


def _truncate(text: str, width: int, size: float) -> str:
    while text and cv2.getTextSize(text, FONT, size, 1)[0][0] > width:
        text = text[:-2]
    return text


def _draw_count_panel(canvas, counts: Dict[str, Dict[str, int]]) -> None:
    """Non-zero region counts for each hand role, bottom left.

    Drawn on an opaque strip clear of the region grid; text overlapping a region
    label is the fastest way to make a diagnostic unreadable.
    """
    top = CANVAS_H - PANEL_HEIGHT + 34
    cv2.rectangle(canvas, (8, top - 26), (8 + PANEL_WIDTH, CANVAS_H - 8),
                  (8, 8, 8), -1)

    y = top
    cv2.putText(canvas, "counted points per region", (18, y), FONT, 0.46,
                (235, 235, 235), 1, cv2.LINE_AA)
    y += 24

    for role in HAND_ROLES:
        entries = [(region, value) for region, value in counts.get(role, {}).items()
                   if value and region != "missing"]
        entries.sort(key=lambda item: -item[1])
        text = "  ".join(f"{REGION_LABELS.get(r, r)}={v}" for r, v in entries) or "-"
        cv2.putText(canvas, _truncate(f"{role:<14}{text}", PANEL_WIDTH - 20, 0.44),
                    (18, y), FONT, 0.44, ROLE_COLORS.get(role, (220, 220, 220)),
                    1, cv2.LINE_AA)
        y += 24


def draw_region_canvas(
    clip,
    index: int,
    yaw_rad: float,
    mirror: bool,
    mapper=None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    """The normalised signing space for one frame, with the classified points."""
    from region_counts import hands_by_role  # local: avoids a circular import

    mapper = mapper or make_mapper()
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND

    pose = clip.pose_xyv[index]
    if not shoulders_ok(pose):
        cv2.putText(canvas, "NO RELIABLE SHOULDERS", (40, 70), FONT, 1.0,
                    (0, 0, 255), 2, cv2.LINE_AA)
        return canvas, {role: {} for role in HAND_ROLES}

    pose_norm = normalize_points(pose[:, :2], pose, yaw_rad, mirror)
    face_norm = normalize_points(clip.face_xy[index], pose, yaw_rad, mirror)
    lines = reference_lines(pose_norm, face_norm)
    rects = core_rectangles(lines)
    extremes = extreme_rectangles(lines)

    for name in EXTREME_REGION_KEYS:
        _draw_region(canvas, mapper, extremes[name], REGION_COLORS[name],
                     REGION_LABELS.get(name, name), alpha=0.07, thickness=1)
    for name in CORE_REGION_KEYS:
        _draw_region(canvas, mapper, rects[name], REGION_COLORS[name],
                     REGION_LABELS.get(name, name))
    _draw_region(canvas, mapper, rects["periphery_bounds"], (0, 0, 255),
                 "extreme outside", alpha=0.0, thickness=3)

    # Anatomical reference lines, the things most worth eyeballing.
    x1, x2 = lines["periphery_right_x"], lines["periphery_left_x"]
    for label, key, colour in (
        ("face top", "face_top_y", (255, 170, 0)),
        ("chin", "chin_y", (0, 0, 255)),
        ("shoulders", "shoulder_y", (230, 230, 230)),
        ("mid torso", "mid_torso_y", (180, 180, 255)),
        ("estimated torso bottom", "hip_y", (200, 200, 200)),
    ):
        value = lines[key]
        pts = mapper(np.asarray([[x1, value], [x2, value]], dtype=np.float32)).astype(int)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), colour, 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (int(min(pts[:, 0])) + 4, int(pts[0][1]) - 4),
                    FONT, 0.33, colour, 1, cv2.LINE_AA)

    pose_px = mapper(pose_norm)
    _draw_links(canvas, pose_px, POSE_PAIRS, (230, 230, 230), 2)
    for shoulder in (LEFT_SHOULDER, RIGHT_SHOULDER):
        if shoulder < len(pose_px):
            x, y = pose_px[shoulder]
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(canvas, (int(round(x)), int(round(y))), 5,
                           (255, 255, 255), -1, cv2.LINE_AA)

    hands = hands_by_role(clip, "left" if mirror else "right")
    counts: Dict[str, Dict[str, int]] = {}

    for role in HAND_ROLES:
        hand_norm = normalize_points(hands[role][index], pose, yaw_rad, mirror)
        if hand_norm is None or not np.all(np.isfinite(hand_norm)):
            counts[role] = {}
            continue
        _draw_links(canvas, mapper(hand_norm), HAND_PAIRS, ROLE_COLORS[role], 2)
        _draw_counted_points(canvas, mapper, hand_norm, rects, lines)
        counts[role] = classify_hand(hand_norm, rects, lines)

    _draw_count_panel(canvas, counts)
    cv2.putText(canvas, _truncate("region names are in the SIGNER's frame: the "
                                  "signer's right is on your left, as in the video",
                                  PANEL_WIDTH - 20, 0.40),
                (18, CANVAS_H - 20), FONT, 0.40, (170, 170, 170), 1, cv2.LINE_AA)
    return canvas, counts


# ===========================================================================
# FRAME OVERLAY
# ===========================================================================

def draw_frame_overlay(frame: np.ndarray, clip, index: int) -> np.ndarray:
    """The original crop with the stored landmarks drawn on it.

    Hands are labelled by MediaPipe's anatomical side, not by role, because this
    view exists to check the detection - if the sides are swapped here, every
    role assignment downstream is wrong.
    """
    out = frame.copy()
    height, width = out.shape[:2]

    def to_pixels(points: np.ndarray) -> np.ndarray:
        pixels = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
        pixels[:, 0] *= width
        pixels[:, 1] *= height
        return pixels

    pose = clip.pose_xyv[index]
    if pose is not None:
        _draw_links(out, to_pixels(pose[:, :2]), POSE_PAIRS, (0, 255, 255), 2)

    face = to_pixels(clip.face_xy[index])
    for point_index, colour in ((FACE_TOP, (255, 170, 0)), (CHIN, (0, 0, 255))):
        if point_index < len(face) and np.all(np.isfinite(face[point_index])):
            x, y = face[point_index]
            cv2.circle(out, (int(round(x)), int(round(y))), 5, colour, -1, cv2.LINE_AA)

    for hand_array, source, colour, label in (
        (clip.left_hand_xy, clip.left_hand_source, (255, 0, 0), "MP LEFT"),
        (clip.right_hand_xy, clip.right_hand_source, (0, 255, 0), "MP RIGHT"),
    ):
        hand = to_pixels(hand_array[index])
        if not np.all(np.isfinite(hand)):
            continue
        _draw_links(out, hand, HAND_PAIRS, colour, 2)
        for x, y in hand:
            cv2.circle(out, (int(round(x)), int(round(y))), 4, colour, -1, cv2.LINE_AA)
        note = source[index] if index < len(source) else ""
        cv2.putText(out, f"{label} ({note})",
                    (int(hand[0][0]) - 40, max(16, int(hand[0][1]) - 14)),
                    FONT, 0.45, colour, 1, cv2.LINE_AA)

    cv2.putText(out, "hand labels are MediaPipe anatomical left/right",
                (12, height - 12), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ===========================================================================
# EXAMPLE VIDEO
# ===========================================================================

def render_clip_debug_video(
    video_path: Path,
    clip,
    out_path: Path,
    side: str,
    handedness: str,
    yaw: Sequence[float],
    max_frames: int = 0,
) -> Optional[Path]:
    """Write ``clip beside signing space`` for one clip. Returns None on failure.

    Never raises: a diagnostic that can abort a long run is worse than no
    diagnostic. Any failure is reported by returning ``None``.
    """
    video_path, out_path = Path(video_path), Path(out_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    ok, first = capture.read()
    if not ok:
        capture.release()
        return None

    original_height, original_width = first.shape[:2]
    scale = CANVAS_H / float(original_height)
    resized_width = max(2, int(round(original_width * scale)))
    fps = capture.get(cv2.CAP_PROP_FPS) or clip.fps or 30.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (resized_width + CANVAS_W, CANVAS_H))
    if not writer.isOpened():
        capture.release()
        return None

    mirror = str(handedness).lower() == "left"
    mapper = make_mapper()
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    limit = len(clip) if max_frames <= 0 else min(len(clip), max_frames)
    written = 0

    try:
        for index in range(limit):
            ok, frame = capture.read()
            if not ok:
                break

            angle = float(yaw[index]) if index < len(yaw) else 0.0
            left_panel = cv2.resize(draw_frame_overlay(frame, clip, index),
                                    (resized_width, CANVAS_H))
            canvas, _counts = draw_region_canvas(clip, index, angle, mirror,
                                                 mapper)
            combined = np.concatenate([left_panel, canvas], axis=1)

            cv2.putText(combined, "original crop + landmarks", (20, 30), FONT,
                        0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined,
                        f"frame {index}  side={side}  handedness={handedness}"
                        f"{'  MIRRORED' if mirror else ''}  "
                        f"yaw={math.degrees(angle):+.1f}deg",
                        (20, 60), FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(combined, "normalised signing space", (resized_width + 20, 30),
                        FONT, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(combined)
            written += 1
    except Exception:  # noqa: BLE001 - a diagnostic must never break a run
        written = written
    finally:
        writer.release()
        capture.release()

    return out_path if written else None


# ===========================================================================
# YOLO PERSON MASKS
# ===========================================================================

def draw_person_masks(frame: np.ndarray, persons: List, target=None) -> np.ndarray:
    """Green for the signer being tracked, red for everyone painted out."""
    out = frame.copy()
    for person in persons:
        colour = (0, 255, 0) if person is target else (0, 0, 255)
        overlay = out.copy()
        overlay[person.mask] = colour
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        x1, y1, x2, y2 = person.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(out, f"{'target' if person is target else 'remove'} "
                         f"cx={person.center_x:.2f}",
                    (x1, max(20, y1 - 8)), FONT, 0.55, colour, 2, cv2.LINE_AA)
    return out


def save_person_mask_debug(
    frame: np.ndarray,
    persons: List,
    target,
    masked_frame: np.ndarray,
    out_dir: Path,
) -> List[Path]:
    """Write the two first-frame images that explain what YOLO removed.

    ``yolo_person_masks.jpg`` shows which silhouette was kept;
    ``mediapipe_input.jpg`` is the frame MediaPipe actually received. If a hand
    is being tracked onto the wrong signer, the second image shows it directly.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for name, image in (("yolo_person_masks.jpg", draw_person_masks(frame, persons, target)),
                        ("mediapipe_input.jpg", masked_frame)):
        path = out_dir / name
        if cv2.imwrite(str(path), image):
            written.append(path)
    return written
