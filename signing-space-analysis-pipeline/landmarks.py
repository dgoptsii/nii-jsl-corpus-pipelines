"""Extract the reduced landmark set from a cropped clip.

Only the points the analysis uses are kept (shoulders, elbows, wrists, the
knuckle of every finger, chin and head top): about 50 floats per frame instead
of MediaPipe's ~1200. Detection cost is reduced separately, by defaulting to
model complexity 1 and the unrefined face mesh.

Each clip is cropped from a two-person recording, so the other signer can be
partly in frame. YOLO segments people, the non-target signer is painted out
with a static background estimate, and any hand not sitting on the target's
silhouette is rejected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    BACKGROUND_RESIZE_WIDTH,
    BACKGROUND_SAMPLE_FRAMES,
    BRIGHTNESS_BETA,
    CONTRAST_ALPHA,
    DEFAULT_MODEL_COMPLEXITY,
    DEFAULT_REFINE_FACE,
    FACE_LANDMARK_IDS,
    FRAME_PADDING_RATIO,
    GATE_HANDS_TO_TARGET,
    HAND_GATE_MASK_DILATE_PX,
    HAND_GATE_MIN_INSIDE_FRACTION,
    HAND_LANDMARK_IDS,
    LANDMARK_FORMAT_VERSION,
    MASK_NON_TARGET_PERSON,
    MASK_ONLY_IF_MULTIPLE_PERSONS,
    MAX_FACE_INTERPOLATION_GAP,
    MAX_HAND_INTERPOLATION_GAP,
    MIN_DETECTION_CONFIDENCE,
    MIN_HAND_POINTS_FOR_PRESENT,
    MIN_TRACKING_CONFIDENCE,
    NON_TARGET_MASK_DILATE_PX,
    N_FACE,
    N_HAND,
    N_POSE,
    POSE_LANDMARK_IDS,
    USE_BACKGROUND_MODEL,
    USE_FRAME_PADDING,
    USE_IMAGE_ENHANCEMENT,
    USE_YOLO_PERSON_MASK,
    YOLO_CONF,
    YOLO_EVERY_N_FRAMES,
    YOLO_IOU,
    YOLO_PERSON_CLASS_ID,
    YOLO_SEG_MODEL,
)
from geometry import points_inside_frame


@dataclass
class ClipLandmarks:
    """Reduced landmarks for one clip, one row per frame."""

    pose_xyv: np.ndarray          # (n, N_POSE, 3) x, y, visibility
    pose_xyz: np.ndarray          # (n, N_POSE, 3) x, y, z  (yaw estimation)
    face_xy: np.ndarray           # (n, N_FACE, 2)
    left_hand_xy: np.ndarray      # (n, N_HAND, 2)
    right_hand_xy: np.ndarray     # (n, N_HAND, 2)
    left_hand_source: List[str] = field(default_factory=list)
    right_hand_source: List[str] = field(default_factory=list)
    face_source: List[str] = field(default_factory=list)
    fps: float = 30.0
    width: int = 0
    height: int = 0
    n_hands_rejected: int = 0

    def __len__(self) -> int:
        return len(self.pose_xyv)


# ===========================================================================
# YOLO PERSON SEGMENTATION
# ===========================================================================

@dataclass
class PersonInstance:
    mask: np.ndarray
    center_x: float
    area: int
    bbox: Tuple[int, int, int, int]


_YOLO_MODEL = None


def get_yolo_model():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "ultralytics is required for person masking.\n"
                "  pip install ultralytics\n"
                "or run with --no-person-mask to skip it."
            ) from exc
        _YOLO_MODEL = YOLO(YOLO_SEG_MODEL)
    return _YOLO_MODEL


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    size = int(pixels) | 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8) * 255, kernel, iterations=1) > 0


def detect_persons(frame: np.ndarray) -> List[PersonInstance]:
    results = get_yolo_model().predict(
        source=frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False
    )
    if not results or results[0].masks is None or results[0].boxes is None:
        return []

    result = results[0]
    height, width = frame.shape[:2]
    persons: List[PersonInstance] = []

    for raw_mask, class_id in zip(
        result.masks.data.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int)
    ):
        if class_id != YOLO_PERSON_CLASS_ID:
            continue
        mask = cv2.resize(raw_mask.astype(np.float32), (width, height),
                          interpolation=cv2.INTER_LINEAR) > 0.5
        area = int(mask.sum())
        if area <= 0:
            continue
        ys, xs = np.where(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        persons.append(PersonInstance(
            mask=mask,
            center_x=float((bbox[0] + bbox[2]) / 2.0) / float(width),
            area=area,
            bbox=bbox,
        ))

    persons.sort(key=lambda person: person.area, reverse=True)
    return persons


def choose_target(persons: List[PersonInstance], side: str) -> Optional[PersonInstance]:
    """The clip was cropped from one side, so the target is the outermost person."""
    if not persons:
        return None
    return (min if str(side).lower() == "left" else max)(persons, key=lambda p: p.center_x)


def build_background_model(video_path: Path) -> Optional[np.ndarray]:
    """Temporal median of sampled frames, used to paint out the other signer."""
    if not USE_BACKGROUND_MODEL:
        return None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if min(n_frames, width, height) <= 0:
        capture.release()
        return None

    small_w = min(BACKGROUND_RESIZE_WIDTH, width)
    small_h = max(1, round(height * small_w / width))
    samples = []

    for index in np.linspace(0, n_frames - 1, min(BACKGROUND_SAMPLE_FRAMES, n_frames)).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            samples.append(cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA))
    capture.release()

    if not samples:
        return None

    median = np.median(np.stack(samples), axis=0).astype(np.uint8)
    return cv2.resize(median, (width, height), interpolation=cv2.INTER_LINEAR)


def hand_is_on_target(
    hand_xy: Optional[np.ndarray],
    target: Optional[PersonInstance],
) -> bool:
    """Reject a hand belonging to the other signer.

    Tested against the target's segmentation mask rather than its bounding box:
    when the target extends an arm its box widens toward the other signer and can
    swallow the wrong hand, but the silhouette does not.
    """
    if not GATE_HANDS_TO_TARGET or hand_xy is None or target is None:
        return True

    mask = dilate_mask(target.mask, HAND_GATE_MASK_DILATE_PX)
    height, width = mask.shape[:2]
    xs = np.clip(np.round(hand_xy[:, 0] * width).astype(int), 0, width - 1)
    ys = np.clip(np.round(hand_xy[:, 1] * height).astype(int), 0, height - 1)

    return float(mask[ys, xs].mean()) >= HAND_GATE_MIN_INSIDE_FRACTION


# ===========================================================================
# FRAME PREPROCESSING
# ===========================================================================

def mask_non_target(
    frame: np.ndarray,
    persons: List[PersonInstance],
    side: str,
    background: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Paint the other signer out of the frame with the background estimate.

    Kept separate from padding and enhancement so the diagnostic images can show
    exactly this frame - the masking is what goes wrong, and it is invisible once
    the frame has been padded and contrast-stretched.
    """
    info: Dict[str, object] = {
        "person_count": len(persons),
        "person_mask_used": False,
        "target_center_x": "",
    }
    if not (USE_YOLO_PERSON_MASK and persons):
        return frame, info

    target = choose_target(persons, side)
    info["target_center_x"] = round(target.center_x, 4) if target else ""

    if MASK_ONLY_IF_MULTIPLE_PERSONS and len(persons) < 2:
        return frame, info

    non_target = np.zeros(frame.shape[:2], dtype=bool)
    if target is not None and MASK_NON_TARGET_PERSON:
        for person in persons:
            if person is not target:
                non_target |= person.mask
        non_target = dilate_mask(non_target, NON_TARGET_MASK_DILATE_PX)

    if not non_target.any():
        return frame, info

    cleaned = frame.copy()
    if background is not None and background.shape[:2] == frame.shape[:2]:
        cleaned[non_target] = background[non_target]
    else:
        cleaned[non_target] = 0
    info["person_mask_used"] = True
    return cleaned, info


def pad_and_enhance(frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Pad and contrast-stretch for MediaPipe. Returns the padding applied."""
    working = frame
    pad_x = pad_y = 0

    if USE_FRAME_PADDING:
        pad_y = int(round(working.shape[0] * FRAME_PADDING_RATIO))
        pad_x = int(round(working.shape[1] * FRAME_PADDING_RATIO))
        working = cv2.copyMakeBorder(working, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)

    if USE_IMAGE_ENHANCEMENT:
        working = cv2.convertScaleAbs(working, alpha=CONTRAST_ALPHA, beta=BRIGHTNESS_BETA)

    return working, pad_x, pad_y


def preprocess_frame(
    frame: np.ndarray,
    persons: List[PersonInstance],
    side: str,
    background: Optional[np.ndarray],
) -> Tuple[np.ndarray, int, int, Dict[str, object]]:
    """Mask, pad and enhance in one call. Returns the padding used."""
    cleaned, info = mask_non_target(frame, persons, side, background)
    prepared, pad_x, pad_y = pad_and_enhance(cleaned)
    return prepared, pad_x, pad_y, info


def _select(landmark_list, ids: List[int], with_z: bool, with_visibility: bool) -> Optional[np.ndarray]:
    """Pull only the wanted MediaPipe landmarks into a compact array."""
    if landmark_list is None:
        return None

    points = landmark_list.landmark
    columns = 2 + int(with_z) + int(with_visibility)
    out = np.full((len(ids), columns), np.nan, dtype=np.float32)

    for row, index in enumerate(ids):
        if index >= len(points):
            continue
        landmark = points[index]
        values = [landmark.x, landmark.y]
        if with_z:
            values.append(landmark.z)
        if with_visibility:
            values.append(getattr(landmark, "visibility", 1.0))
        out[row] = values

    return out


def _unpad(array: Optional[np.ndarray], original: Tuple[int, int],
           padded: Tuple[int, int], pad_x: int, pad_y: int) -> Optional[np.ndarray]:
    """Convert padded-normalised x/y back to original-frame normalised x/y."""
    if array is None:
        return None
    out = array.copy()
    out[:, 0] = (out[:, 0] * padded[0] - pad_x) / float(original[0])
    out[:, 1] = (out[:, 1] * padded[1] - pad_y) / float(original[1])
    return out


# ===========================================================================
# EXTRACTION
# ===========================================================================

def extract_clip_landmarks(
    video_path: Path,
    side: str,
    model_complexity: int = DEFAULT_MODEL_COMPLEXITY,
    refine_face: bool = DEFAULT_REFINE_FACE,
    use_person_mask: bool = USE_YOLO_PERSON_MASK,
    debug_dir: Optional[Path] = None,
) -> ClipLandmarks:
    """Run detection over a clip and return only the stored landmark set.

    ``debug_dir`` writes two first-frame images showing what YOLO kept and what
    MediaPipe was actually given. Off unless a path is passed - it is a
    diagnostic, not part of the analysis.
    """
    import mediapipe as mp

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    background = build_background_model(video_path) if use_person_mask else None

    pose_xyv: List[Optional[np.ndarray]] = []
    pose_xyz: List[Optional[np.ndarray]] = []
    face_xy: List[Optional[np.ndarray]] = []
    left_xy: List[Optional[np.ndarray]] = []
    right_xy: List[Optional[np.ndarray]] = []
    rejected = 0

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        refine_face_landmarks=refine_face,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as holistic:
        persons: List[PersonInstance] = []
        index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if use_person_mask and (index % YOLO_EVERY_N_FRAMES == 0 or not persons):
                persons = detect_persons(frame)

            cleaned, _info = mask_non_target(frame, persons, side, background)
            prepared, pad_x, pad_y = pad_and_enhance(cleaned)
            padded = (prepared.shape[1], prepared.shape[0])
            original = (width, height)

            if debug_dir is not None and index == 0:
                from debug_render import save_person_mask_debug
                save_person_mask_debug(
                    frame, persons,
                    choose_target(persons, side) if persons else None,
                    cleaned, Path(debug_dir),
                )

            result = holistic.process(cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB))

            pose = _select(result.pose_landmarks, POSE_LANDMARK_IDS, True, True)
            face = _select(result.face_landmarks, FACE_LANDMARK_IDS, False, False)
            left = _select(result.left_hand_landmarks, HAND_LANDMARK_IDS, False, False)
            right = _select(result.right_hand_landmarks, HAND_LANDMARK_IDS, False, False)

            if USE_FRAME_PADDING:
                pose = _unpad(pose, original, padded, pad_x, pad_y)
                face = _unpad(face, original, padded, pad_x, pad_y)
                left = _unpad(left, original, padded, pad_x, pad_y)
                right = _unpad(right, original, padded, pad_x, pad_y)

            target = choose_target(persons, side) if use_person_mask else None
            if left is not None and not hand_is_on_target(left, target):
                left, rejected = None, rejected + 1
            if right is not None and not hand_is_on_target(right, target):
                right, rejected = None, rejected + 1

            pose_xyv.append(pose[:, [0, 1, 3]] if pose is not None else None)
            pose_xyz.append(pose[:, [0, 1, 2]] if pose is not None else None)
            face_xy.append(face)
            left_xy.append(left)
            right_xy.append(right)
            index += 1

    capture.release()

    clip = ClipLandmarks(
        pose_xyv=_stack(pose_xyv, N_POSE, 3),
        pose_xyz=_stack(pose_xyz, N_POSE, 3),
        face_xy=_stack(face_xy, N_FACE, 2),
        left_hand_xy=_stack(left_xy, N_HAND, 2),
        right_hand_xy=_stack(right_xy, N_HAND, 2),
        fps=float(fps),
        width=width,
        height=height,
        n_hands_rejected=rejected,
    )
    interpolate_gaps(clip)
    return clip


def _stack(sequence: List[Optional[np.ndarray]], n_points: int, n_dims: int) -> np.ndarray:
    out = np.full((len(sequence), n_points, n_dims), np.nan, dtype=np.float32)
    for i, array in enumerate(sequence):
        if array is not None and array.shape == (n_points, n_dims):
            out[i] = array
    return out


# ===========================================================================
# GAP FILLING
# ===========================================================================

def _observed_hand(frame: np.ndarray) -> bool:
    return points_inside_frame(frame) >= MIN_HAND_POINTS_FOR_PRESENT


def _interpolate_series(array: np.ndarray, observed: np.ndarray,
                        max_gap: int, fill_edges: bool) -> List[str]:
    """Linearly fill short gaps in place; returns a per-frame provenance label."""
    n = len(array)
    source = ["observed" if observed[i] else "missing" for i in range(n)]
    indices = np.flatnonzero(observed)
    if len(indices) == 0:
        return source

    for previous, following in zip(indices, indices[1:]):
        gap = following - previous - 1
        if gap <= 0 or (max_gap > 0 and gap > max_gap):
            continue
        for step, frame_index in enumerate(range(previous + 1, following), start=1):
            weight = step / float(gap + 1)
            array[frame_index] = (1 - weight) * array[previous] + weight * array[following]
            source[frame_index] = "interpolated"

    if fill_edges:
        for frame_index in range(0, indices[0]):
            array[frame_index] = array[indices[0]]
            source[frame_index] = "nearest"
        for frame_index in range(indices[-1] + 1, n):
            array[frame_index] = array[indices[-1]]
            source[frame_index] = "nearest"

    return source


def interpolate_gaps(clip: ClipLandmarks) -> None:
    """Fill short hand gaps and missing face anchors."""
    face_observed = np.array([np.all(np.isfinite(f)) for f in clip.face_xy])
    clip.face_source = _interpolate_series(
        clip.face_xy, face_observed, MAX_FACE_INTERPOLATION_GAP, True
    )

    for attribute, label in (("left_hand_xy", "left_hand_source"),
                             ("right_hand_xy", "right_hand_source")):
        array = getattr(clip, attribute)
        observed = np.array([_observed_hand(frame) for frame in array])
        setattr(clip, label,
                _interpolate_series(array, observed, MAX_HAND_INTERPOLATION_GAP, False))


# ===========================================================================
# SERIALISATION
# ===========================================================================

def save_landmarks(out_dir: Path, clip: ClipLandmarks, meta: Dict[str, object]) -> Tuple[Path, Path]:
    from config import LANDMARKS_FILE, LANDMARKS_META_FILE

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / LANDMARKS_FILE
    meta_path = out_dir / LANDMARKS_META_FILE

    np.savez_compressed(
        npz_path,
        pose_xyv=clip.pose_xyv,
        pose_xyz=clip.pose_xyz,
        face_xy=clip.face_xy,
        left_hand_xy=clip.left_hand_xy,
        right_hand_xy=clip.right_hand_xy,
        left_hand_source=np.asarray(clip.left_hand_source, dtype="<U16"),
        right_hand_source=np.asarray(clip.right_hand_source, dtype="<U16"),
        face_source=np.asarray(clip.face_source, dtype="<U16"),
    )

    payload = {
        "format_version": LANDMARK_FORMAT_VERSION,
        "n_frames": len(clip),
        "fps": clip.fps,
        "width": clip.width,
        "height": clip.height,
        "hands_rejected_offtarget": clip.n_hands_rejected,
        "pose_landmark_ids": POSE_LANDMARK_IDS,
        "hand_landmark_ids": HAND_LANDMARK_IDS,
        "face_landmark_ids": FACE_LANDMARK_IDS,
        **meta,
    }
    meta_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return npz_path, meta_path


def load_landmarks(npz_path: Path) -> Tuple[ClipLandmarks, Dict[str, object]]:
    from config import LANDMARKS_META_FILE

    npz_path = Path(npz_path)
    meta_path = npz_path.with_name(LANDMARKS_META_FILE)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    data = np.load(npz_path, allow_pickle=False)
    clip = ClipLandmarks(
        pose_xyv=data["pose_xyv"],
        pose_xyz=data["pose_xyz"],
        face_xy=data["face_xy"],
        left_hand_xy=data["left_hand_xy"],
        right_hand_xy=data["right_hand_xy"],
        left_hand_source=[str(v) for v in data["left_hand_source"]],
        right_hand_source=[str(v) for v in data["right_hand_source"]],
        face_source=[str(v) for v in data["face_source"]],
        fps=float(meta.get("fps", 30.0)),
        width=int(meta.get("width", 0)),
        height=int(meta.get("height", 0)),
    )
    return clip, meta
