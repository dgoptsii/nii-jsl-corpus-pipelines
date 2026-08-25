"""Shared configuration: landmark sets, signing-space geometry, output layout.

Everything a maintainer is likely to tweak lives here, so the other modules stay
a description of the method rather than a pile of constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

# ===========================================================================
# LANDMARKS THAT ARE ACTUALLY STORED
# ===========================================================================
# MediaPipe reports ~1200 floats per frame; only the ~50 the analysis uses are
# stored, a 24x reduction in file size and load time.

#: Pose landmarks kept, in MediaPipe's own indexing.
POSE_LANDMARK_IDS: List[int] = [
    11,  # left shoulder   - normalisation anchor, yaw estimation
    12,  # right shoulder  - normalisation anchor, yaw estimation
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
]

POSE_NAMES: List[str] = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]

#: Index of each kept pose landmark WITHIN the stored array.
POSE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(POSE_NAMES)}

LEFT_SHOULDER = POSE_INDEX["left_shoulder"]
RIGHT_SHOULDER = POSE_INDEX["right_shoulder"]

#: Hand landmarks kept: the wrist plus the knuckle (MCP) of every finger.
HAND_LANDMARK_IDS: List[int] = [
    0,   # wrist
    2,   # thumb knuckle
    5,   # index knuckle
    9,   # middle knuckle
    13,  # ring knuckle
    17,  # pinky knuckle
]

HAND_NAMES: List[str] = [
    "wrist", "thumb_mcp", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp",
]

HAND_INDEX: Dict[str, int] = {name: i for i, name in enumerate(HAND_NAMES)}

#: Every stored hand point is counted into a region, the thumb knuckle
#: included: an abducted thumb often sits in a different region from the other
#: knuckles, so leaving it out discarded real signing space.
REGION_COUNT_HAND_POINTS: List[str] = list(HAND_NAMES)

#: Face landmarks kept, in MediaPipe's own indexing.
FACE_LANDMARK_IDS: List[int] = [
    152,  # chin
    10,   # top of the head
]

FACE_NAMES: List[str] = ["chin", "face_top"]
FACE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FACE_NAMES)}

CHIN = FACE_INDEX["chin"]
FACE_TOP = FACE_INDEX["face_top"]

N_POSE = len(POSE_LANDMARK_IDS)
N_HAND = len(HAND_LANDMARK_IDS)
N_FACE = len(FACE_LANDMARK_IDS)


# ===========================================================================
# MEDIAPIPE / DETECTION SETTINGS
# ===========================================================================

#: Holistic model complexity. 1 is roughly twice as fast as 2 and is enough for
#: shoulders, elbows, wrists and coarse face anchors.
DEFAULT_MODEL_COMPLEXITY = 1

#: The refined 478-point face mesh only improves eyes and lips, neither of which
#: is used here, so it is off by default.
DEFAULT_REFINE_FACE = False

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

#: A shoulder below this visibility makes the frame's normalisation unreliable.
MIN_SHOULDER_VISIBILITY = 0.25

#: How many of the stored hand points must be inside the frame for the hand to
#: count as present.
MIN_HAND_POINTS_FOR_PRESENT = 3

USE_FRAME_PADDING = True
FRAME_PADDING_RATIO = 0.15

USE_IMAGE_ENHANCEMENT = True
CONTRAST_ALPHA = 1.15
BRIGHTNESS_BETA = 8

# Short-gap interpolation of missing landmarks.
MAX_HAND_INTERPOLATION_GAP = 5
MAX_FACE_INTERPOLATION_GAP = 0  # 0 = fill every internal gap
FILL_EDGE_MISSING_FACES = True


# ===========================================================================
# YOLO PERSON SEGMENTATION
# ===========================================================================
# Clips are cropped from two-person recordings. YOLO segments people, the
# non-target signer is painted out, and hands off the target's silhouette are
# rejected.

USE_YOLO_PERSON_MASK = True
YOLO_SEG_MODEL = "yolov8n-seg.pt"
YOLO_PERSON_CLASS_ID = 0
YOLO_CONF = 0.25
YOLO_IOU = 0.50
YOLO_EVERY_N_FRAMES = 5

MASK_NON_TARGET_PERSON = True
NON_TARGET_MASK_DILATE_PX = 9
MASK_ONLY_IF_MULTIPLE_PERSONS = True

USE_BACKGROUND_MODEL = True
BACKGROUND_SAMPLE_FRAMES = 80
BACKGROUND_RESIZE_WIDTH = 360

#: Reject a hand unless this fraction of its points lies on the target signer.
GATE_HANDS_TO_TARGET = True
HAND_GATE_MASK_DILATE_PX = 21
HAND_GATE_MIN_INSIDE_FRACTION = 0.6


# ===========================================================================
# YAW CORRECTION
# ===========================================================================
# Shoulder normalisation removes camera roll but is blind to yaw: a torso
# turned away from the lens compresses the horizontal axis. Yaw is estimated
# from the shoulder depth difference and undone by stretching x.

YAW_ENABLED = True
YAW_GAIN = 1.0
YAW_MAX_DEG = 35.0
YAW_MEDIAN_WINDOW = 7
YAW_MEAN_WINDOW = 15

#: Below this fraction of usable frames, or this peak angle, the depth estimate
#: is not trustworthy and the clip falls back to plain 2D normalisation.
YAW_MIN_VALID_FRACTION = 0.25
YAW_MIN_SIGNAL_DEG = 1.5

#: Direction is fixed by which side of the frame the signer was cropped from,
#: which is deterministic in a fixed-camera rig; depth supplies only magnitude.
YAW_SIGN_LEFT = +1.0
YAW_SIGN_RIGHT = -1.0

#: Guard against 1/cos blowing up near 90 degrees.
MIN_COS_FOR_UNFORESHORTEN = 0.34


# ===========================================================================
# SIGNING-SPACE GEOMETRY
# ===========================================================================
# Coordinates below are in shoulder-normalised units: shoulder midpoint at the
# origin, shoulders on the x-axis one unit apart, y growing downward.

CENTER_SIDE_EXTRA = 0.00
SIDE_PERIPHERY_WIDTH = 0.50

FALLBACK_FACE_TOP_Y = -1.05
TORSO_HEIGHT_IN_SHOULDER_WIDTHS = 1.6
LOWER_PERIPHERY_TORSO_FRACTION = 0.50

#: Bounds of the drawing/classification canvas in normalised units.
CANVAS_X_MIN, CANVAS_X_MAX = -2.25, 2.25
CANVAS_Y_MIN, CANVAS_Y_MAX = -1.70, 2.65

#: Regions inside the periphery bounds, tested in this order.
CORE_REGION_KEYS: List[str] = [
    "upper_torso", "lower_torso",
    "p_upper_left", "p_upper_center", "p_upper_right",
    "p_left_upper_torso", "p_left_lower_torso",
    "p_right_upper_torso", "p_right_lower_torso",
    "p_lower_left", "p_lower_center", "p_lower_right",
]

#: Regions outside the periphery bounds.
EXTREME_REGION_KEYS: List[str] = [
    "ep_upper_left", "ep_upper_center", "ep_upper_right",
    "ep_left_upper", "ep_left_upper_torso", "ep_left_lower_torso", "ep_left_lower",
    "ep_right_upper", "ep_right_upper_torso", "ep_right_lower_torso", "ep_right_lower",
    "ep_lower_left", "ep_lower_center", "ep_lower_right",
]

#: Every region a point can be assigned to.
REGION_KEYS: List[str] = CORE_REGION_KEYS + EXTREME_REGION_KEYS + ["missing"]

CENTRAL_REGIONS = ["upper_torso", "lower_torso"]
PERIPHERY_REGIONS = [k for k in REGION_KEYS if k.startswith("p_")]
EXTREME_REGIONS = [k for k in REGION_KEYS if k.startswith("ep_")]

REGION_GROUPS: Dict[str, List[str]] = {
    "central torso": CENTRAL_REGIONS,
    "upper/head periphery": ["p_upper_left", "p_upper_center", "p_upper_right"],
    "side periphery": ["p_left_upper_torso", "p_left_lower_torso",
                       "p_right_upper_torso", "p_right_lower_torso"],
    "below torso periphery": ["p_lower_left", "p_lower_center", "p_lower_right"],
    "extreme periphery": EXTREME_REGIONS,
}

REGION_LABELS: Dict[str, str] = {
    "upper_torso": "upper torso", "lower_torso": "lower torso",
    "p_upper_left": "P upper left", "p_upper_center": "P upper center",
    "p_upper_right": "P upper right",
    "p_left_upper_torso": "P left upper torso",
    "p_left_lower_torso": "P left lower torso",
    "p_right_upper_torso": "P right upper torso",
    "p_right_lower_torso": "P right lower torso",
    "p_lower_left": "P lower left", "p_lower_center": "P lower center",
    "p_lower_right": "P lower right",
    "ep_upper_left": "EP upper left", "ep_upper_center": "EP upper center",
    "ep_upper_right": "EP upper right",
    "ep_left_upper": "EP left upper", "ep_left_upper_torso": "EP left upper torso",
    "ep_left_lower_torso": "EP left lower torso", "ep_left_lower": "EP left lower",
    "ep_right_upper": "EP right upper", "ep_right_upper_torso": "EP right upper torso",
    "ep_right_lower_torso": "EP right lower torso", "ep_right_lower": "EP right lower",
    "ep_lower_left": "EP lower left", "ep_lower_center": "EP lower center",
    "ep_lower_right": "EP lower right",
    "missing": "missing",
}

# ===========================================================================
# HANDS
# ===========================================================================
# After mirroring, hands are named by role rather than side, so left- and
# right-handers' dominant hands land in the same column and pool correctly.

HAND_ROLES: List[str] = ["dominant", "non_dominant"]

DEFAULT_HANDEDNESS = "right"


# ===========================================================================
# STATISTICS
# ===========================================================================

#: Resampling is done over SIGNERS, not clips: clips from one signer are not
#: independent observations, so a clip-level interval would be far too narrow.
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 20260806

#: Below this many signers a bootstrap interval is itself unstable; tables flag
#: those cells. Five is not arbitrary: a round draws n signers from n with
#: replacement, so only C(2n-1, n) distinct resamples exist, 10 at three signers
#: against 126 at five. With three, the percentiles are read off a ten-point
#: staircase and the endpoints jump rather than move.
MIN_SIGNERS_FOR_STABLE_CI = 5

#: A cell needs at least this many signers for an interval to be computed at
#: all. Below two there is no between-signer variation to resample.
MIN_SIGNERS_FOR_ANY_CI = 2

#: Reported alongside the boolean flag, so a reader can see how much to
#: discount a cell rather than only whether to: ``solid`` is stable,
#: ``indicative`` means read the width and not the endpoints, ``unstable``
#: means treat it as no interval.
CI_QUALITY_TIERS = [(8, "solid"), (MIN_SIGNERS_FOR_STABLE_CI, "usable"),
                    (3, "indicative"), (MIN_SIGNERS_FOR_ANY_CI, "unstable")]

#: Age bands, matching the corpus. The spreadsheet records a decade band per
#: signer rather than an exact age, so the bands are the same decades; anything
#: coarser would invent a precision the source lacks. ``<20`` is only a guard,
#: so an out-of-range age lands there rather than becoming "unknown".
AGE_GROUPS: List[Tuple[str, float, float]] = [
    ("<20", 0, 20),
    ("20", 20, 30),
    ("30", 30, 40),
    ("40", 40, 50),
    ("50", 50, 60),
    ("60", 60, 70),
    ("70", 70, 80),
    ("80", 80, 200),
]

#: A coarse two-way split, reported alongside the decades. Seven decades x
#: seven prefectures x two genders splits 122 signers too thin for the
#: bootstrap to mean anything, so this is the split to use for the cross-table
#: and for a poster.
AGE_BANDS: List[Tuple[str, float, float]] = [
    ("<50", 0, 50),
    ("50+", 50, 200),
]

#: Gender is carried through purely as a label; it changes no geometry.
DEFAULT_GENDER = "unknown"


# ===========================================================================
# FILE LAYOUT
# ===========================================================================

DEFAULT_OUTPUT_FOLDER = Path("signing_space_output")

KEY_ROWS_SUBFOLDER = "key_rows"
CLIPS_SUBFOLDER = "clips"
LANDMARKS_SUBFOLDER = "landmarks"
REGION_COUNTS_SUBFOLDER = "region_counts"
TABLES_SUBFOLDER = "tables"
FIGURES_SUBFOLDER = "figures"

#: Diagnostic renders live here, and only when explicitly asked for.
DEBUG_SUBFOLDER = "debug"
DEBUG_VIDEO_FILE = "signing_space.mp4"

#: How many clips get a diagnostic render when one is requested. Rendering all
#: of them costs more than the analysis itself. 0 = no cap.
DEFAULT_DEBUG_LIMIT = 5

LANDMARKS_FILE = "landmarks.npz"
LANDMARKS_META_FILE = "landmarks_meta.json"
REGION_COUNTS_FILE = "region_counts.csv"
CLIP_INDEX_FILE = "clip_index.csv"

VIDEO_EXTENSIONS = [".mp4", ".mov", ".m4v", ".avi"]

CSV_READ_ENCODINGS = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
CSV_WRITE_ENCODING = "utf-8-sig"

#: Bumped when a change alters the stored landmark values, so later stages can
#: warn about stale .npz files instead of silently mixing versions.
LANDMARK_FORMAT_VERSION = 1
