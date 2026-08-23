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
# MediaPipe reports 33 pose + 21 per hand + 468/478 face points per frame.
# The signing-space analysis uses a small fraction of those, so only that
# fraction is written to disk: roughly 50 floats per frame instead of ~1200,
# which is about a 24x reduction in file size and load time.

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

#: Which of the stored hand points are counted into signing-space regions:
#: every point that is stored. The thumb knuckle used to be excluded, and
#: optional, for comparability with an earlier version of the pipeline. It is
#: now always counted -- thumb position distinguishes real handshapes, and an
#: abducted thumb often sits in a different region from the other knuckles, so
#: leaving it out discarded a genuine part of the signing space.
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
# Each clip is cropped from a two-person recording, so the other signer can be
# partly visible. YOLO segments people, the non-target person is painted out
# with a static background estimate, and any hand that does not sit on the
# target's silhouette is rejected.

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
# Shoulder normalisation removes camera roll but is blind to yaw: a signer
# filmed from the side has a torso turned away from the lens, which compresses
# the horizontal axis. Yaw is estimated from the shoulder depth difference and
# undone by stretching x.

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
# All coordinates below are in shoulder-normalised units: the shoulder midpoint
# is the origin, the shoulders lie on the x-axis one unit apart, and y grows
# downward.

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

#: Left/right mirror pairs, used when a left-handed signer's space is flipped.
MIRROR_REGION: Dict[str, str] = {
    "p_upper_left": "p_upper_right", "p_upper_right": "p_upper_left",
    "p_left_upper_torso": "p_right_upper_torso",
    "p_right_upper_torso": "p_left_upper_torso",
    "p_left_lower_torso": "p_right_lower_torso",
    "p_right_lower_torso": "p_left_lower_torso",
    "p_lower_left": "p_lower_right", "p_lower_right": "p_lower_left",
    "ep_upper_left": "ep_upper_right", "ep_upper_right": "ep_upper_left",
    "ep_left_upper": "ep_right_upper", "ep_right_upper": "ep_left_upper",
    "ep_left_upper_torso": "ep_right_upper_torso",
    "ep_right_upper_torso": "ep_left_upper_torso",
    "ep_left_lower_torso": "ep_right_lower_torso",
    "ep_right_lower_torso": "ep_left_lower_torso",
    "ep_left_lower": "ep_right_lower", "ep_right_lower": "ep_left_lower",
    "ep_lower_left": "ep_lower_right", "ep_lower_right": "ep_lower_left",
}


# ===========================================================================
# HANDS
# ===========================================================================
# After mirroring, hands are named by role rather than by side: the dominant
# hand of a left-handed signer and of a right-handed signer end up in the same
# column, so their distributions can be pooled.

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
#: those cells rather than hiding the problem.
#:
#: Five is not arbitrary. A bootstrap round draws n signers from n with
#: replacement, so the number of genuinely distinct resamples it can produce is
#: C(2n-1, n): 10 at three signers, 35 at four, 126 at five, 6,435 at eight.
#: With three signers the 2,000 iterations can only land on ten distinct values,
#: so the percentiles are read off a ten-point staircase and the interval
#: endpoints jump rather than move; one round in nine is a single signer cloned
#: three times. The interval is still computed -- it is the honest width given
#: the evidence -- but calling it reliable would not be.
MIN_SIGNERS_FOR_STABLE_CI = 5

#: A cell needs at least this many signers for an interval to be computed at
#: all. Below two there is no between-signer variation to resample.
MIN_SIGNERS_FOR_ANY_CI = 2

#: Reported alongside the boolean flag, so a reader can see how much to
#: discount a cell rather than only whether to.
#:   solid      -- enough signers that the interval is stable
#:   indicative -- computed, but the resample space is small; read the width,
#:                 not the endpoints
#:   unstable   -- barely more than one signer; treat as no interval
CI_QUALITY_TIERS = [(8, "solid"), (MIN_SIGNERS_FOR_STABLE_CI, "usable"),
                    (3, "indicative"), (MIN_SIGNERS_FOR_ANY_CI, "unstable")]

#: Age bands, matching the corpus. The spreadsheet records a decade band per
#: signer (20, 30, ... 80) rather than an exact age, so the bands here are the
#: same decades - regrouping them into anything coarser would be inventing a
#: precision the source does not have.
#:
#: ``<20`` exists only as a guard: no signer in the corpus is under 20, so it
#: never appears, but an out-of-range age lands there rather than silently
#: becoming "unknown". ``80`` is open-ended upward for the same reason.
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

#: A coarse two-way split, reported alongside the decades.
#:
#: Seven decades x seven prefectures x two genders splits 122 signers very thin,
#: and a cell with two people supports no claim. This split keeps enough signers
#: per cell for the bootstrap to mean something, so it is the one to use for the
#: cross-table and for anything going on a poster. The decades stay available in
#: their own table for describing the corpus.
AGE_BANDS: List[Tuple[str, float, float]] = [
    ("<50", 0, 50),
    ("50+", 50, 200),
]

#: Gender is carried through purely as a label - it changes no geometry.
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

#: How many clips get a diagnostic render when one is requested. Rendering every
#: clip costs more time than the analysis itself and produces more video than the
#: corpus, and a handful is enough to see whether tracking is sane. 0 = no cap.
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
