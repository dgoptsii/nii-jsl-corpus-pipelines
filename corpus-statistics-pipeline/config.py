"""Shared configuration: column names, region codes, thresholds, output layout.

Everything a maintainer is likely to tweak lives here, so the other modules stay
a description of the method rather than a pile of constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

# ===========================================================================
# PARSED-CSV SCHEMA
# ===========================================================================
# These names come from the annotation parser. If the parser's schema changes,
# this is the only place that has to follow.

METADATA_COLUMNS: List[str] = ["speaker_id", "time_start", "time_end"]

LEXICAL_COLUMN = "lexical_item"
ANNOTATION_COLUMN = "annotation"
SIGNER_COLUMN = "speaker_id"

#: Marker columns, in the order the parser writes them.
KEY_COLUMNS: List[str] = [
    "pt", "dw", "fs", "aw", "lh", "rh", "d", "cl", "m", "ges", "nmm",
    "rep", "stop", "hold", "index", "keep", "fal", "un", "qm", "past", "neg",
]

#: Whole-row flags. An annotation carrying either is excluded from the
#: "successfully parsed" set: a compound is deliberately left unparsed, and an
#: ambiguous row is one the parser could not resolve.
COMPOUND_COLUMN = "compound"
AMBIGUOUS_COLUMN = "ambiguous"

#: Markers that say the annotation's content cannot be trusted. They are counted
#: separately from the linguistic markers because they mean something different.
BLOCKING_KEYS: List[str] = ["d", "fal", "un"]

#: Linguistic markers, i.e. everything except the blocking ones.
LINGUISTIC_KEYS: List[str] = [k for k in KEY_COLUMNS if k not in BLOCKING_KEYS]


# ===========================================================================
# REGIONS
# ===========================================================================
#: Filename prefix -> prefecture. The prefix is the part before the first
#: underscore in the parsed CSV's name (FO_01-02_AniN_parsed.csv -> FO).
REGION_NAMES: Dict[str, str] = {
    "GM": "Gunma", "NR": "Nara", "NS": "Nagasaki", "FO": "Fukuoka",
    "IS": "Ishikawa", "TY": "Toyama", "IK": "Ibaraki",
}

GLOBAL_TAG = "GLOBAL"


# ===========================================================================
# MOUTH ACTION
# ===========================================================================
MOUTH_CATEGORIES: List[str] = ["Mouthing", "MouthGesture", "Others"]

#: A tier is a MouthAction tier when any of its descriptors mentions it.
MOUTH_TIER_HINTS: List[str] = ["mouthaction", "mouthact", "ma"]
WORD_TIER_HINTS: List[str] = ["wordjp", "jpword", "wordjapanese"]

#: Two annotations count as overlapping when they share at least this many
#: milliseconds. Zero would make touching intervals overlap.
MIN_OVERLAP_MS = 1.0


# ===========================================================================
# TOP-N COVERAGE TABLE
# ===========================================================================
#: The headline vocabulary table: for each cutoff, how many signers a gloss must
#: have, and how the occurrence counts distribute.
#:
#: (label, top_n or None for all, minimum unique signers)
TOP_N_SPECS: List[Tuple[str, int, int]] = [
    ("Top 100", 100, 5),
    ("Top 200", 200, 5),
    ("Top 500", 500, 5),
    ("Top 900", 900, 3),
    ("All glosses", 0, 2),          # 0 = no cutoff
]

#: Occurrence cap. A handful of very frequent glosses would otherwise dominate
#: any total, and a training set would be just as imbalanced, so the capped
#: total is the more useful number for anyone planning a split.
OCCURRENCE_CAP = 500

#: Signer counts below this make a per-gloss statistic unreliable.
MIN_SIGNERS_DEFAULT = 2


# ===========================================================================
# MACHINE-LEARNING READINESS
# ===========================================================================
#: Vocabulary sizes at which to report how much of the corpus is covered.
COVERAGE_CUTOFFS: List[int] = [50, 100, 200, 500, 900, 1000, 2000]

#: The cutoffs marked on the coverage curve. They match TOP_N_SPECS, so the
#: figure and the coverage table can be read against each other.
COVERAGE_CURVE_MARKERS: List[int] = [100, 200, 500, 900]

#: Minimum examples per gloss for it to be usable as a class.
CLASS_SIZE_THRESHOLDS: List[int] = [1, 20, 50, 100, 200, 500, 900]

#: Signer floors crossed with the example floors above.
CLASS_SIZE_SIGNER_FLOORS: List[int] = [1, 3, 5, 8]

#: A signer-disjoint split needs every held-out gloss to appear in training.
DEFAULT_TEST_FRACTION = 0.2


# ===========================================================================
# FILE LAYOUT
# ===========================================================================
DEFAULT_OUTPUT_FOLDER = Path("corpus_statistics_output")

TABLES_SUBFOLDER = "tables"
FIGURES_SUBFOLDER = "figures"
DIAGNOSTICS_SUBFOLDER = "diagnostics"

ELAN_INDEX_FILE = "elan_index.csv"
ANNOTATIONS_FILE = "annotations.csv"

CSV_READ_ENCODINGS = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
CSV_WRITE_ENCODING = "utf-8-sig"
