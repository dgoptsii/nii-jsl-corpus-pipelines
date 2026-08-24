"""Shared configuration: column schemas, tier naming and matching tolerances.

Everything that a future maintainer is likely to want to tweak lives here, so
that the rule engine in :mod:`elan_pipeline.parsing` stays a pure description of
the annotation conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

#: Columns carried through from the extraction step so that parsed rows can be
#: joined back onto the original ELAN annotations.
#:
#: ``speaker_id`` is the signer (``FO_03``) and ``tier_id`` is the Word tier
#: the row came from. They are separate because a file can hold more than one
#: Word tier for the same signer: the identity has to survive that, and the
#: rebuild has to know which tier to hang the parsed children off.
METADATA_COLUMNS: List[str] = ["speaker_id", "tier_id", "time_start", "time_end"]

#: Columns produced by the rule parser, in the order they are written to CSV.
PARSED_COLUMNS: List[str] = [
    "annotation", "lexical_item", "pt", "dw", "fs", "aw", "lh", "rh", "d", "cl", "m",
    "ges", "nmm", "rep", "stop", "hold", "index", "keep", "fal", "un", "qm",
    "past", "neg", "compound", "ambiguous",
]

#: Full parsed-CSV header.
FIELDNAMES: List[str] = METADATA_COLUMNS + PARSED_COLUMNS

#: Parsed columns that hold marker attributes (i.e. everything except the raw
#: annotation, the lexical ``lexical_item`` and the two whole-row flags).
ATTR_COLUMNS: List[str] = [
    column for column in PARSED_COLUMNS
    if column not in {"annotation", "lexical_item", "compound", "ambiguous"}
]

#: The two hand columns, used when a marker has to be attributed to one hand.
HAND_COLUMNS: Set[str] = {"lh", "rh"}

#: Header of the intermediate "extracted word annotations" CSV.
EXTRACTED_FIELDNAMES: List[str] = ["file_id", "tier_id", "start_ms", "end_ms", "annotation"]

#: Columns of the parsed CSV that are metadata rather than linguistic output;
#: they never become ELAN tiers.
CSV_METADATA_COLUMNS: Set[str] = {
    "speaker_id", "tier_id", "time_start", "time_end", "annotation",
}

#: Order in which parsed columns become ELAN child tiers.
PREFERRED_PARSED_COLUMN_ORDER: List[str] = [
    column for column in PARSED_COLUMNS if column != "annotation"
]

# ---------------------------------------------------------------------------
# ELAN tier handling
# ---------------------------------------------------------------------------

#: Substrings that identify the Japanese Word tier in a TIER_ID.
WORD_TIER_PATTERNS = ("word-jp",)

#: LINGUISTIC_TYPE used for the generated parsed child tiers.
PARSED_CHILD_LINGUISTIC_TYPE_ID = "parsed_symbolic_association"

#: Template for generated tier IDs: ``<SPEAKER>-PARSED-<COLUMN>``.
PARSED_TIER_TEMPLATE = "{speaker}-PARSED-{column}"

#: Separator used when several parsed values land on the same parent annotation.
PARSED_VALUE_SEPARATOR = " | "

#: Exact time matching is tried first; this tolerance (ms) only rescues small
#: timing shifts between the extracted CSV and the source EAF.
TIME_MATCH_TOLERANCE_MS = 120

#: Minimum temporal overlap ratio accepted when boundaries do not match exactly.
MIN_OVERLAP_RATIO = 0.50

# ---------------------------------------------------------------------------
# Default folder layout
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_FOLDER = Path("pipeline_output")
DEFAULT_EAF_SUBFOLDER = "parsed_elan_files"
DEFAULT_DEBUG_SUBFOLDER = "debug"
EXTRACTED_DEBUG_SUBFOLDER = "extracted_word_annotations"
PARSED_DEBUG_SUBFOLDER = "parsed_annotations"
AMBIGUOUS_DEBUG_SUBFOLDER = "ambiguous_annotations"
TIER_REPORT_SUBFOLDER = "tier_reports"

#: Encodings tried, in order, when reading a CSV written by a colleague.
CSV_READ_ENCODINGS = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]

#: Encoding used for every CSV this pipeline writes (BOM keeps Excel happy with
#: Japanese text).
CSV_WRITE_ENCODING = "utf-8-sig"


@dataclass
class PipelineConfig:
    """Runtime configuration for a single pipeline run."""

    elan_folder: Path
    output_folder: Path = DEFAULT_OUTPUT_FOLDER
    recursive: bool = True
    file_list: Optional[Path] = None
    exceptions_file: Optional[Path] = None
    regions: List[str] = field(default_factory=list)
    save_debug: bool = False
    overwrite: bool = True

    @property
    def eaf_output_folder(self) -> Path:
        return self.output_folder / DEFAULT_EAF_SUBFOLDER

    @property
    def debug_folder(self) -> Path:
        return self.output_folder / DEFAULT_DEBUG_SUBFOLDER

    @property
    def extracted_debug_folder(self) -> Path:
        return self.debug_folder / EXTRACTED_DEBUG_SUBFOLDER

    @property
    def parsed_debug_folder(self) -> Path:
        return self.debug_folder / PARSED_DEBUG_SUBFOLDER

    @property
    def ambiguous_debug_folder(self) -> Path:
        return self.debug_folder / AMBIGUOUS_DEBUG_SUBFOLDER

    @property
    def tier_report_folder(self) -> Path:
        return self.debug_folder / TIER_REPORT_SUBFOLDER
