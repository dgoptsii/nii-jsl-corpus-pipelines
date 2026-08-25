"""Signer metadata: identity, handedness, age and gender.

Driven by one optional CSV (``input_lists/signers.csv``: ``signer_id``,
``handedness``, ``age``, ``gender``). Only the ID is required; blank handedness
means right-handed and blank age reports the signer as "unknown". Column names
and IDs are matched loosely: case, punctuation and zero-padding are ignored,
and the prefix ``FO_07`` matches ``FO_07_FK_40F``.

Age, gender and handedness all come from that file, which
``tools/signers_from_xlsx.py`` generates from the recording spreadsheet. The
tier label is never consulted for them: report~1 shows the label contradicts
the spreadsheet for 23 signers, so the spreadsheet is the single source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from config import AGE_BANDS, AGE_GROUPS, DEFAULT_GENDER, DEFAULT_HANDEDNESS
from io_utils import read_csv_safely

#: Accepted column names in the signers CSV.
ID_COLUMNS = {"signer_id", "speaker_id", "participant_id", "signer", "speaker", "id"}
HANDEDNESS_COLUMNS = {"handedness", "hand", "dominant_hand"}
LEFT_FLAG_COLUMNS = {"left_handed", "is_left_handed", "left", "lefthanded"}
AGE_COLUMNS = {"age", "age_years", "years"}
GENDER_COLUMNS = {"gender", "sex", "性", "性別"}

#: Values in a left-handed flag column that mean "yes".
TRUTHY = {"1", "true", "t", "yes", "y", "x", "left", "l"}

#: Gender spellings folded onto one label each. Anything else is kept verbatim,
#: so a corpus using other categories is reported rather than flattened.
GENDER_LABELS = {
    "m": "M", "male": "M", "man": "M", "男": "M", "男性": "M",
    "f": "F", "female": "F", "woman": "F", "女": "F", "女性": "F",
}


def normalise_gender(value: str) -> str:
    """``M`` / ``F`` / the original spelling / ``unknown`` when blank."""
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "?"}:
        return DEFAULT_GENDER
    return GENDER_LABELS.get(text.lower(), text)


def _pick_column(frame, candidates):
    """First column whose name matches, ignoring case, spaces and underscores."""
    for column in frame.columns:
        if str(column).strip().lower().replace(" ", "_") in candidates:
            return column
    return None


#: Columns in the parsed CSVs that may hold the participant ID, best first.
SIGNER_ID_COLUMNS = [
    "speaker_id", "file_id", "participant_id", "participant",
    "signer_id", "signer", "speaker", "tier_id", "tier", "tier_name",
]


def normalise_signer_key(value: str) -> str:
    """Comparison key for a signer ID: upper case, padding gone, segments kept.

    Leading zeros are stripped from every run of digits, so ``GM_05`` and
    ``GM_5`` compare equal. Whether the corpus zero-pads is not something the
    person maintaining the signers file should have to know.

    Separators are collapsed to a single ``_`` rather than deleted, because the
    segment boundary is what tells ``IS_13`` apart from ``IS_1`` when both are
    being matched against ``IS_13_40M``. See ``_match``.
    """
    key = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_").upper()
    return re.sub(r"(?<![0-9])0+([0-9])", r"\1", key)


def signer_id_from_row(row) -> str:
    """Pull the participant ID out of a parsed annotation row."""
    for column in SIGNER_ID_COLUMNS:
        if column in getattr(row, "index", []) or (
            isinstance(row, dict) and column in row
        ):
            value = str(row[column]).strip()
            if value and value.lower() not in {"nan", "none"}:
                return value
    return ""


def _bucket(age: Optional[float], bands) -> str:
    """Label the band an age falls in. ``"unknown"`` when there is no age."""
    if age is None:
        return "unknown"
    try:
        value = float(age)
    except (TypeError, ValueError):
        return "unknown"

    for label, low, high in bands:
        if low <= value < high:
            return label
    return "unknown"


def age_group_for(age: Optional[float]) -> str:
    """The corpus decade: 20, 30, ... 80."""
    return _bucket(age, AGE_GROUPS)


def age_band_for(age: Optional[float]) -> str:
    """The coarse split: ``<50`` or ``50+``."""
    return _bucket(age, AGE_BANDS)


@dataclass
class SignerMetadata:
    """Handedness and age lookups, both optional."""

    left_handed_keys: List[str] = field(default_factory=list)
    ages_by_key: Dict[str, float] = field(default_factory=dict)
    genders_by_key: Dict[str, str] = field(default_factory=dict)
    signers_file: Optional[Path] = None
    n_rows: int = 0

    # -- lookups ----------------------------------------------------------

    def _match(self, signer_id: str, candidates: Sequence[str]) -> Optional[str]:
        """Exact key match, else the longest listed key that prefixes it.

        A prefix may not end in the middle of a number: without that guard
        ``GM_1`` would claim every clip by ``GM_11`` through ``GM_19``, silently
        and with no error to notice.

        The guard is applied on whole segments first. Matching on the compacted
        key alone gets ``IS_13`` versus ``IS_13_40M`` wrong --- compacted they
        read ``IS13`` and ``IS1340M``, the character after the prefix is a digit,
        and the guard rejects a pair that is in fact the same person. That
        silently un-mirrored two left-handed signers. Segment-wise, ``IS_13`` is
        a prefix of ``IS_13_40M`` and ``IS_1`` is not, which is the intent.
        """
        key = normalise_signer_key(signer_id)
        if not key:
            return None
        if key in candidates:
            return key

        compact = key.replace("_", "")
        same = [c for c in candidates if c and c.replace("_", "") == compact]
        if same:
            return max(same, key=len)

        prefixes = []
        for candidate in candidates:
            if not candidate:
                continue
            # Segment-wise: the prefix has to end where a segment ends.
            if key.startswith(candidate + "_"):
                prefixes.append(candidate)
                continue
            # Compacted, for a signers file written without separators. Here the
            # digit guard is the only boundary available.
            flat = candidate.replace("_", "")
            if compact.startswith(flat) and not (
                flat[-1].isdigit()
                and compact[len(flat):len(flat) + 1].isdigit()
            ):
                prefixes.append(candidate)
        if prefixes:
            return max(prefixes, key=len)
        return None

    def handedness(self, signer_id: str) -> str:
        """``"left"`` or ``"right"``. Unlisted signers are right-handed."""
        if self._match(signer_id, self.left_handed_keys) is not None:
            return "left"
        return DEFAULT_HANDEDNESS


    def age(self, signer_id: str) -> Optional[float]:
        """The signer's age from the spreadsheet, else None."""
        matched = self._match(signer_id, list(self.ages_by_key))
        return self.ages_by_key.get(matched) if matched else None

    def age_group(self, signer_id: str) -> str:
        """The decade band."""
        return age_group_for(self.age(signer_id))

    def age_band(self, signer_id: str) -> str:
        """The coarse ``<50`` / ``50+`` split."""
        return age_band_for(self.age(signer_id))

    def gender(self, signer_id: str) -> str:
        """The signer's gender from the spreadsheet, else unknown."""
        matched = self._match(signer_id, list(self.genders_by_key))
        return self.genders_by_key.get(matched, DEFAULT_GENDER) if matched else DEFAULT_GENDER

    def n_left_handed(self) -> int:
        return len(self.left_handed_keys)

    @property
    def n_with_age(self) -> int:
        return len(self.ages_by_key)

    @property
    def n_with_gender(self) -> int:
        return len(self.genders_by_key)

    def describe(self) -> str:
        source = str(self.signers_file) if self.signers_file else "(none - all right-handed, no ages)"
        return "\n".join([
            f"Signers file:       {source}",
            f"  rows read:        {self.n_rows}",
            f"  left-handed:      {self.n_left_handed}",
            f"  with a known age: {self.n_with_age}",
            f"  with a gender:    {self.n_with_gender}",
        ])


def load_signer_metadata(signers_file: Optional[Path] = None) -> SignerMetadata:
    """Load the one optional signers CSV. No file means empty lookups.

    Anyone absent from the file - or present with a blank handedness - is
    treated as right-handed, so the file only has to list the exceptions.
    """
    if signers_file is None:
        return SignerMetadata()

    signers_file = Path(signers_file)
    if not signers_file.exists():
        raise FileNotFoundError(f"Signers file does not exist: {signers_file}")

    frame = read_csv_safely(signers_file)

    id_column = _pick_column(frame, ID_COLUMNS)
    if id_column is None:
        raise ValueError(
            f"{signers_file} needs an ID column, one of: {sorted(ID_COLUMNS)}.\n"
            f"Found: {list(frame.columns)}"
        )

    handedness_column = _pick_column(frame, HANDEDNESS_COLUMNS)
    left_flag_column = _pick_column(frame, LEFT_FLAG_COLUMNS)
    age_column = _pick_column(frame, AGE_COLUMNS)
    gender_column = _pick_column(frame, GENDER_COLUMNS)

    left_keys: List[str] = []
    ages: Dict[str, float] = {}
    genders: Dict[str, str] = {}
    rows = 0

    for _, row in frame.iterrows():
        raw_id = str(row[id_column]).strip()
        # Tolerate comment rows, which hand-edited files and spreadsheets pick up.
        if not raw_id or raw_id.startswith("#"):
            continue

        key = normalise_signer_key(raw_id)
        if not key:
            continue
        rows += 1

        is_left = False
        if handedness_column is not None:
            is_left = str(row[handedness_column]).strip().lower() in {"left", "l"}
        if not is_left and left_flag_column is not None:
            is_left = str(row[left_flag_column]).strip().lower() in TRUTHY
        if is_left:
            left_keys.append(key)

        if age_column is not None:
            raw_age = str(row[age_column]).strip()
            if raw_age:
                try:
                    ages[key] = float(raw_age)
                except ValueError:
                    pass

        if gender_column is not None:
            gender = normalise_gender(row[gender_column])
            if gender != DEFAULT_GENDER:
                genders[key] = gender

    return SignerMetadata(
        left_handed_keys=left_keys,
        ages_by_key=ages,
        genders_by_key=genders,
        signers_file=signers_file,
        n_rows=rows,
    )
