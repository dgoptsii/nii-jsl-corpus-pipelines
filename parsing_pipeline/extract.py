"""Extract Word-tier annotations from ELAN (.eaf) files.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config import EXTRACTED_FIELDNAMES, WORD_TIER_PATTERNS


@dataclass(frozen=True)
class WordAnnotation:
    """A single Word-tier annotation with its timing.

    ``file_id`` identifies the *person*: ``FO_03``. ``tier_id`` records the tier
    the annotation was read from, which is what the ELAN rebuild needs and what
    makes a duplicated tier visible. See :func:`signer_stem`.
    """

    file_id: str
    tier_id: str
    start_ms: str
    end_ms: str
    annotation: str

    def as_row(self) -> Dict[str, str]:
        return asdict(self)


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def local_name(tag: str) -> str:
    """Remove the XML namespace from an ELAN tag."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def get_attr(element: ET.Element, name: str) -> str:
    return element.attrib.get(name, "")


def build_time_slot_map(root: ET.Element) -> Dict[str, str]:
    times: Dict[str, str] = {}

    for element in root.iter():
        if local_name(element.tag) != "TIME_SLOT":
            continue

        slot_id = get_attr(element, "TIME_SLOT_ID")
        time_value = get_attr(element, "TIME_VALUE")

        if slot_id:
            times[slot_id] = time_value

    return times


def normalise_tier_id(tier_id: str) -> str:
    """Lower case, with the full-width dashes the corpus uses folded to ``-``."""
    return str(tier_id or "").lower().replace("−", "-").replace("–", "-").strip()


def is_word_tier(tier_id: str, patterns: Sequence[str] = WORD_TIER_PATTERNS) -> bool:
    """True only for the Word tier itself: the ID *ends* at ``Word-jp``.

    ``FO_01_KT_70F-Word-jp``        -> True
    ``FO_01_KT_70F-Word-jp-T``      -> False
    ``FO_01_KT_70F-Word-jp-SATO``   -> False
    ``FO_01_KT_70F-Word-jp(roman)`` -> False

    Anything after ``Word-jp`` marks a different tier: a second annotation
    pass (``-T``, ``-S``, ``-SATO``, ``-E?``, ``-cp``) or a romanised
    transcription. They annotate the same signing as the Word tier in
    conventions of their own, so reading them counts the same material twice.
    Matching the name as a substring, which is what this used to do, is how
    they were being read as gloss.
    """
    normalized = normalise_tier_id(tier_id)
    return any(normalized.endswith(pattern) for pattern in patterns)


def is_word_tier_variant(tier_id: str) -> bool:
    """A tier named after the Word tier but with something appended.

    Not read. Reported, so that ignoring several thousand annotations is a
    line in the run log rather than a silence.
    """
    normalized = normalise_tier_id(tier_id)
    return (any(pattern in normalized for pattern in WORD_TIER_PATTERNS)
            and not is_word_tier(tier_id))


def participant_from_tier_id(tier_id: str) -> str:
    """Extract the participant ID from a Word-jp tier ID.

    FO_07_FK_40F-Word-jp -> FO_07_FK_40F
    FO_07_FK_40F_Word-jp -> FO_07_FK_40F
    """
    return re.split(
        r"[-_]?Word[-_ ]?jp",
        str(tier_id or ""),
        flags=re.IGNORECASE,
    )[0].strip("-_ ")


#: A signer's identity in the corpus: two-letter prefecture, underscore, number.
SIGNER_STEM_PATTERN = re.compile(r"^([A-Za-z]{2})[ _-]?0*(\d+)")


def signer_stem(participant_id: str) -> str:
    """Reduce a tier-derived participant ID to the signer it names.

    ``FO_03_NG_40F`` -> ``FO_03``

    The rest of the tier name records initials, age band and gender, and the
    corpus is not consistent about them: the same person appears as
    ``FO_03_NG_40F`` in one file and ``FO_03_NG_50F`` in another, and
    ``TY_11`` carries three spellings across three files. Counting those
    strings counts typing, not signers, so everything downstream -- the corpus
    statistics, the signers file, the bootstrap that resamples over signers --
    uses this stem.

    A participant ID that does not start with the corpus pattern is returned
    unchanged rather than silently reshaped, so an unexpected name shows up in
    the output instead of merging into a neighbour.
    """
    text = str(participant_id or "").strip()
    match = SIGNER_STEM_PATTERN.match(text)
    if not match:
        return text
    return f"{match.group(1).upper()}_{int(match.group(2)):02d}"


def one_tier_per_signer(
    tiers: Sequence[Tuple[ET.Element, str]],
) -> Tuple[List[Tuple[ET.Element, str]], List[Tuple[str, str, int]]]:
    """Keep one Word tier per signer, and report the ones set aside.

    Only ``-Word-jp`` tiers reach this function, but a file can still hold two
    of them for one person -- ``FO_03-04_AniN`` carries ``FO_03_NG_40F-Word-jp``
    and ``FO_03_NG_50F-Word-jp``, the same 378 annotations under two spellings
    of the same signer. The fuller tier is kept.

    Returns the tiers to read, and ``(signer, tier_id, n_annotations)`` for
    each tier set aside, so the caller can report it rather than lose it
    silently.
    """
    def size(tier: ET.Element) -> int:
        return sum(1 for element in tier.iter()
                   if local_name(element.tag) == "ANNOTATION")

    by_signer: Dict[str, List[Tuple[ET.Element, str]]] = {}
    for tier, participant_id in tiers:
        by_signer.setdefault(signer_stem(participant_id), []).append(
            (tier, participant_id)
        )

    kept: List[Tuple[ET.Element, str]] = []
    dropped: List[Tuple[str, str, int]] = []

    for signer in sorted(by_signer):
        group = [item for item in by_signer[signer] if size(item[0])]
        if not group:
            continue

        # Most annotations wins; the tier ID breaks a tie so the choice does
        # not depend on the order ElementTree happened to yield the tiers in.
        chosen = max(group, key=lambda item: (size(item[0]),
                                              get_attr(item[0], "TIER_ID")))
        kept.append(chosen)

        for item in group:
            if item is not chosen:
                dropped.append(
                    (signer, get_attr(item[0], "TIER_ID"), size(item[0]))
                )

    return kept, dropped


def annotation_text(annotation_element: ET.Element) -> str:
    for child in annotation_element.iter():
        if local_name(child.tag) == "ANNOTATION_VALUE":
            return child.text.strip() if child.text else ""

    return ""


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def list_word_tiers(root: ET.Element) -> List[str]:
    """Return the TIER_IDs of every Word-jp tier in an EAF root element."""
    tiers: List[str] = []

    for tier in root.iter():
        if local_name(tier.tag) != "TIER":
            continue

        tier_id = get_attr(tier, "TIER_ID")
        if tier_id and is_word_tier(tier_id):
            tiers.append(tier_id)

    return tiers


def extract_word_annotations(
    eaf_path: Path,
    speaker_filter: Optional[Sequence[str]] = None,
    discarded: Optional[List[Tuple[str, str, int]]] = None,
) -> List[WordAnnotation]:
    """Extract every Word-jp annotation from one ELAN file.

    Parameters
    ----------
    eaf_path:
        Path to the ``.eaf`` file.
    speaker_filter:
        Optional list of participant IDs. When given, only tiers whose
        participant matches one of them are extracted; if nothing matches, the
        filter is ignored and all Word-jp tiers are extracted, so a naming
        mismatch never silently produces an empty file.
    """
    tree = ET.parse(eaf_path)
    root = tree.getroot()
    time_slots = build_time_slot_map(root)

    candidate_tiers = []
    matched_tiers = []
    variant_tiers: List[Tuple[str, str, int]] = []

    wanted = {str(item).upper() for item in (speaker_filter or []) if str(item).strip()}

    for tier in root.iter():
        if local_name(tier.tag) != "TIER":
            continue

        tier_id = get_attr(tier, "TIER_ID")

        if is_word_tier_variant(tier_id):
            # Named after the Word tier but not the Word tier. Counted, so the
            # run says how much was left unread, then skipped.
            filled = sum(1 for element in tier.iter()
                         if local_name(element.tag) == "ANNOTATION")
            if filled:
                variant_tiers.append((
                    signer_stem(participant_from_tier_id(tier_id)
                                or get_attr(tier, "PARTICIPANT")),
                    tier_id, filled,
                ))
            continue

        if not is_word_tier(tier_id):
            continue

        participant_id = participant_from_tier_id(tier_id) or get_attr(tier, "PARTICIPANT")
        candidate_tiers.append((tier, participant_id))

        if wanted and participant_id.upper() in wanted:
            matched_tiers.append((tier, participant_id))

    selected_tiers, dropped = one_tier_per_signer(
        matched_tiers if matched_tiers else candidate_tiers
    )
    if discarded is not None:
        discarded.extend(sorted(variant_tiers) + dropped)

    rows: List[WordAnnotation] = []

    for tier, participant_id in selected_tiers:
        for annotation_container in tier.iter():
            if local_name(annotation_container.tag) != "ANNOTATION":
                continue

            alignable = None
            for child in annotation_container:
                if local_name(child.tag) == "ALIGNABLE_ANNOTATION":
                    alignable = child
                    break

            if alignable is None:
                continue

            text = annotation_text(alignable)
            if not text:
                continue

            start_ref = get_attr(alignable, "TIME_SLOT_REF1")
            end_ref = get_attr(alignable, "TIME_SLOT_REF2")

            rows.append(
                WordAnnotation(
                    file_id=signer_stem(participant_id),
                    tier_id=get_attr(tier, "TIER_ID"),
                    start_ms=time_slots.get(start_ref, ""),
                    end_ms=time_slots.get(end_ref, ""),
                    annotation=text,
                )
            )

    # Tier before time, deliberately. The parser reads a compound group off
    # *consecutive* rows, so rows from two Word tiers of the same signer must
    # not interleave: sorting by time first would let a compound opened on one
    # annotation pass be closed on the other. Grouping by tier is what the old
    # sort did implicitly, back when the participant string carried the tier
    # variant in it.
    rows.sort(
        key=lambda row: (
            row.file_id,
            row.tier_id,
            int(row.start_ms) if row.start_ms.isdigit() else -1,
        )
    )

    return rows


def extract_rows(
    eaf_path: Path,
    speaker_filter: Optional[Sequence[str]] = None,
    discarded: Optional[List[Tuple[str, str, int]]] = None,
) -> List[Dict[str, str]]:
    """Same as :func:`extract_word_annotations` but returns plain dictionaries."""
    return [annotation.as_row() for annotation
            in extract_word_annotations(eaf_path, speaker_filter, discarded)]


__all__ = [
    "EXTRACTED_FIELDNAMES",
    "WordAnnotation",
    "extract_rows",
    "extract_word_annotations",
    "is_word_tier",
    "list_word_tiers",
    "participant_from_tier_id",
    "signer_stem",
    "is_word_tier_variant",
    "one_tier_per_signer",
]
