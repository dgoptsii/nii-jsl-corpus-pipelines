"""Reading ELAN .eaf files: durations, tiers, and MouthAction annotations.

Only what the statistics need is read. The parser pipeline owns the annotation
content; this module answers three questions about the original documents:
how long is the recording, which tiers exist, and what does the MouthAction
tier say at a given moment.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from config import MOUTH_CATEGORIES, MOUTH_TIER_HINTS, WORD_TIER_HINTS


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def normalise(value: object) -> str:
    """Lower-case, alphanumerics only. Used for every identifier comparison."""
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


@dataclass(frozen=True)
class Annotation:
    annotation_id: str
    tier_id: str
    participant: str
    start_ms: float
    end_ms: float
    value: str


@dataclass
class ElanFile:
    """One .eaf document, reduced to what the statistics need."""

    path: Path
    duration_ms: float = 0.0
    tier_ids: List[str] = field(default_factory=list)
    participants: Set[str] = field(default_factory=set)
    mouth_tier_ids: Set[str] = field(default_factory=set)
    word_tier_ids: Set[str] = field(default_factory=set)
    mouth_annotations: List[Annotation] = field(default_factory=list)
    word_annotations: List[Annotation] = field(default_factory=list)

    @property
    def has_mouth_tiers(self) -> bool:
        return bool(self.mouth_tier_ids)

    def mouth_labels_overlapping(self, start_ms: float, end_ms: float,
                                 speaker_id: str = "",
                                 min_overlap_ms: float = 1.0) -> List[Annotation]:
        """MouthAction annotations overlapping an interval, for one signer.

        Restricting by signer happens *after* the temporal search: a valid
        overlap should not be discarded merely because tier naming differs
        between the Word and MouthAction tiers.
        """
        if end_ms <= start_ms:
            end_ms = start_ms + min_overlap_ms

        overlapping = [
            a for a in self.mouth_annotations
            if min(a.end_ms, end_ms) - max(a.start_ms, start_ms) >= min_overlap_ms
        ]
        if not overlapping or not speaker_id:
            return overlapping

        key = normalise(speaker_id)
        same_signer = [a for a in overlapping if _signer_matches(key, a)]
        return same_signer if same_signer else overlapping


def _signer_matches(signer_key: str, annotation: Annotation) -> bool:
    participant = normalise(annotation.participant)
    tier = normalise(annotation.tier_id)
    if not signer_key:
        return False
    if participant and (signer_key == participant
                        or signer_key in participant or participant in signer_key):
        return True
    return bool(tier) and signer_key in tier


def classify_mouth_value(value: object) -> Optional[str]:
    """Map a MouthAction value onto Mouthing / MouthGesture / Others.

    MouthGesture is tested first: both labels contain the substring "mouth",
    so the more specific one has to win.
    """
    text = str(value or "").strip()
    if not text:
        return None
    compact = normalise(text)
    if "mouthgesture" in compact or compact in {"mg", "gesture", "mouthgestures"}:
        return "MouthGesture"
    if "mouthing" in compact or compact in {"m", "mouth", "mouthings"}:
        return "Mouthing"
    return "Others"


def _mentions(value: str, hints: List[str]) -> bool:
    compact = normalise(value)
    if not compact:
        return False
    if any(hint in compact for hint in hints if len(hint) > 2):
        return True
    # Short hints such as "MA" must match a whole path segment, never a
    # substring, or every tier containing those two letters would qualify.
    segments = [s.lower() for s in re.split(r"[^A-Za-z0-9]+", str(value)) if s]
    return any(segment in hints for segment in segments)


def read_elan(path: Path) -> ElanFile:
    """Parse one .eaf file. Raises ValueError when the XML is unreadable."""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"Could not parse {path}: {error}") from error

    slots: Dict[str, float] = {}
    for element in root.iter():
        if _local(element.tag) == "TIME_SLOT":
            slot = element.attrib.get("TIME_SLOT_ID", "")
            raw = element.attrib.get("TIME_VALUE")
            if slot and raw not in (None, ""):
                try:
                    slots[slot] = float(raw)
                except ValueError:
                    pass

    types_to_cv: Dict[str, str] = {}
    for element in root.iter():
        if _local(element.tag) == "LINGUISTIC_TYPE":
            type_id = element.attrib.get("LINGUISTIC_TYPE_ID", "")
            if type_id:
                types_to_cv[type_id] = element.attrib.get("CONTROLLED_VOCABULARY_REF", "")

    raw: Dict[str, tuple] = {}
    descriptors: Dict[str, List[str]] = {}
    parents: Dict[str, str] = {}
    participants: Dict[str, str] = {}

    for tier in root.iter():
        if _local(tier.tag) != "TIER":
            continue
        tier_id = tier.attrib.get("TIER_ID", "")
        participant = tier.attrib.get("PARTICIPANT", "")
        type_ref = tier.attrib.get("LINGUISTIC_TYPE_REF", "")
        parent = tier.attrib.get("PARENT_REF", "")
        descriptors[tier_id] = [tier_id, type_ref, parent, types_to_cv.get(type_ref, "")]
        parents[tier_id] = parent
        participants[tier_id] = participant

        for node in tier.iter():
            kind = _local(node.tag)
            if kind not in {"ALIGNABLE_ANNOTATION", "REF_ANNOTATION"}:
                continue
            annotation_id = node.attrib.get("ANNOTATION_ID", "")
            if not annotation_id:
                continue
            value = ""
            for child in node:
                if _local(child.tag) == "ANNOTATION_VALUE":
                    value = "" if child.text is None else child.text.strip()
                    break
            if kind == "ALIGNABLE_ANNOTATION":
                start = slots.get(node.attrib.get("TIME_SLOT_REF1", ""))
                end = slots.get(node.attrib.get("TIME_SLOT_REF2", ""))
                ref = None
            else:
                start = end = None
                ref = node.attrib.get("ANNOTATION_REF")
            raw[annotation_id] = (tier_id, participant, start, end, ref, value)

    # A REF_ANNOTATION inherits its times from the annotation it points at,
    # which may itself be a reference: resolve transitively, guarding cycles.
    resolved: Dict[str, Optional[Tuple[float, float]]] = {}

    def times_of(annotation_id: str, seen: Optional[Set[str]] = None):
        if annotation_id in resolved:
            return resolved[annotation_id]
        record = raw.get(annotation_id)
        if record is None:
            resolved[annotation_id] = None
            return None
        seen = seen or set()
        if annotation_id in seen:
            resolved[annotation_id] = None
            return None
        seen.add(annotation_id)
        _tier, _participant, start, end, ref, _value = record
        if start is not None and end is not None:
            result = (start, end)
        elif ref:
            result = times_of(ref, seen)
        else:
            result = None
        seen.discard(annotation_id)
        resolved[annotation_id] = result
        return result

    by_tier: Dict[str, List[Annotation]] = defaultdict(list)
    for annotation_id, (tier_id, participant, _s, _e, _r, value) in raw.items():
        interval = times_of(annotation_id)
        if interval is None:
            continue
        start, end = sorted(interval)
        by_tier[tier_id].append(Annotation(annotation_id, tier_id, participant,
                                           start, end, value))

    mouth_tiers = {t for t, descriptor in descriptors.items()
                   if any(_mentions(d, MOUTH_TIER_HINTS) for d in descriptor)}
    # Children of a MouthAction tier are MouthAction tiers too.
    changed = True
    while changed:
        changed = False
        for tier_id, parent in parents.items():
            if tier_id not in mouth_tiers and parent in mouth_tiers:
                mouth_tiers.add(tier_id)
                changed = True

    word_tiers = {t for t, descriptor in descriptors.items()
                  if any(_mentions(d, WORD_TIER_HINTS) for d in descriptor)}
    if not word_tiers:
        word_tiers = {t for t, descriptor in descriptors.items()
                      if "word" in normalise(descriptor[0])
                      and "mouth" not in normalise(descriptor[0])}

    mouth = sorted((a for t in mouth_tiers for a in by_tier.get(t, []) if a.value),
                   key=lambda a: (a.start_ms, a.end_ms))
    words = sorted((a for t in word_tiers for a in by_tier.get(t, []) if a.value),
                   key=lambda a: (a.start_ms, a.end_ms))

    return ElanFile(
        path=path,
        duration_ms=max(slots.values(), default=0.0),
        tier_ids=sorted(descriptors),
        participants={p for p in participants.values() if p},
        mouth_tier_ids=mouth_tiers,
        word_tier_ids=word_tiers,
        mouth_annotations=mouth,
        word_annotations=words,
    )
