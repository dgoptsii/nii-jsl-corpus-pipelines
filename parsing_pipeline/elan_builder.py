"""Write new ELAN files that carry the parsed annotations as extra tiers.

**No original tier is removed.** The output ``.eaf`` is the input document plus
one new child tier per parsed column and per speaker,
``<SPEAKER>-PARSED-<COLUMN>`` (e.g. ``FO_07_FK_40F-PARSED-cl``).

Each new tier is a ``Symbolic_Association`` child of that speaker's Word-jp
tier, so every parsed value sits under the annotation it came from and inherits
its timing. New elements are inserted in EAF schema order, and the
``Symbolic_Association`` constraint and linguistic type are declared if the
source file did not already declare them.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from config import (
    CSV_METADATA_COLUMNS,
    MIN_OVERLAP_RATIO,
    PARSED_CHILD_LINGUISTIC_TYPE_ID,
    PARSED_TIER_TEMPLATE,
    PARSED_VALUE_SEPARATOR,
    PREFERRED_PARSED_COLUMN_ORDER,
    TIME_MATCH_TOLERANCE_MS,
)
from extract import is_word_tier, local_name, participant_from_tier_id

#: Order of ANNOTATION_DOCUMENT children required by the EAF schema.
EAF_CHILD_ORDER = [
    "LICENSE",
    "HEADER",
    "TIME_ORDER",
    "TIER",
    "LINGUISTIC_TYPE",
    "LOCALE",
    "LANGUAGE",
    "CONSTRAINT",
    "CONTROLLED_VOCABULARY",
    "LEXICON_REF",
    "REF_LINK_SET",
    "EXTERNAL_REF",
]

SYMBOLIC_ASSOCIATION_DESCRIPTION = (
    "Time subdivision of parent annotation's time interval, no time gaps allowed within this interval"
)


@dataclass
class BuildResult:
    """Everything a tier report needs to describe one processed file."""

    file_stem: str
    input_eaf: Path
    output_eaf: Optional[Path] = None
    original_tiers: List[str] = field(default_factory=list)
    created_tiers: List[Tuple[str, int, str]] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    missing_parent_speakers: List[str] = field(default_factory=list)
    unmatched_rows: List[Tuple[str, str, str, str, str]] = field(default_factory=list)
    broken_parent_refs: List[Tuple[str, str]] = field(default_factory=list)
    broken_annotation_refs: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def created_tier_count(self) -> int:
        return len(self.created_tiers)

    @property
    def created_annotation_count(self) -> int:
        return sum(count for _, count, _ in self.created_tiers)


@dataclass(frozen=True)
class ParentAnnotation:
    annotation_id: str
    start_ms: int
    end_ms: int
    value: str


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def clean_cell(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip()

    if text.lower() in {"", "nan", "none", "null"}:
        return ""

    return text


def parse_int_ms(value: object) -> Optional[int]:
    text = clean_cell(value)
    if not text:
        return None

    try:
        return int(round(float(text)))
    except ValueError:
        return None


def safe_tier_id(text: str) -> str:
    text = clean_cell(text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-.ぁ-んァ-ン一-龥ー]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "EMPTY"


def get_all_tier_ids(root: ET.Element) -> List[str]:
    return [
        tier.attrib.get("TIER_ID", "")
        for tier in root.findall("TIER")
        if tier.attrib.get("TIER_ID", "")
    ]


def find_tier(root: ET.Element, tier_id: str) -> Optional[ET.Element]:
    for tier in root.findall("TIER"):
        if tier.attrib.get("TIER_ID") == tier_id:
            return tier
    return None


def existing_annotation_ids(root: ET.Element) -> Set[str]:
    ids: Set[str] = set()

    for tag in ("ALIGNABLE_ANNOTATION", "REF_ANNOTATION"):
        for annotation in root.findall(f".//{tag}"):
            annotation_id = annotation.attrib.get("ANNOTATION_ID")
            if annotation_id:
                ids.add(annotation_id)

    return ids


def next_annotation_id_factory(root: ET.Element, prefix: str = "pa"):
    """Return a callable that hands out unused ANNOTATION_IDs."""
    used = existing_annotation_ids(root)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_number = 0

    for annotation_id in used:
        match = pattern.match(annotation_id)
        if match:
            max_number = max(max_number, int(match.group(1)))

    counter = max_number + 1

    def make_id() -> str:
        nonlocal counter

        while True:
            annotation_id = f"{prefix}{counter}"
            counter += 1
            if annotation_id not in used:
                used.add(annotation_id)
                return annotation_id

    return make_id


def unique_tier_id(root: ET.Element, desired_tier_id: str) -> str:
    existing = set(get_all_tier_ids(root))

    if desired_tier_id not in existing:
        return desired_tier_id

    counter = 2
    while True:
        candidate = f"{desired_tier_id}_{counter}"
        if candidate not in existing:
            return candidate
        counter += 1


def insert_in_schema_order(root: ET.Element, element: ET.Element) -> None:
    """Insert ``element`` at the position the EAF schema expects."""
    tag = local_name(element.tag)

    try:
        target_rank = EAF_CHILD_ORDER.index(tag)
    except ValueError:  # pragma: no cover - unknown element, append at the end
        root.append(element)
        return

    insert_at = len(root)
    for index, child in enumerate(list(root)):
        child_tag = local_name(child.tag)
        child_rank = (
            EAF_CHILD_ORDER.index(child_tag)
            if child_tag in EAF_CHILD_ORDER
            else len(EAF_CHILD_ORDER)
        )
        if child_rank > target_rank:
            insert_at = index
            break

    root.insert(insert_at, element)


def ensure_constraint(root: ET.Element, stereotype: str, description: str = "") -> None:
    for constraint in root.findall("CONSTRAINT"):
        if constraint.attrib.get("STEREOTYPE") == stereotype:
            return

    element = ET.Element(
        "CONSTRAINT",
        {"DESCRIPTION": description or stereotype, "STEREOTYPE": stereotype},
    )
    insert_in_schema_order(root, element)


def ensure_linguistic_type(
    root: ET.Element,
    linguistic_type_id: str,
    time_alignable: bool,
    constraints: Optional[str] = None,
) -> None:
    for linguistic_type in root.findall("LINGUISTIC_TYPE"):
        if linguistic_type.attrib.get("LINGUISTIC_TYPE_ID") == linguistic_type_id:
            return

    attributes = {
        "LINGUISTIC_TYPE_ID": linguistic_type_id,
        "TIME_ALIGNABLE": "true" if time_alignable else "false",
        "GRAPHIC_REFERENCES": "false",
    }

    if constraints:
        attributes["CONSTRAINTS"] = constraints

    insert_in_schema_order(root, ET.Element("LINGUISTIC_TYPE", attributes))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def collect_time_slot_values(root: ET.Element) -> Dict[str, int]:
    result: Dict[str, int] = {}

    for time_slot in root.findall(".//TIME_SLOT"):
        slot_id = time_slot.attrib.get("TIME_SLOT_ID", "")
        value = parse_int_ms(time_slot.attrib.get("TIME_VALUE", ""))

        if slot_id and value is not None:
            result[slot_id] = value

    return result


def build_annotation_time_map(root: ET.Element) -> Dict[str, Tuple[int, int]]:
    """Map every annotation ID to timing, following REF_ANNOTATION chains."""
    time_slots = collect_time_slot_values(root)
    annotation_to_time: Dict[str, Tuple[int, int]] = {}

    for annotation in root.findall(".//ALIGNABLE_ANNOTATION"):
        annotation_id = annotation.attrib.get("ANNOTATION_ID", "")
        slot1 = annotation.attrib.get("TIME_SLOT_REF1", "")
        slot2 = annotation.attrib.get("TIME_SLOT_REF2", "")

        if annotation_id and slot1 in time_slots and slot2 in time_slots:
            annotation_to_time[annotation_id] = (time_slots[slot1], time_slots[slot2])

    changed = True
    while changed:
        changed = False

        for annotation in root.findall(".//REF_ANNOTATION"):
            annotation_id = annotation.attrib.get("ANNOTATION_ID", "")
            parent_id = annotation.attrib.get("ANNOTATION_REF", "")

            if annotation_id and annotation_id not in annotation_to_time and parent_id in annotation_to_time:
                annotation_to_time[annotation_id] = annotation_to_time[parent_id]
                changed = True

    return annotation_to_time


def get_annotation_value(annotation_node: ET.Element) -> str:
    value_node = annotation_node.find("ANNOTATION_VALUE")
    if value_node is None or value_node.text is None:
        return ""
    return value_node.text


def collect_parent_annotations(root: ET.Element, parent_tier_id: str) -> List[ParentAnnotation]:
    tier = find_tier(root, parent_tier_id)
    if tier is None:
        return []

    annotation_to_time = build_annotation_time_map(root)
    result: List[ParentAnnotation] = []

    for annotation in tier.findall(".//ALIGNABLE_ANNOTATION") + tier.findall(".//REF_ANNOTATION"):
        annotation_id = annotation.attrib.get("ANNOTATION_ID", "")
        if not annotation_id or annotation_id not in annotation_to_time:
            continue

        start_ms, end_ms = annotation_to_time[annotation_id]

        result.append(
            ParentAnnotation(
                annotation_id=annotation_id,
                start_ms=start_ms,
                end_ms=end_ms,
                value=get_annotation_value(annotation),
            )
        )

    result.sort(key=lambda item: (item.start_ms, item.end_ms, item.annotation_id))
    return result


def build_exact_parent_time_index(
    parent_annotations: Sequence[ParentAnnotation],
) -> Dict[Tuple[int, int], ParentAnnotation]:
    result: Dict[Tuple[int, int], ParentAnnotation] = {}

    for annotation in parent_annotations:
        result.setdefault((annotation.start_ms, annotation.end_ms), annotation)

    return result


def overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    duration = max(1, a_end - a_start)
    return intersection / duration


def find_matching_parent_annotation(
    parent_annotations: Sequence[ParentAnnotation],
    exact_index: Dict[Tuple[int, int], ParentAnnotation],
    start_ms: int,
    end_ms: int,
) -> Optional[ParentAnnotation]:
    """Match a parsed row onto a parent annotation: exact first, then nearest."""
    exact = exact_index.get((start_ms, end_ms))
    if exact is not None:
        return exact

    best: Optional[ParentAnnotation] = None
    best_key: Tuple[float, int] = (-1.0, 10 ** 12)

    for annotation in parent_annotations:
        overlap = overlap_ratio(start_ms, end_ms, annotation.start_ms, annotation.end_ms)
        boundary_delta = abs(start_ms - annotation.start_ms) + abs(end_ms - annotation.end_ms)

        good_overlap = overlap >= MIN_OVERLAP_RATIO
        close_boundaries = (
            abs(start_ms - annotation.start_ms) <= TIME_MATCH_TOLERANCE_MS
            and abs(end_ms - annotation.end_ms) <= TIME_MATCH_TOLERANCE_MS
        )

        if not (good_overlap or close_boundaries):
            continue

        # Higher overlap wins; on a tie the smaller boundary shift wins.
        key = (overlap, -boundary_delta)
        if key > best_key:
            best_key = key
            best = annotation

    return best


# ---------------------------------------------------------------------------
# Tier construction
# ---------------------------------------------------------------------------

def find_parent_word_tier_id(root: ET.Element, speaker_id: str) -> Optional[str]:
    """Find the Word-jp tier that a speaker's parsed rows belong to.

    The extractor derives ``speaker_id`` from the tier name itself, so the exact
    match is tried first; the looser heuristics only exist for CSVs produced by
    older tooling.
    """
    speaker = clean_cell(speaker_id)
    if not speaker:
        return None

    exact: List[str] = []
    loose: List[str] = []

    for tier in root.findall("TIER"):
        tier_id = tier.attrib.get("TIER_ID", "")
        participant = tier.attrib.get("PARTICIPANT", "")

        if not is_word_tier(tier_id):
            continue

        if participant_from_tier_id(tier_id).casefold() == speaker.casefold():
            exact.append(tier_id)
        elif participant and participant.casefold() == speaker.casefold():
            exact.append(tier_id)
        elif speaker.casefold() in tier_id.casefold():
            loose.append(tier_id)

    candidates = exact or loose
    if not candidates:
        return None

    # Prefer the shortest name: usually the main Word-jp tier rather than a
    # dependent variant.
    return sorted(candidates, key=lambda item: (len(item), item))[0]


def get_parsed_columns(rows: Sequence[Dict[str, str]]) -> List[str]:
    """Return the parsed columns to turn into tiers, in a stable order."""
    if not rows:
        return []

    fieldnames = list(rows[0].keys())

    columns = [
        column for column in PREFERRED_PARSED_COLUMN_ORDER
        if column in fieldnames and column not in CSV_METADATA_COLUMNS
    ]

    # Keep any future parser columns too, after the known order.
    for column in fieldnames:
        if column not in CSV_METADATA_COLUMNS and column not in columns:
            columns.append(column)

    return columns


def append_unique(values: List[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def add_parsed_child_tiers(
    root: ET.Element,
    rows: Sequence[Dict[str, str]],
    result: BuildResult,
    columns: Optional[Sequence[str]] = None,
) -> None:
    """Add one Symbolic_Association child tier per speaker and parsed column."""
    ensure_constraint(root, "Symbolic_Association", SYMBOLIC_ASSOCIATION_DESCRIPTION)
    ensure_linguistic_type(
        root,
        PARSED_CHILD_LINGUISTIC_TYPE_ID,
        time_alignable=False,
        constraints="Symbolic_Association",
    )

    make_annotation_id = next_annotation_id_factory(root, prefix="pa")

    speakers = sorted(
        {
            clean_cell(row.get("speaker_id"))
            for row in rows
            if clean_cell(row.get("speaker_id"))
        }
    )
    result.speakers = speakers

    parsed_columns = list(columns) if columns else get_parsed_columns(rows)

    # One group per source Word tier, not per speaker. A file can hold two Word
    # tiers for the same signer: a second annotation pass under a slightly
    # different tier name: and each set of parsed rows belongs on the tier it
    # was read from. Rows from an older CSV carry no tier, and fall back to
    # finding the tier by the speaker's name, which is what used to happen.
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        speaker_id = clean_cell(row.get("speaker_id"))
        if not speaker_id:
            continue
        groups.setdefault(clean_cell(row.get("tier_id")) or speaker_id, []).append(row)

    for group_key in sorted(groups):
        group_rows = groups[group_key]
        speaker_id = clean_cell(group_rows[0].get("speaker_id"))

        parent_word_tier_id = (
            group_key if find_tier(root, group_key) is not None
            else find_parent_word_tier_id(root, speaker_id)
        )

        if parent_word_tier_id is None:
            result.missing_parent_speakers.append(speaker_id)
            continue

        parent_annotations = collect_parent_annotations(root, parent_word_tier_id)
        exact_index = build_exact_parent_time_index(parent_annotations)

        # Generated tier names stay keyed to the parent tier's participant, so
        # two passes for one signer do not collide.
        tier_name_key = participant_from_tier_id(parent_word_tier_id) or speaker_id

        speaker_rows = group_rows

        for column in parsed_columns:
            parent_to_values: Dict[str, List[str]] = {}

            for row in speaker_rows:
                value = clean_cell(row.get(column))
                if not value:
                    continue

                start_ms = parse_int_ms(row.get("time_start"))
                end_ms = parse_int_ms(row.get("time_end"))

                if start_ms is None or end_ms is None:
                    result.unmatched_rows.append(
                        (
                            speaker_id,
                            column,
                            value,
                            clean_cell(row.get("time_start")),
                            clean_cell(row.get("time_end")),
                        )
                    )
                    continue

                parent_annotation = find_matching_parent_annotation(
                    parent_annotations=parent_annotations,
                    exact_index=exact_index,
                    start_ms=start_ms,
                    end_ms=end_ms,
                )

                if parent_annotation is None:
                    result.unmatched_rows.append(
                        (speaker_id, column, value, str(start_ms), str(end_ms))
                    )
                    continue

                parent_to_values.setdefault(parent_annotation.annotation_id, [])
                append_unique(parent_to_values[parent_annotation.annotation_id], value)

            if not parent_to_values:
                continue

            tier_id = unique_tier_id(
                root,
                PARSED_TIER_TEMPLATE.format(
                    speaker=safe_tier_id(tier_name_key),
                    column=safe_tier_id(column),
                ),
            )

            tier = ET.Element(
                "TIER",
                {
                    "TIER_ID": tier_id,
                    "PARENT_REF": parent_word_tier_id,
                    "LINGUISTIC_TYPE_REF": PARSED_CHILD_LINGUISTIC_TYPE_ID,
                    "PARTICIPANT": tier_name_key,
                },
            )

            for parent_annotation_id, values in parent_to_values.items():
                annotation_element = ET.SubElement(tier, "ANNOTATION")
                ref_annotation = ET.SubElement(
                    annotation_element,
                    "REF_ANNOTATION",
                    {
                        "ANNOTATION_ID": make_annotation_id(),
                        "ANNOTATION_REF": parent_annotation_id,
                    },
                )
                value_element = ET.SubElement(ref_annotation, "ANNOTATION_VALUE")
                value_element.text = PARSED_VALUE_SEPARATOR.join(values)

            insert_in_schema_order(root, tier)
            result.created_tiers.append(
                (tier_id, len(parent_to_values), parent_word_tier_id)
            )


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

def find_broken_annotation_refs(root: ET.Element) -> List[Tuple[str, str, str]]:
    existing_ids = existing_annotation_ids(root)
    broken: List[Tuple[str, str, str]] = []

    for tier in root.findall("TIER"):
        tier_id = tier.attrib.get("TIER_ID", "")
        for ref_annotation in tier.findall(".//REF_ANNOTATION"):
            annotation_id = ref_annotation.attrib.get("ANNOTATION_ID", "")
            parent_id = ref_annotation.attrib.get("ANNOTATION_REF", "")

            if parent_id and parent_id not in existing_ids:
                broken.append((tier_id, annotation_id, parent_id))

    return broken


def find_broken_parent_refs(root: ET.Element) -> List[Tuple[str, str]]:
    tier_ids = set(get_all_tier_ids(root))
    broken: List[Tuple[str, str]] = []

    for tier in root.findall("TIER"):
        tier_id = tier.attrib.get("TIER_ID", "")
        parent_ref = tier.attrib.get("PARENT_REF", "")

        if parent_ref and parent_ref not in tier_ids:
            broken.append((tier_id, parent_ref))

    return broken


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def indent_xml(element: ET.Element, level: int = 0) -> None:
    """Pretty-print XML so the output is readable and diff-friendly."""
    indent = "\n" + level * "  "

    if len(element):
        if not element.text or not element.text.strip():
            element.text = indent + "  "

        for child in element:
            indent_xml(child, level + 1)

        if not element[-1].tail or not element[-1].tail.strip():
            element[-1].tail = indent

    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def build_parsed_eaf(
    input_eaf: Path,
    parsed_rows: Sequence[Dict[str, str]],
    output_eaf: Path,
    columns: Optional[Sequence[str]] = None,
    pretty_print: bool = True,
) -> BuildResult:
    """Write ``output_eaf``: the source document plus the parsed child tiers.

    Every original tier, annotation, time slot, linguistic type and controlled
    vocabulary is preserved untouched.
    """
    input_eaf = Path(input_eaf)
    output_eaf = Path(output_eaf)

    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

    tree = ET.parse(input_eaf)
    root = tree.getroot()

    result = BuildResult(
        file_stem=input_eaf.stem,
        input_eaf=input_eaf,
        original_tiers=get_all_tier_ids(root),
    )

    add_parsed_child_tiers(root, parsed_rows, result, columns=columns)

    result.broken_parent_refs = find_broken_parent_refs(root)
    result.broken_annotation_refs = find_broken_annotation_refs(root)

    output_eaf.parent.mkdir(parents=True, exist_ok=True)

    if pretty_print:
        indent_xml(root)

    tree.write(output_eaf, encoding="UTF-8", xml_declaration=True)
    result.output_eaf = output_eaf

    return result


def write_tier_report(report_path: Path, result: BuildResult, parsed_csv: Optional[Path] = None) -> Path:
    """Write a human-readable report of what happened to one file."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("=" * 80)

    lines.append(f"FILE: {result.file_stem}")
    lines.append(f"Input EAF:  {result.input_eaf}")
    if parsed_csv is not None:
        lines.append(f"Parsed CSV: {parsed_csv}")
    lines.append(f"Output EAF: {result.output_eaf}")

    section("SPEAKERS FOUND IN PARSED ROWS")
    lines.extend(result.speakers or ["(none)"])

    section("ORIGINAL TIERS (ALL PRESERVED)")
    lines.extend(result.original_tiers or ["(none)"])

    section("CREATED PARSED CHILD TIERS")
    if result.created_tiers:
        for tier_id, count, parent in result.created_tiers:
            lines.append(f"{tier_id}: {count} annotations, parent={parent}")
    else:
        lines.append("(none)")

    section("SPEAKERS WITHOUT A WORD-JP PARENT TIER")
    lines.extend(result.missing_parent_speakers or ["(none)"])

    section("UNMATCHED PARSED VALUES")
    if result.unmatched_rows:
        lines.append(f"Total unmatched values: {len(result.unmatched_rows)}")
        lines.append("First 100:")
        for speaker_id, column, value, start, end in result.unmatched_rows[:100]:
            lines.append(
                f"speaker={speaker_id}, column={column}, time=({start}, {end}), value={value!r}"
            )
    else:
        lines.append("(none)")

    section("BROKEN TIER PARENT_REFS")
    if result.broken_parent_refs:
        for tier_id, parent_ref in result.broken_parent_refs:
            lines.append(f"tier={tier_id}, missing_parent_tier={parent_ref}")
    else:
        lines.append("(none)")

    section("BROKEN ANNOTATION_REFS")
    if result.broken_annotation_refs:
        for tier_id, annotation_id, parent_id in result.broken_annotation_refs[:100]:
            lines.append(
                f"tier={tier_id}, annotation={annotation_id}, missing_parent_annotation={parent_id}"
            )
        if len(result.broken_annotation_refs) > 100:
            lines.append(f"... plus {len(result.broken_annotation_refs) - 100} more")
    else:
        lines.append("(none)")

    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return report_path
