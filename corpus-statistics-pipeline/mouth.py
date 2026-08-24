"""How manual annotation co-occurs with mouth action.

Only a subset of the corpus carries MouthAction tiers, so everything here is
reported against its own denominator (the annotations that *could* have been
matched), never against the whole corpus.

Three questions, for the corpus and for each prefecture: what each marker
co-occurs with; what a bare lexical item co-occurs with (the plain lexical
signs, the natural comparison class); and whether the tiers agree, when a
recording has one MouthAction tier per signer or several annotators' passes.

Agreement is measured per *label*, not per annotation, so the numbers can be
read against the total inventory of mouth labels. A label is **disagreed** when
a label from another tier overlaps it and puts it in a different category, and
**agreed** otherwise. Labels no other tier overlaps therefore fall into
"agreed", which is weaker than confirmation, so they are also counted as
``n_uncontested``. A disagreement is often not an error but a sign that the
annotation spans a boundary between two mouth actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from config import (
    GLOBAL_TAG,
    KEY_COLUMNS,
    LEXICAL_COLUMN,
    MIN_OVERLAP_MS,
    MOUTH_CATEGORIES,
    SIGNER_COLUMN,
)
from elan import ElanFile, classify_mouth_value, read_elan
from io_utils import file_key, nonempty, region_of

LEXICAL_UNIT = "lexical_item (no marker)"
ANY_UNIT = "any annotation"


# ===========================================================================
# LOADING THE MOUTH TIERS
# ===========================================================================

def load_mouth_files(elan_index: pd.DataFrame,
                     only_with_mouth_tiers: bool = True) -> Dict[str, ElanFile]:
    """Re-read the .eaf files that have MouthAction tiers, keyed by file_key.

    Step 1 records *whether* a file has mouth tiers; the annotations themselves
    are too bulky to carry in an index, so they are read again here for the
    files that need them.
    """
    if elan_index is None or elan_index.empty:
        return {}

    frame = elan_index
    if only_with_mouth_tiers and "has_mouth_tiers" in frame.columns:
        flag = frame["has_mouth_tiers"].astype(str).str.lower()
        frame = frame[flag.isin({"true", "1", "yes"})]

    loaded: Dict[str, ElanFile] = {}
    for _, row in frame.iterrows():
        path = Path(str(row.get("path", "")))
        if not path.exists():
            continue
        try:
            document = read_elan(path)
        except ValueError:
            continue
        if only_with_mouth_tiers and not document.has_mouth_tiers:
            continue
        loaded[file_key(path.name)] = document
    return loaded


# ===========================================================================
# LABEL-LEVEL AGREEMENT
# ===========================================================================

def label_table(documents: Dict[str, ElanFile],
                min_overlap_ms: float = MIN_OVERLAP_MS) -> pd.DataFrame:
    """Every MouthAction label in the corpus, with its agreement status.

    Two labels are counterparts when they overlap in time and come from
    different tiers. A label is *disagreed* when at least one counterpart puts
    it in a different category, and *agreed* otherwise, including when it has
    no counterpart at all, which is recorded as ``uncontested`` so the weaker
    claim can be separated from genuine confirmation.
    """
    columns = ["file_key", "source_file", "region_code", "annotation_id",
               "tier_id", "start_ms", "end_ms", "value", "category",
               "n_counterparts", "n_agreeing", "n_conflicting", "status"]
    rows = []

    for key, document in documents.items():
        labels = [(a, classify_mouth_value(a.value)) for a in document.mouth_annotations]
        labels = [(a, kind) for a, kind in labels if kind]

        for annotation, category in labels:
            agreeing = conflicting = 0
            for other, other_category in labels:
                if other.annotation_id == annotation.annotation_id:
                    continue
                if other.tier_id == annotation.tier_id:
                    continue
                shared = (min(other.end_ms, annotation.end_ms)
                          - max(other.start_ms, annotation.start_ms))
                if shared < min_overlap_ms:
                    continue
                if other_category == category:
                    agreeing += 1
                else:
                    conflicting += 1

            counterparts = agreeing + conflicting
            rows.append({
                "file_key": key,
                "source_file": document.path.name,
                "region_code": region_of(document.path.name),
                "annotation_id": annotation.annotation_id,
                "tier_id": annotation.tier_id,
                "start_ms": annotation.start_ms,
                "end_ms": annotation.end_ms,
                "value": annotation.value,
                "category": category,
                "n_counterparts": counterparts,
                "n_agreeing": agreeing,
                "n_conflicting": conflicting,
                "status": "disagreed" if conflicting else "agreed",
            })

    return pd.DataFrame(rows, columns=columns)


def category_counts(labels: pd.DataFrame, label: str = GLOBAL_TAG) -> pd.DataFrame:
    """Mouth labels per category, split into agreed and disagreed."""
    columns = ["tag", "category", "n_labels", "n_agreed", "n_disagreed",
               "n_uncontested", "percent_agreed", "percent_disagreed",
               "percent_uncontested", "n_files"]
    if labels.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for category in MOUTH_CATEGORIES:
        subset = labels[labels["category"] == category]
        total = len(subset)
        if total == 0:
            continue
        agreed = int((subset["status"] == "agreed").sum())
        uncontested = int((subset["n_counterparts"] == 0).sum())
        rows.append({
            "tag": label,
            "category": category,
            "n_labels": total,
            "n_agreed": agreed,
            "n_disagreed": total - agreed,
            "n_uncontested": uncontested,
            "percent_agreed": round(100 * agreed / total, 2),
            "percent_disagreed": round(100 * (total - agreed) / total, 2),
            "percent_uncontested": round(100 * uncontested / total, 2),
            "n_files": int(subset["file_key"].nunique()),
        })
    return pd.DataFrame(rows, columns=columns)


def category_counts_by_region(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return category_counts(labels)
    tables = [category_counts(labels, GLOBAL_TAG)]
    for code, group in labels.groupby("region_code", sort=True):
        tables.append(category_counts(group, str(code)))
    return pd.concat(tables, ignore_index=True)


# ===========================================================================
# LABELLING EACH ANNOTATION
# ===========================================================================

def annotate_with_mouth(annotations: pd.DataFrame,
                        documents: Dict[str, ElanFile],
                        min_overlap_ms: float = MIN_OVERLAP_MS) -> pd.DataFrame:
    """Attach the overlapping mouth categories to every annotation row.

    Returns a copy of ``annotations`` restricted to rows whose recording has
    MouthAction tiers, with four columns added:

    ``mouth_categories``   the distinct categories overlapping this annotation
    ``n_mouth_labels``     how many labels overlapped, across all tiers
    ``n_mouth_tiers``      how many distinct tiers contributed one
    ``mouth_agreement``    none / single / agree / disagree
    """
    if annotations.empty or not documents:
        return _empty_labelled(annotations)

    keyed = annotations.copy()
    keyed["file_key"] = keyed["source_file"].map(file_key)
    keyed = keyed[keyed["file_key"].isin(documents)]
    if keyed.empty:
        return _empty_labelled(annotations)

    starts = pd.to_numeric(keyed.get("time_start"), errors="coerce")
    ends = pd.to_numeric(keyed.get("time_end"), errors="coerce")

    categories: List[str] = []
    label_counts: List[int] = []
    tier_counts: List[int] = []
    agreement: List[str] = []
    label_ids: List[str] = []

    for (_, row), start, end in zip(keyed.iterrows(), starts, ends):
        if pd.isna(start) or pd.isna(end):
            categories.append("")
            label_counts.append(0)
            tier_counts.append(0)
            agreement.append("no times")
            label_ids.append("")
            continue

        document = documents[row["file_key"]]
        overlapping = document.mouth_labels_overlapping(
            float(start), float(end),
            speaker_id=str(row.get(SIGNER_COLUMN, "")),
            min_overlap_ms=min_overlap_ms,
        )
        classified = [(a.tier_id, classify_mouth_value(a.value)) for a in overlapping]
        classified = [(tier, kind) for tier, kind in classified if kind]

        found = sorted({kind for _tier, kind in classified})
        tiers = {tier for tier, _kind in classified}

        categories.append(";".join(found))
        label_counts.append(len(classified))
        tier_counts.append(len(tiers))
        label_ids.append(";".join(a.annotation_id for a in overlapping))
        if not classified:
            agreement.append("none")
        elif len(classified) == 1:
            agreement.append("single")
        elif len(found) == 1:
            agreement.append("agree")
        else:
            agreement.append("disagree")

    keyed["mouth_categories"] = categories
    keyed["n_mouth_labels"] = label_counts
    keyed["n_mouth_tiers"] = tier_counts
    keyed["mouth_agreement"] = agreement
    keyed["mouth_label_ids"] = label_ids
    return keyed


def _empty_labelled(annotations: pd.DataFrame) -> pd.DataFrame:
    empty = annotations.iloc[0:0].copy()
    for column, default in [("file_key", ""), ("mouth_categories", ""),
                            ("n_mouth_labels", 0), ("n_mouth_tiers", 0),
                            ("mouth_agreement", ""), ("mouth_label_ids", "")]:
        empty[column] = pd.Series(dtype=type(default))
    return empty


# ===========================================================================
# MARKER x CATEGORY
# ===========================================================================

def key_category_table(labelled: pd.DataFrame,
                       labels: pd.DataFrame,
                       label: str = GLOBAL_TAG) -> pd.DataFrame:
    """For each marker and each mouth category: labels, agreement, rows reached.

    Two units appear side by side because they answer different questions.
    ``n_labels`` counts the mouth annotations that overlapped: the unit the
    agreement figures are about. ``n_rows`` counts the marker's own annotations
    that touched at least one label of that category, and ``percent_of_rows``
    puts it over every annotation carrying the marker, which is the reading a
    linguist wants: how often this marker comes with mouthing at all.
    """
    columns = ["tag", "unit", "category", "n_labels", "n_labels_agreed",
               "n_labels_disagreed", "n_rows", "n_unit_rows", "percent_of_rows"]
    if labelled.empty or labels.empty:
        return pd.DataFrame(columns=columns)

    parsed = labelled[labelled["is_parsed"]] if "is_parsed" in labelled.columns \
        else labelled

    status = dict(zip(zip(labels["file_key"], labels["annotation_id"]),
                      zip(labels["category"], labels["status"])))

    rows = []
    for unit, mask in _unit_masks(parsed):
        subset = parsed[mask]
        if subset.empty:
            continue
        counts = {c: {"labels": 0, "agreed": 0, "disagreed": 0, "rows": 0}
                  for c in MOUTH_CATEGORIES}

        for file_id, ids in zip(subset["file_key"], subset["mouth_label_ids"]):
            seen = set()
            for annotation_id in str(ids or "").split(";"):
                if not annotation_id:
                    continue
                entry = status.get((file_id, annotation_id))
                if entry is None:
                    continue
                category, state = entry
                counts[category]["labels"] += 1
                counts[category]["agreed" if state == "agreed" else "disagreed"] += 1
                seen.add(category)
            for category in seen:
                counts[category]["rows"] += 1

        for category in MOUTH_CATEGORIES:
            entry = counts[category]
            rows.append({
                "tag": label,
                "unit": unit,
                "category": category,
                "n_labels": entry["labels"],
                "n_labels_agreed": entry["agreed"],
                "n_labels_disagreed": entry["disagreed"],
                "n_rows": entry["rows"],
                "n_unit_rows": int(len(subset)),
                "percent_of_rows": round(100 * entry["rows"] / (len(subset) or 1), 2),
            })

    return pd.DataFrame(rows, columns=columns)


# ===========================================================================
# THE OVERLAP TABLE
# ===========================================================================

def _unit_masks(labelled: pd.DataFrame) -> List[tuple]:
    """(unit name, row mask) for every marker plus the bare-lexical class."""
    units: List[tuple] = [(ANY_UNIT, pd.Series(True, index=labelled.index))]
    for key in KEY_COLUMNS:
        if key in labelled.columns:
            units.append((key, nonempty(labelled[key])))
    if LEXICAL_COLUMN in labelled.columns:
        units.append((LEXICAL_UNIT, labelled["lexical_only"]))
    return units


def overlap_table(labelled: pd.DataFrame, label: str = GLOBAL_TAG) -> pd.DataFrame:
    """One row per marker (plus bare lexical items): what the mouth was doing.

    Percentages use the number of annotations of that unit inside
    mouth-annotated recordings, which is the only honest denominator: a marker
    that never appears in a file with MouthAction tiers has no evidence either
    way, and should not be reported as 0%.
    """
    columns = ["tag", "unit", "kind", "n_annotations", "n_with_mouth",
               "percent_with_mouth", "n_Mouthing", "n_MouthGesture", "n_Others",
               "percent_Mouthing", "percent_MouthGesture", "percent_Others",
               "n_multi_label", "n_agree", "n_disagree", "percent_agreement",
               "n_signers"]
    if labelled.empty:
        return pd.DataFrame(columns=columns)

    parsed = labelled[labelled["is_parsed"]] if "is_parsed" in labelled.columns else labelled

    rows = []
    for unit, mask in _unit_masks(parsed):
        subset = parsed[mask]
        total = len(subset)
        if total == 0:
            continue

        with_mouth = subset[subset["n_mouth_labels"] > 0]
        denominator = len(with_mouth) or 1

        counts = {}
        for category in MOUTH_CATEGORIES:
            counts[category] = int(
                with_mouth["mouth_categories"]
                .str.split(";").apply(lambda values: category in values).sum())

        multi = subset[subset["n_mouth_labels"] > 1]
        agree = int((multi["mouth_agreement"] == "agree").sum())
        disagree = int((multi["mouth_agreement"] == "disagree").sum())

        row = {
            "tag": label,
            "unit": unit,
            "kind": ("summary" if unit == ANY_UNIT
                     else "lexical" if unit == LEXICAL_UNIT else "key"),
            "n_annotations": total,
            "n_with_mouth": len(with_mouth),
            "percent_with_mouth": round(100 * len(with_mouth) / total, 2),
            "n_multi_label": int(len(multi)),
            "n_agree": agree,
            "n_disagree": disagree,
            "percent_agreement": round(100 * agree / (agree + disagree), 2)
                                 if (agree + disagree) else float("nan"),
            "n_signers": int(subset[SIGNER_COLUMN].nunique())
                         if SIGNER_COLUMN in subset.columns else 0,
        }
        for category in MOUTH_CATEGORIES:
            row[f"n_{category}"] = counts[category]
            row[f"percent_{category}"] = round(100 * counts[category] / denominator, 2)
        rows.append(row)

    table = pd.DataFrame(rows)[columns]
    # Summary first, then the bare-lexical comparison class, then the markers by
    # frequency: the reader needs the baseline before the rows compared to it.
    order = {"summary": 0, "lexical": 1, "key": 2}
    table["_order"] = table["kind"].map(order)
    return (table.sort_values(["_order", "n_annotations"], ascending=[True, False])
            .drop(columns="_order").reset_index(drop=True))


def overlap_table_by_region(labelled: pd.DataFrame) -> pd.DataFrame:
    """The overlap table for the corpus and for each prefecture."""
    if labelled.empty:
        return overlap_table(labelled)
    tables = [overlap_table(labelled, GLOBAL_TAG)]
    for code, group in labelled.groupby("region_code", sort=True):
        tables.append(overlap_table(group, str(code)))
    return pd.concat(tables, ignore_index=True)


# ===========================================================================
# DISAGREEMENT DETAIL
# ===========================================================================

def disagreement_detail(labelled: pd.DataFrame, limit: int = 500) -> pd.DataFrame:
    """The annotations whose mouth tiers disagreed, for manual inspection.

    A count of disagreements is not actionable on its own; this is the list an
    annotator would open in ELAN to see what happened.
    """
    if labelled.empty:
        return pd.DataFrame(columns=["source_file", SIGNER_COLUMN, "time_start",
                                     "time_end", LEXICAL_COLUMN,
                                     "mouth_categories", "n_mouth_tiers"])
    rows = labelled[labelled["mouth_agreement"] == "disagree"]
    wanted = [c for c in ["source_file", SIGNER_COLUMN, "time_start", "time_end",
                          "annotation", LEXICAL_COLUMN, "mouth_categories",
                          "n_mouth_labels", "n_mouth_tiers"] if c in rows.columns]
    return rows[wanted].head(int(limit)).reset_index(drop=True)


def coverage_summary(elan_index: pd.DataFrame,
                     labelled: pd.DataFrame) -> pd.DataFrame:
    """How much of the corpus this analysis actually speaks for."""
    if elan_index is None or elan_index.empty:
        return pd.DataFrame([{"n_elan_files": 0, "n_with_mouth_tiers": 0,
                              "percent_files_with_mouth_tiers": 0.0,
                              "n_annotations_covered": int(len(labelled))}])

    flag = elan_index.get("has_mouth_tiers", pd.Series("", index=elan_index.index))
    with_mouth = flag.astype(str).str.lower().isin({"true", "1", "yes"})
    duration = pd.to_numeric(elan_index.get("duration_ms", 0), errors="coerce").fillna(0)

    return pd.DataFrame([{
        "n_elan_files": int(len(elan_index)),
        "n_with_mouth_tiers": int(with_mouth.sum()),
        "percent_files_with_mouth_tiers": round(
            100 * with_mouth.sum() / (len(elan_index) or 1), 2),
        "recording_ms_with_mouth_tiers": round(float(duration[with_mouth].sum()), 1),
        "percent_recording_with_mouth_tiers": round(
            100 * duration[with_mouth].sum() / (duration.sum() or 1), 2),
        "n_annotations_covered": int(len(labelled)),
        "n_annotations_with_overlap": int((labelled["n_mouth_labels"] > 0).sum())
                                      if not labelled.empty else 0,
    }])
