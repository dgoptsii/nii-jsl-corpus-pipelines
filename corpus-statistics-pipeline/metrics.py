"""Corpus counts, computed once and reported globally and per prefecture.

Every function takes the annotation table built by step 2 and returns a
DataFrame. None of them filter by region: the driver calls them once per region
and once for the whole corpus, so a regional and a global number can never be
computed different ways.

**Successfully parsed** means the row carries neither the ``compound`` nor the
``ambiguous`` flag. Both are legitimate outcomes and are counted, but neither is
part of the analysable set. **Unique** is reported for two things: unique
annotation strings measure notational variety, unique lexical items measure
vocabulary size, and only the second is what a machine-learning user means by
"vocabulary".
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from config import (
    AMBIGUOUS_COLUMN,
    ANNOTATION_COLUMN,
    BLOCKING_KEYS,
    COMPOUND_COLUMN,
    KEY_COLUMNS,
    LEXICAL_COLUMN,
    SIGNER_COLUMN,
)
from io_utils import file_key, format_hms, nonempty


# ===========================================================================
# ROW CLASSIFICATION
# ===========================================================================

def add_flags(annotations: pd.DataFrame) -> pd.DataFrame:
    """Add the boolean columns every later count depends on.

    Doing this once, in one place, is what keeps the definitions consistent
    between the global and the per-region tables.
    """
    frame = annotations.copy()

    frame["is_compound"] = nonempty(frame.get(COMPOUND_COLUMN, pd.Series("", index=frame.index)))
    frame["is_ambiguous"] = nonempty(frame.get(AMBIGUOUS_COLUMN, pd.Series("", index=frame.index)))
    frame["is_parsed"] = ~(frame["is_compound"] | frame["is_ambiguous"])

    frame["has_lexical"] = nonempty(frame.get(LEXICAL_COLUMN, pd.Series("", index=frame.index)))

    present = pd.DataFrame(index=frame.index)
    for key in KEY_COLUMNS:
        present[key] = nonempty(frame[key]) if key in frame.columns else False
    frame["n_keys"] = present.sum(axis=1)
    frame["has_any_key"] = frame["n_keys"] > 0
    frame["has_blocking_key"] = present[[k for k in BLOCKING_KEYS
                                         if k in present.columns]].any(axis=1)

    # The three mutually exclusive shapes an analysable annotation can take.
    frame["lexical_only"] = frame["is_parsed"] & frame["has_lexical"] & ~frame["has_any_key"]
    frame["lexical_with_key"] = frame["is_parsed"] & frame["has_lexical"] & frame["has_any_key"]
    frame["key_only"] = frame["is_parsed"] & ~frame["has_lexical"] & frame["has_any_key"]
    frame["empty_row"] = frame["is_parsed"] & ~frame["has_lexical"] & ~frame["has_any_key"]

    return frame


# ===========================================================================
# HEADLINE SUMMARY
# ===========================================================================

def summary(annotations: pd.DataFrame, elan_index: Optional[pd.DataFrame] = None,
            label: str = "GLOBAL") -> pd.DataFrame:
    """One row: every headline count for a corpus or a prefecture."""
    frame = annotations
    parsed = frame[frame["is_parsed"]]

    duration_ms = 0.0
    n_files_elan = 0
    if elan_index is not None and not elan_index.empty:
        duration_ms = float(pd.to_numeric(elan_index["duration_ms"],
                                          errors="coerce").fillna(0).sum())
        n_files_elan = int(len(elan_index))

    lexical_values = parsed.loc[parsed["has_lexical"], LEXICAL_COLUMN]
    lexical_counts = lexical_values.value_counts()

    row: Dict[str, object] = {
        "tag": label,
        "n_files_parsed": int(frame["source_file"].nunique()),
        "n_files_elan_matched": n_files_elan,
        "total_recording_ms": round(duration_ms, 1),
        "total_recording_hms": format_hms(duration_ms),
        "n_signers": int(frame[SIGNER_COLUMN].nunique()),

        "n_annotations": int(len(frame)),
        "n_parsed": int(len(parsed)),
        "n_ambiguous": int(frame["is_ambiguous"].sum()),
        "n_compound": int(frame["is_compound"].sum()),

        "n_unique_annotation_strings": int(parsed[ANNOTATION_COLUMN].nunique())
                                        if ANNOTATION_COLUMN in parsed.columns else 0,
        "n_unique_lexical_items": int(lexical_values.nunique()),
        "n_lexical_items_occurring_once": int((lexical_counts == 1).sum()),

        "n_with_lexical": int(parsed["has_lexical"].sum()),
        "n_lexical_only": int(parsed["lexical_only"].sum()),
        "n_key_only": int(parsed["key_only"].sum()),
        "n_lexical_with_key": int(parsed["lexical_with_key"].sum()),
        "n_empty": int(parsed["empty_row"].sum()),
    }

    total = row["n_annotations"] or 1
    parsed_total = row["n_parsed"] or 1
    row["parsed_percent"] = round(100 * row["n_parsed"] / total, 2)
    row["ambiguous_percent"] = round(100 * row["n_ambiguous"] / total, 2)
    row["compound_percent"] = round(100 * row["n_compound"] / total, 2)
    row["lexical_only_percent"] = round(100 * row["n_lexical_only"] / parsed_total, 2)
    row["key_only_percent"] = round(100 * row["n_key_only"] / parsed_total, 2)
    row["hapax_percent_of_vocabulary"] = round(
        100 * row["n_lexical_items_occurring_once"]
        / (row["n_unique_lexical_items"] or 1), 2)

    if duration_ms > 0:
        row["annotations_per_minute"] = round(len(frame) / (duration_ms / 60000.0), 2)
    else:
        row["annotations_per_minute"] = float("nan")

    return pd.DataFrame([row])


# ===========================================================================
# PER-KEY COUNTS
# ===========================================================================

def key_counts(annotations: pd.DataFrame, label: str = "GLOBAL") -> pd.DataFrame:
    """How often each marker fires, as a share of successfully parsed rows.

    The denominator is the parsed set, not every annotation: a marker cannot
    fire on a row the parser refused to resolve, so including those rows would
    understate every percentage by the same arbitrary amount.
    """
    parsed = annotations[annotations["is_parsed"]]
    denominator = len(parsed) or 1

    rows = []
    for key in KEY_COLUMNS:
        if key not in parsed.columns:
            continue
        present = nonempty(parsed[key])
        count = int(present.sum())
        rows.append({
            "tag": label,
            "key": key,
            "kind": "blocking" if key in BLOCKING_KEYS else "linguistic",
            "n_annotations": count,
            "percent_of_parsed": round(100 * count / denominator, 3),
            "n_signers": int(parsed.loc[present, SIGNER_COLUMN].nunique()),
            "n_unique_values": int(parsed.loc[present, key].nunique()),
        })

    return (pd.DataFrame(rows)
            .sort_values("n_annotations", ascending=False)
            .reset_index(drop=True))


# ===========================================================================
# PER-FILE DETAIL
# ===========================================================================

def per_file(annotations: pd.DataFrame,
             elan_index: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per parsed file: the detail behind the regional totals."""
    rows = []
    # Keyed on the normalised recording identity, not the raw filename: the
    # index names an .eaf and the annotations name a parsed .csv, so matching
    # the strings themselves silently yields a duration of zero for every file.
    durations = {}
    if elan_index is not None and not elan_index.empty:
        durations = {
            file_key(name): value
            for name, value in zip(elan_index["source_file"],
                                   pd.to_numeric(elan_index["duration_ms"],
                                                 errors="coerce").fillna(0))
        }

    for source, group in annotations.groupby("source_file", sort=True):
        parsed = group[group["is_parsed"]]
        duration = float(durations.get(file_key(source), 0.0))
        rows.append({
            "source_file": source,
            "region_code": group["region_code"].iloc[0],
            "duration_ms": round(duration, 1),
            "duration_hms": format_hms(duration),
            "n_signers": int(group[SIGNER_COLUMN].nunique()),
            "n_annotations": int(len(group)),
            "n_parsed": int(len(parsed)),
            "n_ambiguous": int(group["is_ambiguous"].sum()),
            "n_compound": int(group["is_compound"].sum()),
            "ambiguous_percent": round(100 * group["is_ambiguous"].sum() / (len(group) or 1), 2),
            "n_unique_lexical_items": int(
                parsed.loc[parsed["has_lexical"], LEXICAL_COLUMN].nunique()),
        })

    return pd.DataFrame(rows)


# ===========================================================================
# GLOSS FREQUENCY AND SIGNER COVERAGE
# ===========================================================================

def gloss_statistics(annotations: pd.DataFrame) -> pd.DataFrame:
    """One row per lexical item: occurrences, signers, files, regions.

    Rows carrying a marker are included: a gloss is a gloss whether or not the
    annotation around it also recorded a mouthing. Only unparsed rows are
    excluded, since their lexical field was never resolved.
    """
    parsed = annotations[annotations["is_parsed"] & annotations["has_lexical"]]
    if parsed.empty:
        return pd.DataFrame(columns=["gloss", "occurrences", "n_signers",
                                     "n_files", "n_regions", "regions"])

    grouped = parsed.groupby(LEXICAL_COLUMN, sort=False)
    stats = grouped.agg(
        occurrences=(LEXICAL_COLUMN, "size"),
        n_signers=(SIGNER_COLUMN, "nunique"),
        n_files=("source_file", "nunique"),
        n_regions=("region_code", "nunique"),
    ).reset_index().rename(columns={LEXICAL_COLUMN: "gloss"})

    regions = (grouped["region_code"]
               .agg(lambda values: ";".join(sorted(set(values))))
               .reset_index(name="regions")
               .rename(columns={LEXICAL_COLUMN: "gloss"}))
    stats = stats.merge(regions, on="gloss", how="left")
    stats["occurrences_per_signer"] = (stats["occurrences"]
                                       / stats["n_signers"]).round(2)

    return (stats.sort_values(["occurrences", "n_signers", "gloss"],
                              ascending=[False, False, True])
            .reset_index(drop=True))
