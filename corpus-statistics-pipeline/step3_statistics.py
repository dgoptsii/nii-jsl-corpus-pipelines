"""Step 3: every count, globally and per prefecture.

Reads the annotation table from step 2 and writes the tables that answer the
corpus-description questions: how much material there is, how much of it parsed,
what the vocabulary looks like, how often each marker fires, and what a
machine-learning user would need to know before planning a split.

Rerunnable on its own: it reads only the two CSVs written by steps 1 and 2, so
changing a threshold in ``config.py`` costs seconds, not another pass over the
corpus.

    python3 step3_statistics.py --out output
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import (
    ANNOTATIONS_FILE,
    CLASS_SIZE_THRESHOLDS,
    COVERAGE_CUTOFFS,
    DEFAULT_OUTPUT_FOLDER,
    DEFAULT_TEST_FRACTION,
    ELAN_INDEX_FILE,
    GLOBAL_TAG,
    OCCURRENCE_CAP,
    TABLES_SUBFOLDER,
    TOP_N_SPECS,
)
from io_utils import file_key, read_csv_safely, region_label, write_csv
from latex import write_report_tables
from metrics import (
    add_flags,
    gloss_statistics,
    key_counts,
    per_file,
    summary,
)
from mlready import (
    class_size_table,
    coverage_curve,
    duration_distribution,
    examples_per_signer,
    full_coverage_curve,
    marker_cooccurrence,
    signer_balance,
    signing_rate,
    split_feasibility,
)
from topn import coverage_table, gloss_statistics_for_regions, top_glosses


def load_inputs(out_root: Path):
    """The annotation table and the ELAN index, with flags already added."""
    annotations_path = Path(out_root) / ANNOTATIONS_FILE
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"{annotations_path} not found. Run step2_build_table.py first.")

    annotations = read_csv_safely(annotations_path)
    if "is_parsed" not in annotations.columns:
        annotations = add_flags(annotations)
    else:
        for column in ["is_parsed", "is_compound", "is_ambiguous", "has_lexical",
                       "has_any_key", "has_blocking_key", "lexical_only",
                       "lexical_with_key", "key_only", "empty_row"]:
            if column in annotations.columns:
                annotations[column] = (annotations[column].astype(str).str.lower()
                                       .isin({"true", "1", "yes"}))
        annotations["n_keys"] = pd.to_numeric(annotations.get("n_keys"),
                                              errors="coerce").fillna(0).astype(int)

    index_path = Path(out_root) / ELAN_INDEX_FILE
    elan_index = read_csv_safely(index_path) if index_path.exists() else None
    return annotations, elan_index


def _index_for(elan_index: Optional[pd.DataFrame],
               annotations: pd.DataFrame) -> Optional[pd.DataFrame]:
    """The ELAN rows belonging to this slice of the annotation table.

    Restricting by file key rather than by region code means a regional duration
    counts only recordings that actually contributed annotations, so the
    annotations-per-minute figures stay comparable across prefectures.
    """
    if elan_index is None or elan_index.empty:
        return None
    keys = set(annotations["file_key"]) if "file_key" in annotations.columns \
        else {file_key(name) for name in annotations["source_file"].unique()}
    return elan_index[elan_index["file_key"].isin(keys)]


def compute(annotations: pd.DataFrame,
            elan_index: Optional[pd.DataFrame],
            test_fraction: float = DEFAULT_TEST_FRACTION,
            top_n_for_signer_shape: int = 200) -> Dict[str, pd.DataFrame]:
    """Every table, keyed by the name it will be written under."""
    regions = sorted(annotations["region_code"].dropna().unique())

    summaries: List[pd.DataFrame] = [
        summary(annotations, _index_for(elan_index, annotations), GLOBAL_TAG)]
    keys: List[pd.DataFrame] = [key_counts(annotations, GLOBAL_TAG)]
    coverage: List[pd.DataFrame] = []
    curves: List[pd.DataFrame] = []
    classes: List[pd.DataFrame] = []
    splits: List[pd.DataFrame] = []
    durations: List[pd.DataFrame] = [duration_distribution(annotations, GLOBAL_TAG)]
    cooccurrence: List[pd.DataFrame] = [marker_cooccurrence(annotations, GLOBAL_TAG)]

    global_stats = gloss_statistics(annotations)
    coverage.append(coverage_table(global_stats, TOP_N_SPECS, OCCURRENCE_CAP, GLOBAL_TAG))
    curves.append(coverage_curve(global_stats, COVERAGE_CUTOFFS, GLOBAL_TAG))
    classes.append(class_size_table(global_stats, CLASS_SIZE_THRESHOLDS,
                                    label=GLOBAL_TAG))
    splits.append(split_feasibility(annotations, test_fraction, label=GLOBAL_TAG))

    for code in regions:
        group = annotations[annotations["region_code"] == code]
        label = str(code)
        summaries.append(summary(group, _index_for(elan_index, group), label))
        keys.append(key_counts(group, label))
        stats = gloss_statistics(group)
        if stats.empty:
            continue
        coverage.append(coverage_table(stats, TOP_N_SPECS, OCCURRENCE_CAP, label))
        curves.append(coverage_curve(stats, COVERAGE_CUTOFFS, label))
        classes.append(class_size_table(stats, CLASS_SIZE_THRESHOLDS, label=label))
        durations.append(duration_distribution(group, label))
        cooccurrence.append(marker_cooccurrence(group, label))
        if group["speaker_id"].nunique() >= 2:
            splits.append(split_feasibility(group, test_fraction, label=label))

    summary_table = pd.concat(summaries, ignore_index=True)
    summary_table.insert(1, "region_name",
                         [region_label(t) if t != GLOBAL_TAG else "All prefectures"
                          for t in summary_table["tag"]])

    tables: Dict[str, pd.DataFrame] = {
        "summary": summary_table,
        "keys": pd.concat(keys, ignore_index=True),
        "per_file": per_file(annotations, elan_index),
        "gloss_statistics": global_stats,
        "coverage": pd.concat(coverage, ignore_index=True),
        "coverage_curve": pd.concat(curves, ignore_index=True),
        "coverage_curve_full": full_coverage_curve(global_stats),
        "class_sizes": pd.concat(classes, ignore_index=True),
        "duration_distribution": pd.concat([d for d in durations if not d.empty],
                                           ignore_index=True),
        "marker_cooccurrence": pd.concat([c for c in cooccurrence if not c.empty],
                                         ignore_index=True)
                               if any(not c.empty for c in cooccurrence)
                               else pd.DataFrame(),
        "signing_rate": signing_rate(annotations, elan_index),
        "signer_balance": signer_balance(annotations),
        "split_feasibility": pd.concat([s for s in splits if not s.empty],
                                       ignore_index=True)
                             if any(not s.empty for s in splits) else pd.DataFrame(),
        "examples_per_signer": examples_per_signer(annotations, top_n_for_signer_shape),
    }

    # The three headline gloss lists, ready to hand to anyone building a lexicon.
    for label, top, min_signers in TOP_N_SPECS:
        slug = label.lower().replace(" ", "_")
        tables[f"glosses_{slug}"] = top_glosses(global_stats, top, min_signers)

    return tables


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION,
                        help="token share held out in the signer-disjoint split")
    parser.add_argument("--no-latex", action="store_true",
                        help="skip the .tex exports")
    args = parser.parse_args(argv)

    annotations, elan_index = load_inputs(args.out)
    tables = compute(annotations, elan_index, args.test_fraction)

    folder = Path(args.out) / TABLES_SUBFOLDER
    for name, frame in tables.items():
        if frame is not None and not frame.empty:
            write_csv(folder / f"{name}.csv", frame)

    if not args.no_latex:
        written = write_report_tables(tables, folder)
        print(f"Wrote {len(written)} LaTeX table(s)")

    headline = tables["summary"].iloc[0]
    print(f"\n{'Corpus':<28}{headline['n_files_parsed']} files, "
          f"{headline['total_recording_hms']}, {headline['n_signers']} signers")
    print(f"{'Annotations':<28}{headline['n_annotations']:,} "
          f"({headline['n_parsed']:,} parsed, "
          f"{headline['n_ambiguous']:,} ambiguous, "
          f"{headline['n_compound']:,} compound)")
    print(f"{'Vocabulary':<28}{headline['n_unique_lexical_items']:,} lexical items, "
          f"{headline['n_lexical_items_occurring_once']:,} seen once "
          f"({headline['hapax_percent_of_vocabulary']}%)")
    print(f"\nWrote {len([f for f in tables.values() if f is not None and not f.empty])} "
          f"table(s) to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
