"""Step 4 -- MouthAction overlap.

Only for recordings that have MouthAction tiers. For each such recording the
.eaf is read again, every parsed annotation is matched to the mouth labels that
overlap it in time, and the results are aggregated per marker and for bare
lexical items.

Where several mouth labels overlap one annotation -- because the recording has
one tier per signer, or a segmentation tier and a category tier -- the stage
reports both how often they agree and how often they do not, and writes the
disagreements out for inspection.

    python3 step4_mouth_overlap.py --out output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    ANNOTATIONS_FILE,
    DEFAULT_OUTPUT_FOLDER,
    DIAGNOSTICS_SUBFOLDER,
    ELAN_INDEX_FILE,
    MIN_OVERLAP_MS,
    TABLES_SUBFOLDER,
)
from io_utils import read_csv_safely, write_csv
from latex import MOUTH_COLUMNS, MOUTH_HEADERS, write_report_tables
from mouth import (
    annotate_with_mouth,
    category_counts_by_region,
    coverage_summary,
    disagreement_detail,
    key_category_table,
    label_table,
    load_mouth_files,
    overlap_table_by_region,
)
from step3_statistics import load_inputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--min-overlap-ms", type=float, default=MIN_OVERLAP_MS,
                        help="milliseconds two intervals must share to overlap")
    parser.add_argument("--save-labelled", action="store_true",
                        help="also write the per-annotation mouth labels")
    parser.add_argument("--no-latex", action="store_true")
    args = parser.parse_args(argv)

    annotations, elan_index = load_inputs(args.out)
    if elan_index is None or elan_index.empty:
        print("No ELAN index: run step1_index_elan.py before this stage.")
        return 1

    documents = load_mouth_files(elan_index)
    print(f"{len(documents)} recording(s) with MouthAction tiers")
    if not documents:
        print("Nothing to do. This corpus subset has no MouthAction tiers.")
        write_csv(Path(args.out) / TABLES_SUBFOLDER / "mouth_coverage.csv",
                  coverage_summary(elan_index, pd.DataFrame()))
        return 0

    labelled = annotate_with_mouth(annotations, documents, args.min_overlap_ms)
    print(f"{len(labelled):,} annotation(s) lie in those recordings")

    labels = label_table(documents, args.min_overlap_ms)
    print(f"{len(labels):,} MouthAction label(s) across those recordings")

    folder = Path(args.out) / TABLES_SUBFOLDER
    overlap = overlap_table_by_region(labelled)
    write_csv(folder / "mouth_overlap.csv", overlap)
    write_csv(folder / "mouth_coverage.csv", coverage_summary(elan_index, labelled))
    write_csv(folder / "mouth_categories.csv", category_counts_by_region(labels))

    key_categories = pd.concat(
        [key_category_table(labelled, labels, "GLOBAL")]
        + [key_category_table(group, labels, str(code))
           for code, group in labelled.groupby("region_code", sort=True)],
        ignore_index=True)
    write_csv(folder / "mouth_key_categories.csv", key_categories)

    disagreements = disagreement_detail(labelled)
    if not disagreements.empty:
        write_csv(Path(args.out) / DIAGNOSTICS_SUBFOLDER / "mouth_disagreements.csv",
                  disagreements)

    if not args.no_latex and not overlap.empty:
        write_report_tables({"mouth": overlap}, folder)

    if args.save_labelled and not labelled.empty:
        write_csv(Path(args.out) / DIAGNOSTICS_SUBFOLDER / "annotations_with_mouth.csv",
                  labelled)

    if not overlap.empty:
        overall = overlap[(overlap["tag"] == "GLOBAL")
                          & (overlap["unit"] == "any annotation")]
        if not overall.empty:
            row = overall.iloc[0]
            print(f"\n{row['percent_with_mouth']}% of those annotations overlap a "
                  f"mouth label")
            print(f"  Mouthing {row['percent_Mouthing']}%, "
                  f"MouthGesture {row['percent_MouthGesture']}%, "
                  f"Others {row['percent_Others']}%")
            if row["n_multi_label"]:
                print(f"  {int(row['n_multi_label']):,} annotation(s) overlap more "
                      f"than one label: {int(row['n_agree']):,} agree, "
                      f"{int(row['n_disagree']):,} disagree "
                      f"({row['percent_agreement']}% agreement)")

    print(f"\nWrote mouth tables to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
