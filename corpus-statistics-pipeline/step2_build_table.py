"""Step 2: build one annotation table from the parsed CSVs.

    python3 step2_build_table.py --annotations /path/to/parsed --out output

Reads every parsed CSV, tags each row with its source file and prefecture, adds
the classification flags, and writes a single table. Everything after this
stage works on that one file, which is what keeps the global and per-region
numbers from drifting apart.

The stage also reports how many parsed files it could match to an ELAN
document. An unmatched file is not fatal, since the annotation counts are still
correct, but its recording duration is missing from every rate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from config import (
    AMBIGUOUS_COLUMN,
    ANNOTATIONS_FILE,
    ANNOTATION_COLUMN,
    COMPOUND_COLUMN,
    DEFAULT_OUTPUT_FOLDER,
    ELAN_INDEX_FILE,
    KEY_COLUMNS,
    LEXICAL_COLUMN,
    SIGNER_COLUMN,
)
from io_utils import file_key, find_files, read_csv_safely, region_of, write_csv
from metrics import add_flags

EXPECTED_COLUMNS = ([SIGNER_COLUMN, "time_start", "time_end", ANNOTATION_COLUMN,
                     LEXICAL_COLUMN] + KEY_COLUMNS
                    + [COMPOUND_COLUMN, AMBIGUOUS_COLUMN])


def load_parsed(annotations_folder: Path, verbose: bool = True) -> pd.DataFrame:
    paths = find_files(annotations_folder, ".csv")
    # A previous run's outputs may sit in the same tree; they are not annotations.
    paths = [p for p in paths if not p.name.startswith(("tab_", "summary_", "top_"))]
    if verbose:
        print(f"Found {len(paths)} parsed CSV file(s) under {annotations_folder}")

    frames: List[pd.DataFrame] = []
    missing_columns = set()
    for path in paths:
        try:
            frame = read_csv_safely(path)
        except Exception as error:                      # noqa: BLE001 - reported, not raised
            print(f"  skipped {path.name}: {error}", file=sys.stderr)
            continue
        if frame.empty:
            continue
        if ANNOTATION_COLUMN not in frame.columns and LEXICAL_COLUMN not in frame.columns:
            print(f"  skipped {path.name}: no '{ANNOTATION_COLUMN}' or "
                  f"'{LEXICAL_COLUMN}' column", file=sys.stderr)
            continue

        missing_columns |= {c for c in EXPECTED_COLUMNS if c not in frame.columns}
        for column in EXPECTED_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""

        frame["source_file"] = path.name
        frame["file_key"] = file_key(path.name)
        frame["region_code"] = region_of(path.name)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=EXPECTED_COLUMNS
                            + ["source_file", "file_key", "region_code"])

    if verbose and missing_columns:
        print(f"  note: {len(missing_columns)} expected column(s) absent from at "
              f"least one file and filled as empty: "
              f"{', '.join(sorted(missing_columns))}")

    return pd.concat(frames, ignore_index=True)


def report_matching(annotations: pd.DataFrame, elan_index: pd.DataFrame) -> None:
    """Print how many parsed files found their .eaf, and name the ones that did not."""
    if elan_index is None or elan_index.empty:
        print("No ELAN index available: recording durations and rates will be blank.")
        return

    elan_keys = set(elan_index["file_key"])
    parsed_keys = set(annotations["file_key"])
    matched = parsed_keys & elan_keys
    print(f"{len(matched)}/{len(parsed_keys)} parsed file(s) matched an .eaf document")

    unmatched = sorted(parsed_keys - elan_keys)
    if unmatched:
        print(f"  {len(unmatched)} parsed file(s) with no .eaf: "
              f"{', '.join(unmatched[:8])}"
              + (" ..." if len(unmatched) > 8 else ""))
    unused = sorted(elan_keys - parsed_keys)
    if unused:
        print(f"  {len(unused)} .eaf file(s) with no parsed CSV: "
              f"{', '.join(unused[:8])}" + (" ..." if len(unused) > 8 else ""))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True, type=Path,
                        help="folder of parsed annotation CSVs")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    annotations = load_parsed(args.annotations, verbose=not args.quiet)
    if annotations.empty:
        print(f"No usable parsed CSVs under {args.annotations}", file=sys.stderr)
        return 1

    annotations = add_flags(annotations)

    index_path = Path(args.out) / ELAN_INDEX_FILE
    elan_index = read_csv_safely(index_path) if index_path.exists() else None
    report_matching(annotations, elan_index)

    path = write_csv(Path(args.out) / ANNOTATIONS_FILE, annotations)
    print(f"\n{len(annotations):,} annotations from "
          f"{annotations['source_file'].nunique()} file(s), "
          f"{annotations[SIGNER_COLUMN].nunique()} signer(s)")
    print(f"{int(annotations['is_parsed'].sum()):,} successfully parsed "
          f"({100 * annotations['is_parsed'].mean():.1f}%)")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
