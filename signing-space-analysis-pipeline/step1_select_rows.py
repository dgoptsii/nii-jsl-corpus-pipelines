#!/usr/bin/env python3
"""STEP 1: select the parsed annotation rows to analyse.

    python3 step1_select_rows.py PARSED_FOLDER -output_folder OUT \
        --keywords cl fs pt --regions FO GM

Reads the ``*_parsed.csv`` files from the parsing pipeline and, for each
requested keyword column, keeps the rows where that column is non-empty.
``--lexical-only`` keeps only plain lexical rows for the ``lexical_item``
keyword: a lexical item present and every marker column empty.

Output: ``OUT/key_rows/<KEYWORD>/ALL_<KEYWORD>_rows.csv``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from config import DEFAULT_OUTPUT_FOLDER, KEY_ROWS_SUBFOLDER
from io_utils import read_csv_safely, write_csv
from signers import signer_id_from_row

#: Columns treated as parser markers when --lexical-only is used.
MARKER_COLUMNS = [
    "pt", "dw", "fs", "aw", "lh", "rh", "d", "cl", "m", "ges", "nmm",
    "rep", "stop", "hold", "index", "keep", "fal", "un", "qm",
    "past", "neg", "compound", "ambiguous",
]

DEFAULT_KEYWORDS = ["lexical_item", "cl", "fs", "pt"]


def nonempty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def region_of(path: Path) -> str:
    return path.stem.split("_", 1)[0].upper()


def select_rows(frame: pd.DataFrame, keyword: str, lexical_only: bool) -> pd.DataFrame:
    """Rows where ``keyword`` is filled, or plain lexical rows when asked."""
    if keyword not in frame.columns:
        return pd.DataFrame(columns=frame.columns)

    mask = nonempty(frame[keyword])

    if lexical_only and keyword == "lexical_item":
        for column in MARKER_COLUMNS:
            if column in frame.columns:
                mask &= ~nonempty(frame[column])

    return frame[mask].copy()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="step1_select_rows.py",
        description="Select parsed annotation rows for the chosen keyword columns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python3 step1_select_rows.py ./parsed -output_folder ./out --keywords cl fs\n",
    )
    parser.add_argument("parsed_folder", type=Path,
                        help="Folder containing *_parsed.csv files.")
    parser.add_argument("-output_folder", "--output_folder", dest="output_folder",
                        type=Path, default=DEFAULT_OUTPUT_FOLDER,
                        help=f"Where results go. Default: {DEFAULT_OUTPUT_FOLDER}")
    parser.add_argument("--keywords", nargs="+", default=DEFAULT_KEYWORDS,
                        help="Parsed columns to select, e.g. cl fs pt lexical_item.")
    parser.add_argument("--regions", nargs="*", default=[],
                        help="Only these geographical regions, e.g. FO GM. Empty = all.")
    parser.add_argument("--lexical-only", "--lexical_only", dest="lexical_only",
                        action="store_true",
                        help="For lexical_item, keep only rows with no marker columns set.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    parsed_folder = args.parsed_folder.expanduser().resolve()
    out_root = args.output_folder.expanduser().resolve() / KEY_ROWS_SUBFOLDER
    keywords = [k.strip().lower() for k in args.keywords if k.strip()]
    regions = {r.strip().upper() for r in args.regions if r.strip()}

    if not parsed_folder.is_dir():
        print(f"ERROR: parsed folder does not exist: {parsed_folder}", file=sys.stderr)
        return 1

    files = sorted(parsed_folder.glob("*_parsed.csv"))
    if regions:
        files = [path for path in files if region_of(path) in regions]

    if not files:
        print(f"No *_parsed.csv files found in {parsed_folder}"
              + (f" for regions {sorted(regions)}" if regions else ""))
        return 1

    print(f"Parsed folder: {parsed_folder}")
    print(f"Output folder: {out_root}")
    print(f"Keywords:      {', '.join(keywords)}")
    print(f"Regions:       {', '.join(sorted(regions)) if regions else 'ALL'}")
    print(f"Files:         {len(files)}\n")

    collected = {keyword: [] for keyword in keywords}

    for path in files:
        frame = read_csv_safely(path)
        source_file = path.stem.replace("_parsed", "")

        for keyword in keywords:
            selected = select_rows(frame, keyword, args.lexical_only)
            if selected.empty:
                continue
            selected.insert(0, "source_file", source_file)
            selected.insert(1, "region_code", region_of(path))
            selected["signer_id"] = [
                signer_id_from_row(row) for _, row in selected.iterrows()
            ]
            collected[keyword].append(selected)

        print(f"{source_file:32s} " + "  ".join(
            f"{k}={len(select_rows(frame, k, args.lexical_only))}" for k in keywords
        ))

    print()
    total = 0
    for keyword, chunks in collected.items():
        combined = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        target = out_root / keyword.upper() / f"ALL_{keyword.upper()}_rows.csv"
        write_csv(target, combined)
        total += len(combined)
        signers = combined["signer_id"].nunique() if len(combined) else 0
        print(f"{keyword.upper():14s} {len(combined):6d} rows, {signers:3d} signers -> {target}")

    print(f"\nDone. {total} rows selected.")
    print(f"Next:  python3 step2_extract_clips.py {args.output_folder} --video-root /path/to/videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
