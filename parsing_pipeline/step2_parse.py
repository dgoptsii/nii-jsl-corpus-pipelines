#!/usr/bin/env python3
"""STEP 2 (optional, standalone): parse extracted annotation CSVs.

You only need this if you are running the steps separately. The normal path is
run_pipeline.py, which does all three steps in one go.

    python step2_parse.py EXTRACTED_FOLDER -output_folder PARSED_FOLDER [options]

Example
-------
    python step2_parse.py ./extracted -output_folder ./parsed
    python step2_parse.py ./extracted -output_folder ./parsed --exceptions-file input_lists/exceptions.txt

Input
-----
    EXTRACTED_FOLDER/*_word_annotations.csv   (from step1_extract.py)

Output
------
    PARSED_FOLDER/parsed/<name>_parsed.csv            every row
    PARSED_FOLDER/ambiguous/<name>_ambiguous_rows.csv only rows needing review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import parsing
from config import FIELDNAMES
from locate_elan_files import strip_stage_suffix
from io_utils import read_csv_rows, write_csv_rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="step2_parse.py",
        description="Parse *_word_annotations.csv files into structured columns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python step2_parse.py ./extracted -output_folder ./parsed --exceptions-file input_lists/exceptions.txt\n"
        ),
    )
    parser.add_argument("input_folder", type=Path,
                        help="Folder containing *_word_annotations.csv files.")
    parser.add_argument("-output_folder", "--output_folder", "--output-folder",
                        dest="output_folder", type=Path,
                        default=Path("parsed_annotations"),
                        help="Folder for the parsed/ and ambiguous/ subfolders.")
    parser.add_argument("--exceptions_file", "--exceptions-file",
                        dest="exceptions_file", type=Path, default=None,
                        metavar="EXCEPTIONS.txt",
                        help="Text file of annotations that must never be marked ambiguous.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    input_folder = args.input_folder.expanduser().resolve()
    output_folder = args.output_folder.expanduser().resolve()
    parsed_folder = output_folder / "parsed"
    ambiguous_folder = output_folder / "ambiguous"

    if not input_folder.is_dir():
        print(f"ERROR: input folder does not exist: {input_folder}", file=sys.stderr)
        return 1

    exceptions_file = (
        args.exceptions_file.expanduser().resolve() if args.exceptions_file else None
    )
    parsing.load_exceptions(exceptions_file)

    inputs = sorted(input_folder.glob("*_word_annotations.csv"))
    if not inputs:
        print(f"No *_word_annotations.csv files found in {input_folder}")
        return 1

    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Exceptions:    {exceptions_file or '(none)'}")
    print(f"Files:         {len(inputs)}")
    print()

    total_rows = 0
    total_ambiguous = 0

    for path in inputs:
        rows = read_csv_rows(path)
        parsed = parsing.parse_rows(rows)
        ambiguous = parsing.select_ambiguous_rows(parsed)
        statistics = parsing.parse_statistics(parsed)

        stem = strip_stage_suffix(path.stem)
        write_csv_rows(parsed_folder / f"{stem}_parsed.csv", parsed, FIELDNAMES)
        write_csv_rows(ambiguous_folder / f"{stem}_ambiguous_rows.csv", ambiguous, FIELDNAMES)

        total_rows += statistics["total_rows"]
        total_ambiguous += statistics["ambiguous_rows"]

        print(
            f"{stem}: total={statistics['total_rows']} "
            f"parsed={statistics['parsed_rows']} "
            f"ambiguous={statistics['ambiguous_rows']} "
            f"({statistics['parsed_percentage']:.1f}% resolved)"
        )

    resolved = ((total_rows - total_ambiguous) / total_rows * 100) if total_rows else 0.0
    print(f"\nDone. {total_rows} rows, {total_ambiguous} ambiguous ({resolved:.2f}% resolved).")
    print(f"Parsed CSVs:      {parsed_folder}")
    print(f"Review this one:  {ambiguous_folder}")
    print(f"Next step:  python step3_build_elan.py CORPUS_FOLDER -parsed_csv_folder {parsed_folder} -output_folder parsed_elan_files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
