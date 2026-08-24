#!/usr/bin/env python3
"""STEP 3 (optional, standalone): write new ELAN files from parsed CSVs.

Only needed when running the steps separately, typically after hand-correcting
the parsed CSVs from step 2; run_pipeline.py does all three in one go.

    python step3_build_elan.py CORPUS_FOLDER -parsed_csv_folder ./parsed/parsed \
        -output_folder ./parsed_elan_files

Reads the original ``CORPUS_FOLDER/**/*.eaf`` and ``*_parsed.csv`` from step 2.
Writes ``OUTPUT_FOLDER/<name>_parsed.eaf`` (original tiers plus parsed child
tiers) and, with ``--save-debug``, ``OUTPUT_FOLDER/tier_reports/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from locate_elan_files import find_elan_files, strip_stage_suffix
from elan_builder import build_parsed_eaf, write_tier_report
from io_utils import read_csv_rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="step3_build_elan.py",
        description=(
            "Combine original ELAN files with parsed CSVs, writing new .eaf files "
            "that keep every original tier and add the parsed columns as child tiers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python step3_build_elan.py ./corpus -parsed_csv_folder ./parsed/parsed -output_folder ./parsed_elan_files\n"
        ),
    )
    parser.add_argument("elan_folder", type=Path,
                        help="Folder containing the original ELAN (.eaf) files.")
    parser.add_argument("-parsed_csv_folder", "--parsed_csv_folder", "--parsed-csv-folder",
                        dest="parsed_csv_folder", type=Path, required=True,
                        help="Folder containing *_parsed.csv files.")
    parser.add_argument("-output_folder", "--output_folder", "--output-folder",
                        dest="output_folder", type=Path,
                        default=Path("parsed_elan_files"),
                        help="Folder for the new .eaf files.")
    parser.add_argument("--flat", "--no-recursive", dest="recursive",
                        action="store_false", default=True,
                        help="Read only the .eaf files directly in the folder.")
    parser.add_argument("-file_list", "--file_list", "--file-list",
                        dest="file_list", type=Path, default=None,
                        metavar="FILES.txt",
                        help="Text file listing the files to process, one per line.")
    parser.add_argument("--save_debug", "--save-debug", dest="save_debug",
                        action="store_true",
                        help="Also write a tier report per file.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    elan_folder = args.elan_folder.expanduser().resolve()
    parsed_csv_folder = args.parsed_csv_folder.expanduser().resolve()
    output_folder = args.output_folder.expanduser().resolve()

    try:
        discovery = find_elan_files(
            folder=elan_folder,
            recursive=args.recursive,
            file_list=args.file_list.expanduser().resolve() if args.file_list else None,
        )
    except (FileNotFoundError, NotADirectoryError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    by_stem = {path.stem.casefold(): path for path in discovery.files}

    csv_paths = sorted(parsed_csv_folder.glob("*_parsed.csv"))
    if not csv_paths:
        print(f"No *_parsed.csv files found in {parsed_csv_folder}")
        return 1

    print(f"ELAN folder:      {elan_folder}")
    print(f"Parsed CSVs:      {parsed_csv_folder}")
    print(f"Output folder:    {output_folder}")
    print(f"Parsed CSV files: {len(csv_paths)}")
    print()

    failures = 0
    for csv_path in csv_paths:
        stem = strip_stage_suffix(csv_path.stem)
        eaf_path = by_stem.get(stem.casefold())

        if eaf_path is None:
            print(f"SKIP {stem}: no matching .eaf file found in {elan_folder}")
            failures += 1
            continue

        rows = read_csv_rows(csv_path)
        build = build_parsed_eaf(
            input_eaf=eaf_path,
            parsed_rows=rows,
            output_eaf=output_folder / f"{stem}_parsed.eaf",
        )

        if args.save_debug:
            write_tier_report(
                output_folder / "tier_reports" / f"{stem}_tier_report.txt",
                build,
                parsed_csv=csv_path,
            )

        print(
            f"{stem}: {build.created_tier_count} new tiers, "
            f"{build.created_annotation_count} new annotations -> {build.output_eaf}"
        )

    print(f"\nDone. Output folder: {output_folder}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
