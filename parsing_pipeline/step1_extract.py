#!/usr/bin/env python3
"""STEP 1 (optional, standalone): extract Word-jp annotations into CSV files.

Only needed to inspect or hand-correct the extracted annotations before parsing
them; run_pipeline.py does all three steps in one go.

    python step1_extract.py CORPUS_FOLDER -output_folder EXTRACTED_FOLDER [options]

Output: ``EXTRACTED_FOLDER/<name>_word_annotations.csv``, columns file_id,
tier_id, start_ms, end_ms, annotation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from config import EXTRACTED_FIELDNAMES
from locate_elan_files import find_elan_files
from extract import extract_rows
from io_utils import write_csv_rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="step1_extract.py",
        description="Extract Word-jp annotations from ELAN files into CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python step1_extract.py ./corpus -output_folder ./extracted\n"
        ),
    )
    parser.add_argument("elan_folder", type=Path,
                        help="Folder containing the ELAN (.eaf) files.")
    parser.add_argument("-output_folder", "--output_folder", "--output-folder",
                        dest="output_folder", type=Path,
                        default=Path("extracted_word_annotations"),
                        help="Folder for the *_word_annotations.csv files.")
    parser.add_argument("--flat", "--no-recursive", dest="recursive",
                        action="store_false", default=True,
                        help="Read only the .eaf files directly in the folder.")
    parser.add_argument("-file_list", "--file_list", "--file-list",
                        dest="file_list", type=Path, default=None,
                        metavar="FILES.txt",
                        help="Text file listing the files to process, one per line.")
    parser.add_argument("--regions", nargs="*", default=[], metavar="XX",
                        help="Only process these region prefixes, e.g. --regions FO NS.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    elan_folder = args.elan_folder.expanduser().resolve()
    output_folder = args.output_folder.expanduser().resolve()

    try:
        discovery = find_elan_files(
            folder=elan_folder,
            recursive=args.recursive,
            file_list=args.file_list.expanduser().resolve() if args.file_list else None,
            regions=list(args.regions or []),
        )
    except (FileNotFoundError, NotADirectoryError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"ELAN folder:   {elan_folder}")
    print(f"Search mode:   {'recursive' if args.recursive else 'flat (top level only)'}")
    print(f"Output folder: {output_folder}")
    print(f"ELAN files:    {len(discovery.files)}")

    for stem in discovery.missing_stems:
        print(f"WARNING: requested file not found: {stem}")

    if not discovery.files:
        print("\nNothing to do: no matching .eaf files found.")
        return 1

    print()
    total = 0
    for path in discovery.files:
        rows = extract_rows(path)
        target = write_csv_rows(
            output_folder / f"{path.stem}_word_annotations.csv",
            rows,
            EXTRACTED_FIELDNAMES,
        )
        total += len(rows)
        print(f"{path.stem}: {len(rows)} annotations -> {target}")

    print(f"\nDone. {len(discovery.files)} file(s), {total} annotations.")
    print(f"Next step:  python step2_parse.py {output_folder} -output_folder parsed_annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
