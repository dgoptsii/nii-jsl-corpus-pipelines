#!/usr/bin/env python3
"""Run the whole ELAN annotation parsing pipeline.

    python run_pipeline.py CORPUS_FOLDER -output_folder OUTPUT_FOLDER [options]

What it does, for every .eaf file it finds:

    1. extract the Word-jp annotations           (extract.py)
    2. parse them into structured columns        (parsing.py)
    3. write a new .eaf file that keeps every    (elan_builder.py)
       original tier and adds the parsed
       columns as child tiers

Examples
--------
    # everything under ./corpus, recursively
    python run_pipeline.py ./corpus -output_folder ./output

    # only the .eaf files sitting directly in ./corpus
    python run_pipeline.py ./corpus -output_folder ./output --flat

    # only the files listed in a text file
    python run_pipeline.py ./corpus -output_folder ./output --file-list input_lists/files_of_interest.txt

    # also keep the intermediate CSVs for checking
    python run_pipeline.py ./corpus -output_folder ./output --save-debug

    # show which files would be processed, then stop
    python run_pipeline.py ./corpus --list-only

Run with --help for the full list of options.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import parsing
from config import EXTRACTED_FIELDNAMES, FIELDNAMES, PipelineConfig
from locate_elan_files import (
    find_elan_files,
    write_manifest,
)
from elan_builder import BuildResult, build_parsed_eaf, write_tier_report
from extract import extract_rows
from io_utils import write_csv_rows

LINE = "=" * 78


# ===========================================================================
# Results
# ===========================================================================

@dataclass
class FileResult:
    """What happened to a single ELAN file."""

    file_stem: str
    input_eaf: Path
    output_eaf: Optional[Path] = None
    extracted_rows: int = 0
    parsed_rows: int = 0
    ambiguous_rows: int = 0
    created_tiers: int = 0
    created_annotations: int = 0
    unmatched_values: int = 0
    speakers: List[str] = field(default_factory=list)
    missing_parent_speakers: List[str] = field(default_factory=list)
    #: (signer, tier_id, n_annotations) for each extra Word tier not read.
    extra_word_tiers: List[Tuple[str, str, int]] = field(default_factory=list)
    debug_files: Dict[str, Path] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def parsed_percentage(self) -> float:
        total = self.parsed_rows + self.ambiguous_rows
        return (self.parsed_rows / total * 100) if total else 0.0


@dataclass
class PipelineResult:
    """Aggregate result of a whole run."""

    config: PipelineConfig
    discovery: object
    files: List[FileResult] = field(default_factory=list)

    @property
    def succeeded(self) -> List[FileResult]:
        return [item for item in self.files if item.ok]

    @property
    def failed(self) -> List[FileResult]:
        return [item for item in self.files if not item.ok]

    @property
    def total_annotations(self) -> int:
        return sum(item.parsed_rows + item.ambiguous_rows for item in self.files)

    @property
    def total_ambiguous(self) -> int:
        return sum(item.ambiguous_rows for item in self.files)

    @property
    def parsed_percentage(self) -> float:
        total = self.total_annotations
        return ((total - self.total_ambiguous) / total * 100) if total else 0.0


# ===========================================================================
# The pipeline itself
# ===========================================================================

def process_one_file(
    eaf_path: Path,
    config: PipelineConfig,
    columns: Optional[Sequence[str]] = None,
) -> FileResult:
    """Run all three stages for a single ELAN file."""
    file_stem = eaf_path.stem
    result = FileResult(file_stem=file_stem, input_eaf=eaf_path)

    try:
        # --- 1. extract -----------------------------------------------------
        discarded: List[Tuple[str, str, int]] = []
        extracted = extract_rows(eaf_path, discarded=discarded)
        result.extracted_rows = len(extracted)
        result.extra_word_tiers = discarded

        if config.save_debug:
            result.debug_files["extracted"] = write_csv_rows(
                config.extracted_debug_folder / f"{file_stem}_word_annotations.csv",
                extracted,
                EXTRACTED_FIELDNAMES,
            )

        # --- 2. parse -------------------------------------------------------
        parsed = parsing.parse_rows(extracted)
        ambiguous = parsing.select_ambiguous_rows(parsed)
        result.ambiguous_rows = len(ambiguous)
        result.parsed_rows = len(parsed) - len(ambiguous)

        if config.save_debug:
            result.debug_files["parsed"] = write_csv_rows(
                config.parsed_debug_folder / f"{file_stem}_parsed.csv",
                parsed,
                FIELDNAMES,
            )
            result.debug_files["ambiguous"] = write_csv_rows(
                config.ambiguous_debug_folder / f"{file_stem}_ambiguous_rows.csv",
                ambiguous,
                FIELDNAMES,
            )

        # --- 3. write the new ELAN file -------------------------------------
        output_eaf = config.eaf_output_folder / f"{file_stem}_parsed.eaf"

        if output_eaf.exists() and not config.overwrite:
            result.error = f"output already exists (remove --no-overwrite): {output_eaf}"
            return result

        build: BuildResult = build_parsed_eaf(
            input_eaf=eaf_path,
            parsed_rows=parsed,
            output_eaf=output_eaf,
            columns=columns,
        )

        result.output_eaf = build.output_eaf
        result.created_tiers = build.created_tier_count
        result.created_annotations = build.created_annotation_count
        result.unmatched_values = len(build.unmatched_rows)
        result.speakers = build.speakers
        result.missing_parent_speakers = build.missing_parent_speakers

        if config.save_debug:
            result.debug_files["tier_report"] = write_tier_report(
                config.tier_report_folder / f"{file_stem}_tier_report.txt",
                build,
                parsed_csv=result.debug_files.get("parsed"),
            )

    except Exception as error:  # noqa: BLE001 - one bad file must not stop the run
        result.error = f"{type(error).__name__}: {error}"

    return result


def run_pipeline(
    config: PipelineConfig,
    columns: Optional[Sequence[str]] = None,
    progress=None,
) -> PipelineResult:
    """Run the pipeline over a folder of ELAN files.

    Importable, so you can call this from a notebook:

        from run_pipeline import run_pipeline
        from config import PipelineConfig
        result = run_pipeline(PipelineConfig(elan_folder=..., output_folder=...))
    """
    parsing.load_exceptions(config.exceptions_file)

    discovery = find_elan_files(
        folder=config.elan_folder,
        recursive=config.recursive,
        file_list=config.file_list,
        regions=config.regions,
    )

    result = PipelineResult(config=config, discovery=discovery)
    total = len(discovery.files)

    for index, eaf_path in enumerate(discovery.files, start=1):
        file_result = process_one_file(eaf_path, config, columns=columns)
        result.files.append(file_result)

        if progress is not None:
            progress(index, total, file_result)

    return result


# ===========================================================================
# Command line
# ===========================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "Parse JSL Word-tier annotations in ELAN files and write new ELAN "
            "files that keep every original tier and add the parsed columns as "
            "child tiers."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_pipeline.py ./corpus -output_folder ./output\n"
            "  python run_pipeline.py ./corpus -output_folder ./output --flat\n"
            "  python run_pipeline.py ./corpus -output_folder ./output --file-list input_lists/files_of_interest.txt\n"
            "  python run_pipeline.py ./corpus -output_folder ./output --save-debug\n"
            "  python run_pipeline.py ./corpus --list-only\n"
        ),
    )

    parser.add_argument(
        "elan_folder",
        type=Path,
        help="Folder containing the ELAN (.eaf) files.",
    )
    parser.add_argument(
        "-output_folder", "--output_folder", "--output-folder",
        dest="output_folder",
        type=Path,
        default=Path("pipeline_output"),
        help="Where results are written. Default: pipeline_output",
    )

    parser.add_argument(
        "--flat", "--no-recursive",
        dest="recursive",
        action="store_false",
        default=True,
        help="Read only the .eaf files sitting directly in the folder. "
             "Default is to search the whole folder tree.",
    )
    parser.add_argument(
        "-file_list", "--file_list", "--file-list",
        dest="file_list",
        type=Path,
        default=None,
        metavar="FILES.txt",
        help="Text file listing the files to process, one per line. Bare stems, "
             "file names and full paths all work; '#' starts a comment.",
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        default=[],
        metavar="XX",
        help="Only process files whose name starts with these region prefixes, "
             "e.g. --regions FO NS IS. Ignored when --file-list is given.",
    )

    parser.add_argument(
        "--save_debug", "--save-debug",
        dest="save_debug",
        action="store_true",
        help="Also write the intermediate files: extracted word annotations, "
             "parsed annotations, ambiguous annotations and tier reports.",
    )
    parser.add_argument(
        "--exceptions_file", "--exceptions-file",
        dest="exceptions_file",
        type=Path,
        default=None,
        metavar="EXCEPTIONS.txt",
        help="Text file of annotations that must never be marked ambiguous.",
    )
    parser.add_argument(
        "--no_overwrite", "--no-overwrite",
        dest="overwrite",
        action="store_false",
        default=True,
        help="Fail instead of overwriting an existing output .eaf file.",
    )
    parser.add_argument(
        "--list_only", "--list-only",
        dest="list_only",
        action="store_true",
        help="Print which files would be processed, then stop without doing anything.",
    )

    return parser


def print_selection(discovery, config: PipelineConfig) -> None:
    print(LINE)
    print(f"ELAN folder:   {config.elan_folder}")
    print(f"Search mode:   {'recursive' if config.recursive else 'flat (top level only)'}")
    print(f"File list:     {config.file_list or '(none, all files)'}")
    print(f"Regions:       {', '.join(config.regions) if config.regions else 'ALL'}")
    print(f"Exceptions:    {config.exceptions_file or '(none)'}")
    print(f"Output folder: {config.output_folder}")
    print(f"Debug output:  {'yes' if config.save_debug else 'no (use --save-debug)'}")
    print(f"ELAN files:    {len(discovery.files)}")

    if discovery.missing_stems:
        print(f"\nWARNING: {len(discovery.missing_stems)} requested file(s) not found:")
        for stem in discovery.missing_stems:
            print(f"  {stem}")

    if discovery.duplicate_stems:
        print(f"\nERROR: {len(discovery.duplicate_stems)} name(s) exist in more "
              f"than one place ({discovery.n_duplicate_copies} extra copy(ies)).",
              file=sys.stderr)
        print("Every stage names its output after the file stem, so two files "
              "with one name would overwrite each other.", file=sys.stderr)
        print("The pipeline no longer picks a copy for you. Remove the extra "
              "copies from the ELAN folder, or point --elan-folder at a folder "
              "that holds exactly one file per recording, and run again.",
              file=sys.stderr)
        for stem in sorted(discovery.duplicate_stems):
            print(f"  {stem}", file=sys.stderr)
            for path in discovery.duplicate_stems[stem]:
                print(f"    {path}", file=sys.stderr)


def report_file(index: int, total: int, result: FileResult) -> None:
    status = "OK  " if result.ok else "FAIL"
    print(f"[{index}/{total}] {status} {result.file_stem}")

    if not result.ok:
        print(f"          {result.error}")
        return

    print(
        f"          annotations={result.extracted_rows} "
        f"parsed={result.parsed_rows} ambiguous={result.ambiguous_rows} "
        f"({result.parsed_percentage:.1f}% resolved)"
    )
    print(
        f"          new tiers={result.created_tiers} "
        f"new annotations={result.created_annotations} "
        f"unmatched={result.unmatched_values}"
    )

    if result.extra_word_tiers:
        skipped = sum(count for _, _, count in result.extra_word_tiers)
        print(f"          NOTE {len(result.extra_word_tiers)} tier(s) not read "
              f"({skipped} annotations): only <signer>-Word-jp is read, once "
              f"per signer")
        for signer, tier_id, count in result.extra_word_tiers:
            print(f"               {signer}: {tier_id} ({count} annotations)")

    if result.extracted_rows == 0:
        print("          WARNING no Word-jp annotation read from this file. "
              "If the tiers listed above are the only ones it has, its gloss "
              "is on a tier this pipeline does not accept.")

    if result.missing_parent_speakers:
        print(f"          WARNING no Word-jp parent tier for: "
              f"{', '.join(result.missing_parent_speakers)}")


def print_summary(result: PipelineResult) -> None:
    config = result.config

    print()
    print(LINE)
    print("SUMMARY")
    print(LINE)
    print(f"Files processed:   {len(result.succeeded)}")
    print(f"Files failed:      {len(result.failed)}")
    print(f"Annotations read:  {result.total_annotations}")
    print(f"Ambiguous rows:    {result.total_ambiguous}")
    print(f"Resolved:          {result.parsed_percentage:.2f}%")
    print(f"New ELAN files in: {config.eaf_output_folder}")

    if config.save_debug:
        print(f"Debug output in:   {config.debug_folder}")
        print(f"Review this file:  {config.ambiguous_debug_folder}")

    if result.failed:
        print("\nFailed files:")
        for item in result.failed:
            print(f"  {item.file_stem}: {item.error}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    config = PipelineConfig(
        elan_folder=args.elan_folder.expanduser().resolve(),
        output_folder=args.output_folder.expanduser().resolve(),
        recursive=args.recursive,
        file_list=args.file_list.expanduser().resolve() if args.file_list else None,
        exceptions_file=(
            args.exceptions_file.expanduser().resolve() if args.exceptions_file else None
        ),
        regions=list(args.regions or []),
        save_debug=args.save_debug,
        overwrite=args.overwrite,
    )

    try:
        discovery = find_elan_files(
            folder=config.elan_folder,
            recursive=config.recursive,
            file_list=config.file_list,
            regions=config.regions,
        )
    except (FileNotFoundError, NotADirectoryError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print_selection(discovery, config)

    # Duplicate names are a corpus problem, not something a run should paper
    # over by choosing a copy: whichever copy it chose would silently decide
    # what every downstream number describes. Stop and let the folder be fixed.
    if discovery.duplicate_stems:
        return 1

    # Publish the selection before doing any work, so the other pipelines can be
    # pointed at exactly these files and cannot disagree about which copy of a
    # recording counts. Written for --list-only too: refreshing the manifest
    # should not require reparsing the corpus.
    if discovery.files:
        manifest = write_manifest(discovery,
                                  config.output_folder / "selected_elan_files.csv")
        print(f"\nSelection written to {manifest}")
        print("  the corpus-statistics and signing-space runs should read this "
              "rather than walking the corpus.")

    if args.list_only:
        print()
        for path in discovery.files:
            print(path)
        return 0 if discovery.files else 1

    if not discovery.files:
        print("\nNothing to do: no matching .eaf files found.")
        return 1

    print()
    result = run_pipeline(config, progress=report_file)
    print_summary(result)

    return 0 if not result.failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
