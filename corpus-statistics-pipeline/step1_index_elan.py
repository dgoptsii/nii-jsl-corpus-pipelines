"""Step 1: index the ELAN files.

Walks the corpus folder for ``.eaf`` documents and records what each one can
contribute: how long it is, which tiers it has, whether those tiers include
MouthAction, and which participants appear in it.

This is the only stage that touches the original corpus. Everything downstream
reads the index, so a slow recursive walk over a large corpus happens once.

    python3 step1_index_elan.py --corpus /path/to/corpus --out output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from typing import List, Optional

from config import DEFAULT_OUTPUT_FOLDER, ELAN_INDEX_FILE
from elan import read_elan
from io_utils import (
    file_key,
    find_files,
    format_hms,
    read_csv_safely,
    region_of,
    write_csv,
)


def read_manifest(path: Path, verbose: bool = True) -> List[Path]:
    """The .eaf files the parser selected, from ``selected_elan_files.csv``.

    Reading the parser's own selection is the point: the corpus tree holds the
    same recording in several places, and if this pipeline picked its own copy
    the statistics would describe a different corpus from the annotations they
    are computed over. A plain text file of one path per line is accepted too.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"ELAN list not found: {path}")

    if path.suffix.lower() == ".csv":
        frame = read_csv_safely(path)
        if "path" not in frame.columns:
            raise ValueError(f"{path} has no 'path' column. Expected the "
                             f"parser's selected_elan_files.csv.")
        entries = [str(value).strip() for value in frame["path"]]
    else:
        entries = [line.split("#", 1)[0].strip()
                   for line in path.read_text(encoding="utf-8").splitlines()]

    paths, missing = [], []
    for entry in entries:
        if not entry:
            continue
        candidate = Path(entry).expanduser()
        (paths if candidate.exists() else missing).append(candidate)

    if verbose:
        print(f"Read {len(paths)} .eaf path(s) from {path}")
    if missing:
        print(f"WARNING: {len(missing)} listed file(s) do not exist:")
        for candidate in missing[:5]:
            print(f"  {candidate}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
        print("The list was probably written on another machine, or the corpus "
              "moved. Rerun the parser to refresh it.")
    return paths


def build_index(corpus_folder: Optional[Path] = None, verbose: bool = True,
                elan_list: Optional[Path] = None) -> pd.DataFrame:
    if elan_list is not None:
        paths = read_manifest(elan_list, verbose)
    else:
        paths = find_files(corpus_folder, ".eaf")
        if verbose:
            print(f"Found {len(paths)} .eaf files under {corpus_folder}")
            print("NOTE: every copy of a duplicated recording is indexed. Pass "
                  "--elan-list <parser output>/selected_elan_files.csv to count "
                  "each recording once.")

    rows = []
    failures = []
    for index, path in enumerate(paths, start=1):
        try:
            document = read_elan(path)
        except ValueError as error:
            failures.append((path, str(error)))
            continue

        rows.append({
            "source_file": path.name,
            "file_key": file_key(path.name),
            "region_code": region_of(path.name),
            "path": str(path),
            "duration_ms": round(document.duration_ms, 1),
            "duration_hms": format_hms(document.duration_ms),
            "n_tiers": len(document.tier_ids),
            "has_mouth_tiers": bool(document.mouth_tier_ids),
            "n_mouth_tiers": len(document.mouth_tier_ids),
            "n_mouth_annotations": len(document.mouth_annotations),
            "n_word_annotations": len(document.word_annotations),
            "participants": ";".join(sorted(document.participants)),
            "mouth_tier_ids": ";".join(sorted(document.mouth_tier_ids)),
        })

        if verbose and index % 50 == 0:
            print(f"  read {index}/{len(paths)}")

    if verbose and failures:
        print(f"\n{len(failures)} file(s) could not be read:")
        for path, error in failures[:10]:
            print(f"  {path.name}: {error}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=None,
                        help="folder searched recursively for .eaf files")
    parser.add_argument("--elan-list", "--elan_list", dest="elan_list",
                        type=Path, default=None, metavar="SELECTED.csv",
                        help="the parser's selected_elan_files.csv: index "
                             "exactly the files it chose, one per recording. "
                             "Preferred over --corpus, which indexes every copy "
                             "of a duplicated recording.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.elan_list is None and args.corpus is None:
        print("Give either --elan-list (preferred) or --corpus.", file=sys.stderr)
        return 1

    try:
        index = build_index(args.corpus, verbose=not args.quiet,
                            elan_list=args.elan_list)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if index.empty:
        source = args.elan_list or args.corpus
        print(f"No readable .eaf files from {source}", file=sys.stderr)
        return 1

    duplicated = index["file_key"].duplicated(keep=False)
    if duplicated.any():
        print(f"\nWARNING: {int(duplicated.sum())} indexed file(s) share a "
              f"recording name, so durations and signer counts are double "
              f"counted:")
        for name in sorted(index.loc[duplicated, "source_file"].unique())[:8]:
            print(f"  {name}")
        print("Use --elan-list to index the parser's selection instead.")

    path = write_csv(Path(args.out) / ELAN_INDEX_FILE, index)

    total = pd.to_numeric(index["duration_ms"], errors="coerce").fillna(0).sum()
    with_mouth = int(index["has_mouth_tiers"].sum())
    print(f"\n{len(index)} files indexed, {format_hms(total)} of recording")
    print(f"{with_mouth} file(s) have MouthAction tiers "
          f"({100 * with_mouth / len(index):.1f}%)")
    print(f"Regions: {', '.join(sorted(index['region_code'].unique()))}")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
