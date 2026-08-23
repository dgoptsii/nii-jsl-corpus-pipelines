#!/usr/bin/env python3
"""Recompute the signer metadata columns in an existing clip index.

Stage 2 stamps handedness, age, age group and gender onto every row of
``output/clips/clip_index.csv`` as it cuts the clips, and stage 4 reads that
index. So a correction to ``input_lists/signers.csv`` - or to the lookup in
``signers.py`` - only reaches the analysis once the index is rewritten.

Re-running stage 2 would do it, but stage 2 has no skip logic: it re-cuts every
clip from the source video, which is hours of work to change a text column.
This script recomputes exactly the columns stage 2 derives from the signers
file, using the same code path, and leaves everything else untouched.

    python3 refresh_clip_index.py output --signers-file input_lists/signers.csv
    python3 refresh_clip_index.py output --signers-file input_lists/signers.csv --write

Without --write it reports what would change and exits. The previous index is
kept as clip_index.csv.bak.

``--fix-paths`` additionally rebases ``clip_path`` onto the output folder given
here, for an index written before the project folder was renamed. Only stage 4's
--save-debug reads that column, so it is off by default.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from config import CLIPS_SUBFOLDER, CLIP_INDEX_FILE
from io_utils import read_csv_safely, write_csv
from signers import load_signer_metadata

DERIVED_COLUMNS = ["handedness", "age", "age_group", "gender"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--signers-file", "--signers_file", dest="signers_file",
                        type=Path, required=True)
    parser.add_argument("--write", action="store_true",
                        help="Actually rewrite the index. Off by default.")
    parser.add_argument("--fix-paths", dest="fix_paths", action="store_true",
                        help="Rebase clip_path onto this output folder.")
    args = parser.parse_args(argv)

    index_path = args.output_folder / CLIPS_SUBFOLDER / CLIP_INDEX_FILE
    if not index_path.exists():
        print(f"ERROR: no clip index at {index_path}", file=sys.stderr)
        return 1

    index = read_csv_safely(index_path)
    metadata = load_signer_metadata(args.signers_file)
    print(f"Index:       {index_path}  ({len(index)} rows)")
    print(f"Signers:     {args.signers_file}  ({metadata.n_rows} rows, "
          f"{len(metadata.left_handed_keys)} left-handed)")

    changes = 0
    for column, compute in (
        ("handedness", metadata.handedness),
        ("age", metadata.age),
        ("age_group", metadata.age_group),
        ("gender", metadata.gender),
    ):
        if column not in index.columns:
            continue
        fresh = index["signer_id"].astype(str).map(compute)
        before, after = index[column].astype(str), fresh.astype(str)
        moved = before != after
        if moved.any():
            changes += int(moved.sum())
            print(f"\n  {column}: {int(moved.sum())} rows change")
            for signer, group in index[moved].groupby("signer_id"):
                print(f"    {signer:16s} {before[group.index].iloc[0]:>8s}"
                      f" -> {after[group.index].iloc[0]:<8s} ({len(group)} clips)")
        index[column] = fresh

    if args.fix_paths and "clip_path" in index.columns:
        root = str((args.output_folder / CLIPS_SUBFOLDER).resolve())
        rebased = index["clip_path"].astype(str).map(
            lambda p: str(Path(root) / p.split(f"/{CLIPS_SUBFOLDER}/", 1)[1])
            if f"/{CLIPS_SUBFOLDER}/" in p else p)
        moved = rebased != index["clip_path"].astype(str)
        if moved.any():
            print(f"\n  clip_path: {int(moved.sum())} rows rebased onto {root}")
            changes += int(moved.sum())
        index["clip_path"] = rebased

    if not changes:
        print("\nNothing to change.")
        return 0

    if not args.write:
        print(f"\n{changes} cell(s) would change. Re-run with --write to apply.")
        return 0

    shutil.copy2(index_path, index_path.with_suffix(".csv.bak"))
    write_csv(index_path, index)
    print(f"\nWritten. Previous index kept at {index_path.with_suffix('.csv.bak')}")
    print("Now re-run stage 4 with --overwrite, then stage 5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
