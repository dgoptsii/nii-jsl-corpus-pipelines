#!/usr/bin/env python3
"""STEP 4 - classify landmarks into signing-space regions, frame by frame.

Normalises on the shoulders, undoes the signer's yaw, mirrors the space for
left-handed signers, and counts how many stored hand points fall into each
region. Columns are named by hand ROLE (``dominant`` / ``non_dominant``) so
left- and right-handers pool correctly.

    python3 step4_region_counts.py OUTPUT_FOLDER [options]

Output
------
    OUT/region_counts/<CLIP_ID>/region_counts.csv

With ``--save-debug-video`` (off by default), also:

    OUT/debug/<CLIP_ID>/signing_space.mp4   the clip beside the normalised
                                            space, every counted point coloured
                                            by the region it was assigned to
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from config import (
    CLIPS_SUBFOLDER,
    CLIP_INDEX_FILE,
    DEBUG_SUBFOLDER,
    DEBUG_VIDEO_FILE,
    DEFAULT_DEBUG_LIMIT,
    LANDMARKS_FILE,
    LANDMARKS_SUBFOLDER,
    REGION_COUNTS_FILE,
    REGION_COUNTS_SUBFOLDER,
)
from exclusions import filter_index, load_exclusions
from geometry import build_yaw_series
from io_utils import count_or_all, read_csv_safely, write_csv
from landmarks import load_landmarks
from region_counts import compute_region_counts


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="step4_region_counts.py",
        description="Classify stored landmarks into signing-space regions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python3 step4_region_counts.py ./out\n",
    )
    parser.add_argument("output_folder", type=Path,
                        help="The folder step 3 wrote landmarks into.")
    parser.add_argument("--keywords", nargs="*", default=[])
    parser.add_argument("--regions", nargs="*", default=[])
    parser.add_argument("--no-yaw", dest="yaw", action="store_false", default=True,
                        help="Skip yaw correction; use plain shoulder normalisation.")
    parser.add_argument("--exclude-file", "--exclude_file", dest="exclude_file",
                        type=Path, action="append", default=None,
                        metavar="EXCLUDED.txt",
                        help="Text file of clip names to skip, one per line.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-debug-video", dest="debug_video", action="store_true",
                        help="Render an example video per clip: the crop with its "
                             "landmarks beside the normalised signing space, each "
                             "counted point coloured by its region. Slow; "
                             "diagnostic only.")
    parser.add_argument("--debug-limit", "--debug_limit", dest="debug_limit",
                        type=count_or_all, default=DEFAULT_DEBUG_LIMIT,
                        metavar="N|all",
                        help=f"How many clips get a debug video: a number, or "
                             f"'all'. Default {DEFAULT_DEBUG_LIMIT}.")
    parser.add_argument("--debug-max-frames", "--debug_max_frames",
                        dest="debug_max_frames", type=count_or_all, default=0,
                        metavar="N|all",
                        help="Frames rendered per debug video: a number, or "
                             "'all' for the whole clip (the default).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    output_folder = args.output_folder.expanduser().resolve()
    index_path = output_folder / CLIPS_SUBFOLDER / CLIP_INDEX_FILE
    landmarks_root = output_folder / LANDMARKS_SUBFOLDER
    counts_root = output_folder / REGION_COUNTS_SUBFOLDER

    if not index_path.exists():
        print(f"ERROR: no clip index at {index_path}. Run step 2 first.", file=sys.stderr)
        return 1

    index = read_csv_safely(index_path)
    if args.keywords:
        index = index[index["keyword"].str.upper().isin({k.upper() for k in args.keywords})]
    if args.regions:
        index = index[index["region_code"].str.upper().isin({r.upper() for r in args.regions})]

    exclusions = load_exclusions(args.exclude_file)
    index, n_excluded = filter_index(index, exclusions)

    print(f"Clips in index:  {len(index)}")
    if exclusions:
        print(f"{exclusions.describe()}  ({n_excluded} clips dropped)")
    print(f"Counts folder:   {counts_root}")
    print(f"Yaw correction:  {args.yaw}")
    if args.debug_video:
        print(f"Debug videos:    {output_folder / DEBUG_SUBFOLDER}"
              f"  ({args.debug_limit or 'all'} of {len(index)} clips)")
    print()

    ok = missing = skipped = failed = 0
    yaw_fallbacks = debug_written = 0

    for position, (_, row) in enumerate(index.iterrows(), start=1):
        clip_id = str(row["clip_id"])
        npz_path = landmarks_root / clip_id / LANDMARKS_FILE
        target = counts_root / clip_id / REGION_COUNTS_FILE

        if not npz_path.exists():
            missing += 1
            continue

        # A debug render is still wanted for a clip whose counts already exist -
        # asking for the video is usually the second run, not the first.
        wants_debug = (
            args.debug_video
            and (not args.debug_limit or debug_written < args.debug_limit)
        )
        counts_done = target.exists() and not args.overwrite
        if counts_done and not wants_debug:
            skipped += 1
            continue

        try:
            clip, meta = load_landmarks(npz_path)
            side = str(row.get("side", meta.get("side", "left")))
            handedness = str(row.get("handedness", meta.get("handedness", "right")))

            if counts_done:
                skipped += 1
            else:
                frame = compute_region_counts(
                    clip, side=side, handedness=handedness,
                    yaw_enabled=args.yaw,
                )
                write_csv(target, frame)
                ok += 1
                if not frame.attrs.get("yaw_used_3d", False):
                    yaw_fallbacks += 1
                if position % 25 == 0 or position == len(index):
                    print(f"[{position}/{len(index)}] {clip_id}  {len(frame)} frames")

            if wants_debug:
                from debug_render import render_clip_debug_video
                yaw = build_yaw_series(clip.pose_xyz, clip.pose_xyv, side,
                                       enabled=args.yaw)
                path = render_clip_debug_video(
                    Path(str(row.get("clip_path", ""))), clip,
                    output_folder / DEBUG_SUBFOLDER / clip_id / DEBUG_VIDEO_FILE,
                    side=side, handedness=handedness, yaw=yaw["yaw"],
                    max_frames=args.debug_max_frames,
                )
                debug_written += 1
                print(f"  debug video: {path if path else 'FAILED (clip unreadable)'}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"[{position}/{len(index)}] FAILED {clip_id}: "
                  f"{type(error).__name__}: {error}")

    print(f"\nDone. written={ok} skipped={skipped} "
          f"missing_landmarks={missing} failed={failed}")
    if ok:
        print(f"Clips using plain 2D normalisation (no measurable yaw): "
              f"{yaw_fallbacks}/{ok}")
    if args.debug_video:
        print(f"Debug videos for {debug_written} clips in "
              f"{output_folder / DEBUG_SUBFOLDER}")
    if exclusions:
        exclusions.report_unused()
    print(f"Next:  python3 step5_analyze.py {args.output_folder}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
