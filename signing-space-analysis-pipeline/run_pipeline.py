#!/usr/bin/env python3
"""Run the whole signing-space analysis pipeline.

    python3 run_pipeline.py PARSED_FOLDER --video-root /path/to/videos \\
        -output_folder ./out --keywords cl fs --regions FO GM

Stages, in order:

    1. select    parsed annotation rows for the chosen keywords   (step1)
    2. clips     one cropped video per annotation                 (step2)
    3. landmarks the reduced landmark set per clip                (step3)
    4. regions   per-frame signing-space region counts            (step4)
    5. analyse   tables and figures with signer-level CIs         (step5)

Each stage is also runnable on its own - see the step*.py scripts - which is
what you want when only the analysis changed and the landmarks are expensive to
recompute. ``--from-stage`` and ``--to-stage`` do the same thing here.

Optional metadata, one CSV, meant to be committed:
    --signers-file   signer_id,handedness,age
                     left-handed signers are mirrored; ages drive the age table
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import step1_select_rows
import step2_extract_clips
import step3_extract_landmarks
import step4_region_counts
import step5_analyze
from config import (
    DEFAULT_DEBUG_LIMIT,
    DEFAULT_MODEL_COMPLEXITY,
    DEFAULT_OUTPUT_FOLDER,
)
from io_utils import count_or_all

STAGES = ["select", "clips", "landmarks", "regions", "analyse"]
LINE = "=" * 78


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Signing-space analysis: annotations -> clips -> landmarks -> regions -> figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 run_pipeline.py ./parsed --video-root ~/corpus -output_folder ./out\n"
            "  python3 run_pipeline.py ./parsed --video-root ~/corpus --keywords cl fs --regions FO GM\n"
            "  python3 run_pipeline.py ./parsed --video-root ~/corpus --from-stage analyse\n"
        ),
    )
    parser.add_argument("parsed_folder", type=Path,
                        help="Folder of *_parsed.csv files from the annotation pipeline.")
    parser.add_argument("-output_folder", "--output_folder", dest="output_folder",
                        type=Path, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--video-root", "--video_root", dest="video_root",
                        type=Path, action="append", default=[],
                        help="Folder holding the source videos. Repeatable. "
                             "Required unless you start from a later stage.")

    parser.add_argument("--keywords", nargs="+", default=["lexical_item", "cl", "fs", "pt"])
    parser.add_argument("--regions", nargs="*", default=[])
    parser.add_argument("--lexical-only", dest="lexical_only", action="store_true")

    parser.add_argument("-signers_file", "--signers_file", "--signers-file",
                        dest="signers_file", type=Path, default=None,
                        metavar="SIGNERS.csv",
                        help="CSV of signer_id,handedness,age.")

    parser.add_argument("--workers", "-j", dest="workers",
                        type=step3_extract_landmarks.workers_value, default=1,
                        metavar="N|auto",
                        help="Processes used for landmark extraction, the slow "
                             "stage. 'auto' uses one per core, less one. "
                             "Default 1.")
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2],
                        default=DEFAULT_MODEL_COMPLEXITY)
    parser.add_argument("--refine-face", action="store_true")
    parser.add_argument("--no-person-mask", dest="person_mask",
                        action="store_false", default=True)
    parser.add_argument("--no-yaw", dest="yaw", action="store_false", default=True)
    parser.add_argument("--hand-role", choices=["dominant", "non_dominant"],
                        default="dominant")
    parser.add_argument("--figure-font-scale", "--figure_font_scale",
                        dest="font_scale", type=float, default=1.0,
                        help="Multiply figure type sizes. ~1.6 for an A0 poster.")

    parser.add_argument("--limit", type=int, default=0,
                        help="Cap clips per keyword; useful for a trial run.")
    parser.add_argument("--exclude-file", "--exclude_file", dest="exclude_file",
                        action="append",
                        type=Path, default=None, metavar="EXCLUDED.txt",
                        help="Text file of clip names to drop after inspecting "
                             "the cropped videos. Applied from landmark "
                             "extraction onward; the clips themselves are kept.")
    parser.add_argument("--save-debug", dest="save_debug", action="store_true",
                        help="Diagnostic renders for the first few clips: YOLO "
                             "mask images (step 3) and an example video of the "
                             "clip beside its signing space (step 4). Off by "
                             "default; slow and large.")
    parser.add_argument("--debug-limit", "--debug_limit", dest="debug_limit",
                        type=count_or_all, default=DEFAULT_DEBUG_LIMIT,
                        metavar="N|all",
                        help=f"How many clips get diagnostic renders: a number, "
                             f"or 'all'. Default {DEFAULT_DEBUG_LIMIT}.")
    parser.add_argument("--debug-max-frames", "--debug_max_frames",
                        dest="debug_max_frames", type=count_or_all, default=0,
                        metavar="N|all",
                        help="Frames rendered per debug video: a number, or "
                             "'all' for the whole clip (the default).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--from-stage", choices=STAGES, default="select")
    parser.add_argument("--to-stage", choices=STAGES, default="analyse")
    return parser


def selected_stages(first: str, last: str) -> List[str]:
    return STAGES[STAGES.index(first):STAGES.index(last) + 1]


def announce(stage: str, number: int, total: int) -> float:
    print(f"\n{LINE}\nSTAGE {number}/{total}: {stage}\n{LINE}")
    return time.time()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    stages = selected_stages(args.from_stage, args.to_stage)

    output = str(args.output_folder)
    keywords: List[str] = list(args.keywords)
    regions: List[str] = list(args.regions)

    if "clips" in stages and not args.video_root:
        print("ERROR: --video-root is required to cut clips. "
              "Use --from-stage landmarks to skip cutting.", file=sys.stderr)
        return 1

    print(LINE)
    print("SIGNING-SPACE ANALYSIS PIPELINE")
    print(LINE)
    print(f"Parsed annotations: {args.parsed_folder}")
    print(f"Output folder:      {args.output_folder}")
    print(f"Keywords:           {', '.join(keywords)}")
    print(f"Regions:            {', '.join(regions) if regions else 'ALL'}")
    print(f"Stages:             {' -> '.join(stages)}")
    print(f"Extraction workers: "
          f"{step3_extract_landmarks.resolve_workers(args.workers)}")
    print(f"Signers file:       {args.signers_file or '(none: all right-handed, ages unknown)'}")
    if args.exclude_file:
        print("Excluded clips:     "
              + ", ".join(str(p) for p in args.exclude_file))
    if args.save_debug:
        print(f"Diagnostic renders: {args.debug_limit or 'all'} clips "
              f"-> {args.output_folder}/debug")

    started = time.time()
    total = len(stages)

    for number, stage in enumerate(stages, start=1):
        stage_started = announce(stage, number, total)
        code = 0

        if stage == "select":
            argv_stage = [str(args.parsed_folder), "-output_folder", output,
                          "--keywords", *keywords]
            if regions:
                argv_stage += ["--regions", *regions]
            if args.lexical_only:
                argv_stage.append("--lexical-only")
            code = step1_select_rows.main(argv_stage)

        elif stage == "clips":
            argv_stage = [output]
            for root in args.video_root:
                argv_stage += ["--video-root", str(root)]
            if regions:
                argv_stage += ["--regions", *regions]
            if args.signers_file:
                argv_stage += ["--signers-file", str(args.signers_file)]
            if args.limit:
                argv_stage += ["--limit", str(args.limit)]
            code = step2_extract_clips.main(argv_stage)

        elif stage == "landmarks":
            argv_stage = [output, "--model-complexity", str(args.model_complexity),
                          "--workers", str(args.workers or "auto")]
            if args.refine_face:
                argv_stage.append("--refine-face")
            if not args.person_mask:
                argv_stage.append("--no-person-mask")
            if args.overwrite:
                argv_stage.append("--overwrite")
            if regions:
                argv_stage += ["--regions", *regions]
            if args.exclude_file:
                for excluded in args.exclude_file:
                    argv_stage += ["--exclude-file", str(excluded)]
            if args.save_debug:
                argv_stage += ["--save-debug-images",
                               "--debug-limit", str(args.debug_limit)]
            code = step3_extract_landmarks.main(argv_stage)

        elif stage == "regions":
            argv_stage = [output]
            if not args.yaw:
                argv_stage.append("--no-yaw")
            if args.overwrite:
                argv_stage.append("--overwrite")
            if regions:
                argv_stage += ["--regions", *regions]
            if args.exclude_file:
                for excluded in args.exclude_file:
                    argv_stage += ["--exclude-file", str(excluded)]
            if args.save_debug:
                argv_stage += ["--save-debug-video",
                               "--debug-limit", str(args.debug_limit),
                               "--debug-max-frames", str(args.debug_max_frames)]
            code = step4_region_counts.main(argv_stage)

        elif stage == "analyse":
            argv_stage = [output, "--hand-role", args.hand_role,
                          "--figure-font-scale", str(args.font_scale)]
            if regions:
                argv_stage += ["--regions", *regions]
            if args.signers_file:
                argv_stage += ["--signers-file", str(args.signers_file)]
            if args.exclude_file:
                for excluded in args.exclude_file:
                    argv_stage += ["--exclude-file", str(excluded)]
            code = step5_analyze.main(argv_stage)

        print(f"\n[{stage}] finished in {time.time() - stage_started:.0f}s (exit {code})")
        if code not in (0, 2):
            print(f"\nStopping: stage '{stage}' failed.", file=sys.stderr)
            return code

    print(f"\n{LINE}\nPipeline complete in {time.time() - started:.0f}s")
    print(f"Tables:  {args.output_folder}/tables")
    print(f"Figures: {args.output_folder}/figures")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
