#!/usr/bin/env python3
"""STEP 3 - detect and store the reduced landmark set for every clip.

Only the points the analysis uses are written to disk - shoulders, elbows,
wrists, the knuckle of every finger, chin and head top - about 50 floats per
frame instead of MediaPipe's ~1200.

    python3 step3_extract_landmarks.py OUTPUT_FOLDER [options]

Detection cost is reduced separately from storage: ``--model-complexity 1`` and
the unrefined face mesh are the defaults, since the refined 478-point mesh only
improves eyes and lips, which are unused here. Pass ``--model-complexity 2
--refine-face`` to reproduce the older, slower settings exactly.

This stage is the slow one, and it is CPU-bound: MediaPipe Holistic has no GPU
path in the Python package. Clips are independent, so ``--workers N`` spreads
them over N processes, which is where a many-core server pays off:

    python3 step3_extract_landmarks.py OUTPUT_FOLDER --workers auto

Output
------
    OUT/landmarks/<CLIP_ID>/landmarks.npz
    OUT/landmarks/<CLIP_ID>/landmarks_meta.json

With ``--save-debug-images`` (off by default), also:

    OUT/debug/<CLIP_ID>/yolo_person_masks.jpg   which signer YOLO kept
    OUT/debug/<CLIP_ID>/mediapipe_input.jpg     the frame MediaPipe was given
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from config import (
    CLIPS_SUBFOLDER,
    CLIP_INDEX_FILE,
    DEBUG_SUBFOLDER,
    DEFAULT_DEBUG_LIMIT,
    DEFAULT_MODEL_COMPLEXITY,
    DEFAULT_REFINE_FACE,
    LANDMARKS_FILE,
    LANDMARKS_SUBFOLDER,
)
from exclusions import filter_index, load_exclusions
from io_utils import count_or_all, read_csv_safely

# ``landmarks`` is imported inside process_clip, not here: it pulls in OpenCV
# and MediaPipe, and the parent process only builds a work list.


# ===========================================================================
# ONE CLIP - the unit of work, shared by the serial and the parallel path
# ===========================================================================

def process_clip(task: Dict[str, object]) -> Dict[str, object]:
    """Extract and save one clip's landmarks. Never raises.

    ``task`` is a plain dict so it can be pickled to a worker process, and the
    result is a plain dict for the same reason. Both paths call this, so a
    parallel run cannot drift from a serial one.
    """
    clip_id = str(task["clip_id"])
    clip_path = Path(str(task["clip_path"]))
    result: Dict[str, object] = {"clip_id": clip_id, "position": task.get("position", 0)}

    if not clip_path.exists():
        return {**result, "status": "missing", "message": str(clip_path)}

    try:
        from landmarks import extract_clip_landmarks, save_landmarks

        debug_dir = task.get("debug_dir")
        clip = extract_clip_landmarks(
            clip_path,
            side=str(task["side"]),
            model_complexity=int(task["model_complexity"]),
            refine_face=bool(task["refine_face"]),
            use_person_mask=bool(task["person_mask"]),
            debug_dir=Path(str(debug_dir)) if debug_dir else None,
        )
        save_landmarks(Path(str(task["out_dir"])), clip, dict(task["meta"]))
        return {**result, "status": "ok", "n_frames": len(clip),
                "n_hands_rejected": int(clip.n_hands_rejected)}
    except Exception as error:  # noqa: BLE001 - one bad clip must not stop the run
        return {**result, "status": "failed",
                "message": f"{type(error).__name__}: {error}"}


def _limit_thread_usage() -> None:
    """Keep each worker to one thread.

    OpenCV and the BLAS libraries under MediaPipe both default to using every
    core. With N worker processes that is N x cores threads fighting over the
    same cores, which is reliably slower than running serially.
    """
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:  # pragma: no cover - cv2 always present in practice
        pass


def resolve_workers(value: int) -> int:
    """Turn the ``--workers`` argument into a process count.

    ``0`` means "decide for me": one process per core, less one, so the machine
    stays usable and the parent has room to collect results.
    """
    cores = os.cpu_count() or 1
    if value <= 0:
        return max(1, cores - 1)
    return max(1, value)


def workers_value(text: str) -> int:
    """argparse type: a positive count, or ``auto``."""
    value = str(text).strip().lower()
    if value in {"auto", "max", "cores"}:
        return 0
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number of processes or 'auto', not {text!r}"
        ) from None
    if number < 1:
        raise argparse.ArgumentTypeError("need at least one worker; use 'auto' to pick")
    return number


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="step3_extract_landmarks.py",
        description="Extract the reduced landmark set from every cropped clip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python3 step3_extract_landmarks.py ./out --model-complexity 1\n",
    )
    parser.add_argument("output_folder", type=Path,
                        help="The folder step 2 wrote clips into.")
    parser.add_argument("--keywords", nargs="*", default=[],
                        help="Only these keywords. Empty = all.")
    parser.add_argument("--regions", nargs="*", default=[],
                        help="Only these geographical regions. Empty = all.")
    parser.add_argument("--model-complexity", type=int, choices=[0, 1, 2],
                        default=DEFAULT_MODEL_COMPLEXITY,
                        help=f"MediaPipe pose complexity. Default {DEFAULT_MODEL_COMPLEXITY}.")
    parser.add_argument("--refine-face", action="store_true", default=DEFAULT_REFINE_FACE,
                        help="Use the refined 478-point face mesh (slower, unused here).")
    parser.add_argument("--no-person-mask", dest="person_mask", action="store_false",
                        default=True, help="Skip YOLO masking of the other signer.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract clips that already have landmarks.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many clips (0 = all).")
    parser.add_argument("--exclude-file", "--exclude_file", dest="exclude_file",
                        type=Path, action="append", default=None,
                        metavar="EXCLUDED.txt",
                        help="Text file of clip names to skip, one per line. "
                             "Repeatable: give it once per list, e.g. a list "
                             "written after inspecting the cropped videos and "
                             "one written by audit_landmarks.py.")
    parser.add_argument("--workers", "-j", dest="workers", type=workers_value,
                        default=1, metavar="N|auto",
                        help="Extract this many clips at once, in separate "
                             "processes. 'auto' uses one per core, less one. "
                             "Default 1. Each worker holds its own MediaPipe "
                             "and YOLO models, so budget ~1-2 GB of RAM each.")
    parser.add_argument("--save-debug-images", dest="debug_images", action="store_true",
                        help="Write, per clip, the first frame with YOLO's target "
                             "and removed silhouettes drawn, plus the masked frame "
                             "MediaPipe actually received. Diagnostic only.")
    parser.add_argument("--debug-limit", "--debug_limit", dest="debug_limit",
                        type=count_or_all, default=DEFAULT_DEBUG_LIMIT,
                        metavar="N|all",
                        help=f"How many clips get debug images: a number, or "
                             f"'all'. Default {DEFAULT_DEBUG_LIMIT}.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    output_folder = args.output_folder.expanduser().resolve()
    index_path = output_folder / CLIPS_SUBFOLDER / CLIP_INDEX_FILE
    landmarks_root = output_folder / LANDMARKS_SUBFOLDER

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
    if args.limit:
        index = index.head(args.limit)

    if index.empty:
        print("Nothing to do: no clips match the selection.")
        return 1

    # --- build the work list ---------------------------------------------
    tasks: List[Dict[str, object]] = []
    skipped = debug_written = 0

    for _, row in index.iterrows():
        clip_id = str(row["clip_id"])
        out_dir = landmarks_root / clip_id

        if (out_dir / LANDMARKS_FILE).exists() and not args.overwrite:
            skipped += 1
            continue

        debug_dir = None
        if args.debug_images and (not args.debug_limit or debug_written < args.debug_limit):
            debug_dir = str(output_folder / DEBUG_SUBFOLDER / clip_id)
            debug_written += 1

        tasks.append({
            "clip_id": clip_id,
            "clip_path": str(row["clip_path"]),
            "out_dir": str(out_dir),
            "side": str(row.get("side", "left")),
            "model_complexity": int(args.model_complexity),
            "refine_face": bool(args.refine_face),
            "person_mask": bool(args.person_mask),
            "debug_dir": debug_dir,
            "meta": {
                "clip_id": clip_id,
                "clip_path": str(row["clip_path"]),
                "keyword": row.get("keyword", ""),
                "region_code": row.get("region_code", ""),
                "signer_id": row.get("signer_id", ""),
                "handedness": row.get("handedness", "right"),
                "age_group": row.get("age_group", "unknown"),
                "side": row.get("side", "left"),
                "model_complexity": int(args.model_complexity),
                "refine_face": bool(args.refine_face),
                "person_mask": bool(args.person_mask),
            },
        })

    workers = min(resolve_workers(args.workers), max(1, len(tasks)))

    print(f"Clips:            {len(index)}  ({len(tasks)} to extract, "
          f"{skipped} already done)")
    print(f"Landmarks folder: {landmarks_root}")
    print(f"Model complexity: {args.model_complexity}   refine face: {args.refine_face}")
    print(f"Person masking:   {args.person_mask}")
    if exclusions:
        print(f"{exclusions.describe()}  ({n_excluded} clips dropped)")
    print(f"Workers:          {workers}"
          f"{' (serial)' if workers == 1 else f' processes on {os.cpu_count()} cores'}")
    if args.debug_images:
        print(f"Debug images:     {output_folder / DEBUG_SUBFOLDER}"
              f"  ({args.debug_limit or 'all'} of {len(index)} clips)")
    print()

    for position, task in enumerate(tasks, start=1):
        task["position"] = position

    started = time.time()
    ok = failed = 0
    total = len(tasks)

    def report(result: Dict[str, object], done: int) -> None:
        """One line per clip, in completion order."""
        prefix = f"[{done}/{total}]"
        status = result.get("status")
        if status == "ok":
            print(f"{prefix} {result['clip_id']}  {result.get('n_frames', 0)} frames, "
                  f"{result.get('n_hands_rejected', 0)} off-target hands rejected")
        elif status == "missing":
            print(f"{prefix} MISSING {result.get('message', '')}")
        else:
            print(f"{prefix} FAILED {result['clip_id']}: {result.get('message', '')}")

    if workers == 1:
        for done, task in enumerate(tasks, start=1):
            result = process_clip(task)
            ok += result["status"] == "ok"
            failed += result["status"] != "ok"
            report(result, done)
    else:
        # 'spawn' rather than the Linux default 'fork': MediaPipe and torch both
        # hold state that does not survive being forked into a child.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=_limit_thread_usage) as pool:
            futures = {pool.submit(process_clip, task): task for task in tasks}
            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001 - a worker died outright
                    task = futures[future]
                    result = {"clip_id": task["clip_id"], "status": "failed",
                              "message": f"worker died: {type(error).__name__}: {error}"}
                ok += result["status"] == "ok"
                failed += result["status"] != "ok"
                report(result, done)

    elapsed = time.time() - started
    print(f"\nDone in {elapsed:.0f}s. extracted={ok} skipped={skipped} failed={failed}")
    if ok:
        print(f"Throughput: {ok / max(elapsed, 1e-6):.2f} clips/s "
              f"with {workers} worker{'s' if workers > 1 else ''}")
    if args.debug_images:
        print(f"Debug images for {debug_written} clips in "
              f"{output_folder / DEBUG_SUBFOLDER}")
    if exclusions:
        exclusions.report_unused()
    print(f"Next:  python3 step4_region_counts.py {args.output_folder}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
