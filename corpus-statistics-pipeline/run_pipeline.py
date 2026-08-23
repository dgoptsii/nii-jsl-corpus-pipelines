"""Run the whole corpus-statistics pipeline, or any part of it.

    python3 run_pipeline.py --corpus /path/to/corpus \\
                            --annotations /path/to/parsed \\
                            --out corpus_statistics_output

Each stage writes its own outputs and reads only what an earlier stage wrote, so
any stage can be rerun on its own:

    python3 run_pipeline.py --steps 3 5 --out corpus_statistics_output

    1  index the .eaf files          (slow; the only stage that walks the corpus)
    2  build the annotation table    (fast)
    3  statistics and LaTeX tables   (fast; rerun after changing a threshold)
    4  MouthAction overlap           (re-reads the .eaf files that have the tiers)
    5  figures                       (fast; rerun after restyling)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

from config import DEFAULT_OUTPUT_FOLDER, DEFAULT_TEST_FRACTION

ALL_STEPS = [1, 2, 3, 4, 5]


def _run(step: int, args) -> int:
    if step == 1:
        import step1_index_elan
        if not args.corpus and not args.elan_list:
            print("Step 1 needs --elan-list (preferred) or --corpus",
                  file=sys.stderr)
            return 1
        return step1_index_elan.main(
            (["--elan-list", str(args.elan_list)] if args.elan_list else [])
            + (["--corpus", str(args.corpus)] if args.corpus else [])
            + ["--out", str(args.out)]
            + (["--quiet"] if args.quiet else []))
    if step == 2:
        import step2_build_table
        if not args.annotations:
            print("Step 2 needs --annotations", file=sys.stderr)
            return 1
        return step2_build_table.main(["--annotations", str(args.annotations),
                                       "--out", str(args.out)]
                                      + (["--quiet"] if args.quiet else []))
    if step == 3:
        import step3_statistics
        return step3_statistics.main(["--out", str(args.out),
                                      "--test-fraction", str(args.test_fraction)]
                                     + (["--no-latex"] if args.no_latex else []))
    if step == 4:
        import step4_mouth_overlap
        return step4_mouth_overlap.main(
            ["--out", str(args.out), "--min-overlap-ms", str(args.min_overlap_ms)]
            + (["--save-labelled"] if args.save_labelled else [])
            + (["--no-latex"] if args.no_latex else []))
    if step == 5:
        import step5_figures
        return step5_figures.main(["--out", str(args.out)])
    raise ValueError(f"Unknown step {step}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path,
                        help="folder searched recursively for .eaf files (step 1)")
    parser.add_argument("--elan-list", "--elan_list", dest="elan_list", type=Path,
                        help="the parser's selected_elan_files.csv (step 1). "
                             "Preferred over --corpus: it names one file per "
                             "recording, so duplicated copies are not counted "
                             "twice.")
    parser.add_argument("--annotations", type=Path,
                        help="folder of parsed annotation CSVs (step 2)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--steps", type=int, nargs="+", default=ALL_STEPS,
                        choices=ALL_STEPS, help="which stages to run")
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--min-overlap-ms", type=float, default=1.0)
    parser.add_argument("--save-labelled", action="store_true")
    parser.add_argument("--no-latex", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    failures: List[int] = []

    for step in args.steps:
        print(f"\n{'=' * 72}\nSTEP {step}\n{'=' * 72}")
        started = time.time()
        try:
            code = _run(step, args)
        except FileNotFoundError as error:
            print(f"Step {step} could not start: {error}", file=sys.stderr)
            code = 1
        elapsed = time.time() - started
        print(f"\n[step {step} finished in {elapsed:.1f}s, exit {code}]")
        if code != 0:
            failures.append(step)
            # Step 4 is optional -- a corpus subset may simply have no mouth
            # tiers -- but a failed 1, 2 or 3 leaves nothing for what follows.
            if step in (1, 2, 3):
                print(f"Stopping: later stages depend on step {step}.", file=sys.stderr)
                break

    print(f"\n{'=' * 72}")
    if failures:
        print(f"Finished with failures in step(s): "
              f"{', '.join(str(s) for s in failures)}")
        return 1
    print(f"All requested stages finished. Outputs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
