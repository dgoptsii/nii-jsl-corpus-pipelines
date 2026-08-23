"""Step 5 -- figures.

Draws every figure from the CSVs written by steps 3 and 4, so the plots can be
restyled without recomputing anything. A figure whose input table is missing is
skipped and named, rather than failing the stage.

    python3 step5_figures.py --out output
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DEFAULT_OUTPUT_FOLDER, FIGURES_SUBFOLDER, TABLES_SUBFOLDER
from io_utils import read_csv_safely
import figures


def _load(folder: Path, name: str) -> Optional[pd.DataFrame]:
    path = folder / f"{name}.csv"
    if not path.exists():
        return None
    frame = read_csv_safely(path, dtype=None)
    return frame if not frame.empty else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    args = parser.parse_args(argv)

    figures.apply_style()
    tables = Path(args.out) / TABLES_SUBFOLDER
    destination = Path(args.out) / FIGURES_SUBFOLDER

    plan = [
        ("summary", figures.plot_overview_tiles, "fig_corpus_overview.png"),
        ("summary", figures.plot_annotation_breakdown, "fig_annotation_breakdown.png"),
        ("keys", figures.plot_key_frequency, "fig_marker_frequency.png"),
        ("coverage_curve_full", figures.plot_coverage_curve, "fig_coverage_curve.png"),
        ("coverage", figures.plot_vocabulary_coverage, "fig_vocabulary_coverage.png"),
        ("class_sizes", figures.plot_class_sizes, "fig_class_sizes.png"),
        ("gloss_statistics", figures.plot_top_glosses, "fig_top_glosses.png"),
        ("summary", figures.plot_region_outcome, "fig_region_outcome.png"),
        ("summary", figures.plot_region_vocabulary, "fig_region_vocabulary.png"),
        ("mouth_categories", figures.plot_mouth_category_counts,
         "fig_mouth_categories.png"),
        ("mouth_key_categories", figures.plot_mouth_key_category,
         "fig_mouth_key_categories.png"),
        ("duration_distribution", figures.plot_duration_distribution,
         "fig_duration_distribution.png"),
        ("signer_balance", figures.plot_signer_balance, "fig_signer_balance.png"),
    ]

    written, skipped = [], []
    for table_name, plotter, filename in plan:
        frame = _load(tables, table_name)
        if frame is None:
            skipped.append(f"{filename} (no {table_name}.csv)")
            continue
        try:
            path = plotter(frame, destination / filename)
        except Exception as error:                      # noqa: BLE001 - reported, not raised
            skipped.append(f"{filename} ({type(error).__name__}: {error})")
            continue
        if path is None:
            skipped.append(f"{filename} (nothing to draw)")
        else:
            written.append(path)

    note = figures.missing_font_note()
    if note:
        print(f"\n{note}\n")

    for path in written:
        print(f"  {path}")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for entry in skipped:
            print(f"  {entry}")
    print(f"\nWrote {len(written)} figure(s) to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
