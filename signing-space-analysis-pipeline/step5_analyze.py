#!/usr/bin/env python3
"""STEP 5: aggregate region counts into tables and figures.

    python3 step5_analyze.py OUTPUT_FOLDER [options]

Writes ``tables/`` (clip_level, by_region_and_keyword, by_age_group_and_keyword,
by_age_band_and_keyword, by_gender_and_keyword, by_region_age_gender,
region_distribution, region_groups, central_periphery_summary; all with 95% CIs
bootstrapped over signers) and ``figures/`` (body_map_<KEYWORD>_<REGION>,
avg_regions_by_region / _age_group / _age_band / _gender, region_groups).

Age bands and gender are resolved here, not frozen into the clip index, so
regrouping ages costs seconds rather than a re-extraction:

    python3 step5_analyze.py OUTPUT_FOLDER --signers-file input_lists/signers.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from config import (
    CLIPS_SUBFOLDER,
    CLIP_INDEX_FILE,
    FIGURES_SUBFOLDER,
    HAND_ROLES,
    REGION_COUNTS_FILE,
    REGION_COUNTS_SUBFOLDER,
    TABLES_SUBFOLDER,
)
from exclusions import ClipExclusions, filter_index, load_exclusions
from latex_tables import write_report_tables
from stats import comparison_scoreboard, effect_spread, pairwise_comparisons
from figures import body_map_figure, grouped_bar_figure, region_group_figure
from io_utils import read_csv_safely, write_csv
from signers import SignerMetadata, load_signer_metadata
from stats import (
    age_band_by_keyword_table,
    age_group_by_keyword_table,
    build_clip_table,
    central_periphery_summary,
    cross_table,
    gender_by_keyword_table,
    region_by_keyword_table,
    region_distribution,
    region_group_distribution,
)


def load_frames(output_folder: Path, keywords: Sequence[str],
                regions: Sequence[str],
                metadata: Optional[SignerMetadata] = None,
                exclusions: Optional[ClipExclusions] = None) -> pd.DataFrame:
    """Concatenate every per-frame counts CSV, joined to its clip metadata.

    Age band and gender are *labels*: they change no geometry, so they are resolved
    here rather than frozen into the clip index at step 2. That is what makes
    "regroup the ages" a seconds-long re-run of step 5. Both are read from the
    participant ID (``FO_08_FK_50F`` is a 50-band female signer), so this happens
    with or without ``--signers-file``.

    Handedness is deliberately NOT re-resolved: it decides whether a signer's space
    is mirrored, so changing it invalidates the stored region counts and step 4
    must be re-run.
    """
    index_path = output_folder / CLIPS_SUBFOLDER / CLIP_INDEX_FILE
    counts_root = output_folder / REGION_COUNTS_SUBFOLDER

    index = read_csv_safely(index_path)
    if keywords:
        index = index[index["keyword"].str.upper().isin({k.upper() for k in keywords})]
    if regions:
        index = index[index["region_code"].str.upper().isin({r.upper() for r in regions})]

    if exclusions:
        index, dropped = filter_index(index, exclusions)
        print(f"{exclusions.describe()}  ({dropped} clips dropped)")

    parts: List[pd.DataFrame] = []
    for _, row in index.iterrows():
        path = counts_root / str(row["clip_id"]) / REGION_COUNTS_FILE
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for column in ("keyword", "region_code", "clip_id", "signer_id",
                       "handedness", "age_group", "gender"):
            frame[column] = row.get(column, "")

        signer_id = str(row.get("signer_id", ""))
        frame["age_group"] = metadata.age_group(signer_id)
        frame["age_band"] = metadata.age_band(signer_id)
        frame["gender"] = metadata.gender(signer_id)

        parts.append(frame)

    if not parts:
        raise RuntimeError(
            f"No region-count CSVs found under {counts_root}. Run step 4 first."
        )
    return pd.concat(parts, ignore_index=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="step5_analyze.py",
        description="Aggregate region counts into tables and figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python3 step5_analyze.py ./out --hand-role dominant\n",
    )
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--keywords", nargs="*", default=[])
    parser.add_argument("--regions", nargs="*", default=[])
    parser.add_argument("-signers_file", "--signers_file", "--signers-file",
                        dest="signers_file", type=Path, default=None,
                        metavar="SIGNERS.csv",
                        help="Re-read age and gender from this CSV instead of "
                             "using the labels frozen into the clip index. This "
                             "is how you regroup ages or add gender without "
                             "re-cutting clips or re-extracting landmarks.")
    parser.add_argument("--exclude-file", "--exclude_file", dest="exclude_file",
                        type=Path, action="append", default=None,
                        metavar="EXCLUDED.txt",
                        help="Text file of clip names to leave out after "
                             "visual inspection, one per line. Repeatable: "
                             "give it once per list, e.g. --exclude-file "
                             "input_lists/excluded_clips.txt")
    parser.add_argument("--hand-role", choices=HAND_ROLES, default="dominant",
                        help="Hand role drawn in the bar figures. The tables "
                             "always carry both roles, one row each.")
    parser.add_argument("--no-figures", dest="figures", action="store_false", default=True)
    parser.add_argument("--figure-font-scale", "--figure_font_scale",
                        dest="font_scale", type=float, default=1.0,
                        help="Multiply every type size and the figure size. "
                             "Use ~1.6 for an A0 poster read from two metres.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    output_folder = args.output_folder.expanduser().resolve()
    tables_dir = output_folder / TABLES_SUBFOLDER
    figures_dir = output_folder / FIGURES_SUBFOLDER

    try:
        metadata = load_signer_metadata(args.signers_file)
        exclusions = load_exclusions(args.exclude_file)
        frames = load_frames(output_folder, args.keywords, args.regions,
                             metadata, exclusions)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if exclusions:
        exclusions.report_unused()

    print(f"Frames loaded:   {len(frames)}")
    clip_table = build_clip_table(frames, HAND_ROLES)
    write_csv(tables_dir / "clip_level.csv", clip_table)

    present = clip_table[clip_table["hand_present"]]
    print(f"Clips:           {clip_table['clip_id'].nunique()}")
    print(f"Signers:         {clip_table['signer_id'].nunique()}")
    print(f"Left-handed:     {clip_table[clip_table['handedness'] == 'left']['signer_id'].nunique()} signers")
    if args.signers_file:
        print(f"Age and gender:  re-read from {args.signers_file}")
    signers_once = clip_table.drop_duplicates("signer_id")
    print(f"Age decades:     "
          + ", ".join(f"{band}={n}" for band, n
                      in signers_once["age_group"].value_counts().sort_index().items()))
    print(f"Age bands:       "
          + ", ".join(f"{band}={n}" for band, n
                      in signers_once["age_band"].value_counts().sort_index().items()))
    print(f"Gender:          "
          + ", ".join(f"{value}={n}" for value, n
                      in signers_once["gender"].value_counts().sort_index().items()) + "\n")

    # --- the summary tables ----------------------------------------------
    # Each carries one row per hand role: the dominant and the non-dominant
    # hand are averaged separately, never pooled.
    by_region = region_by_keyword_table(clip_table, hand_roles=HAND_ROLES)
    by_age = age_group_by_keyword_table(clip_table, hand_roles=HAND_ROLES)
    by_age_band = age_band_by_keyword_table(clip_table, hand_roles=HAND_ROLES)
    by_gender = gender_by_keyword_table(clip_table, hand_roles=HAND_ROLES)
    crossed = cross_table(clip_table, hand_roles=HAND_ROLES)

    write_csv(tables_dir / "by_region_and_keyword.csv", by_region)
    write_csv(tables_dir / "by_age_group_and_keyword.csv", by_age)
    write_csv(tables_dir / "by_age_band_and_keyword.csv", by_age_band)
    write_csv(tables_dir / "by_gender_and_keyword.csv", by_gender)
    write_csv(tables_dir / "by_region_age_gender.csv", crossed)

    # --- distributions --------------------------------------------------
    distribution = region_distribution(clip_table, ("keyword", "region_code"), HAND_ROLES)
    groups = region_group_distribution(distribution)
    summary = central_periphery_summary(distribution)
    write_csv(tables_dir / "region_distribution.csv", distribution)
    write_csv(tables_dir / "region_groups.csv", groups)
    write_csv(tables_dir / "central_periphery_summary.csv", summary)

    # --- how much each grouping actually moves the answer -----------------
    # Counting non-overlapping intervals answers "can we tell these apart?",
    # which collapses to zero for any grouping with only two levels and hides a
    # difference in size. The spread of the level means is reported alongside,
    # in regions per sign, so prefecture and gender can be compared on scale
    # rather than only on detectability.
    comparisons, spreads = [], []
    for column in ("region_code", "age_band", "gender", "keyword"):
        if column not in clip_table.columns:
            continue
        within = "region_code" if column == "keyword" else "keyword"
        comparisons.append(pairwise_comparisons(clip_table, column,
                                                within_column=within,
                                                hand_role=args.hand_role))
        spreads.append(effect_spread(clip_table, column, within_column=within,
                                     hand_role=args.hand_role))
    comparisons = pd.concat([c for c in comparisons if not c.empty], ignore_index=True) \
        if any(not c.empty for c in comparisons) else pd.DataFrame()
    spreads = pd.concat([s for s in spreads if not s.empty], ignore_index=True) \
        if any(not s.empty for s in spreads) else pd.DataFrame()

    if not comparisons.empty:
        write_csv(tables_dir / "pairwise_differences.csv", comparisons)
        write_csv(tables_dir / "comparison_scoreboard.csv",
                  comparison_scoreboard(comparisons))
    if not spreads.empty:
        write_csv(tables_dir / "effect_spread.csv", spreads)

    # The report \inputs these straight out of this folder, so they are written
    # here rather than copied by hand. A copied table goes stale silently, and a
    # stale number in a typeset PDF is invisible.
    for path in write_report_tables(tables_dir, args.hand_role):
        print(f"  {path}")

    print("Tables written:")
    for name in ("clip_level", "by_region_and_keyword", "by_age_group_and_keyword",
                 "by_age_band_and_keyword",
                 "by_gender_and_keyword", "by_region_age_gender",
                 "region_distribution", "region_groups", "central_periphery_summary"):
        print(f"  {tables_dir / (name + '.csv')}")

    if not by_region.empty:
        print("\nTable 1 - region x keyword x hand role:")
        print(by_region[["region_code", "keyword", "hand_role", "n_annotations",
                         "hand_present_percent", "n_signers",
                         "avg_regions", "ci_low", "ci_high"]].to_string(index=False))

    if not args.figures:
        return 0

    print("\nFigures:")
    made = 0
    for (keyword, region_code), group in distribution.groupby(["keyword", "region_code"]):
        clips = present[(present["keyword"] == keyword)
                        & (present["region_code"] == region_code)]
        path = body_map_figure(
            group, str(keyword), str(region_code),
            n_clips=int(clips["clip_id"].nunique()),
            n_signers=int(clips["signer_id"].nunique()),
            out_path=figures_dir / f"body_map_{keyword}_{region_code}.png",
            font_scale=args.font_scale,
        )
        print(f"  {path}")
        made += 1

    # The tables hold both hands; a bar chart holding both is unreadable, so the
    # figures show one role - --hand-role picks it, and the title says which.
    role_label = args.hand_role.replace("_", "-") + " hand"
    for table, column, name, title in (
        (by_region, "region_code", "avg_regions_by_region",
         f"Average signing-space regions per clip ({role_label}), by geographical region"),
        (by_age, "age_group", "avg_regions_by_age_group",
         f"Average signing-space regions per clip ({role_label}), by age decade"),
        (by_age_band, "age_band", "avg_regions_by_age_band",
         f"Average signing-space regions per clip ({role_label}), under 50 vs 50+"),
        (by_gender, "gender", "avg_regions_by_gender",
         f"Average signing-space regions per clip ({role_label}), by gender"),
    ):
        table = table[table["hand_role"] == args.hand_role] if not table.empty else table
        path = grouped_bar_figure(table, column, figures_dir / f"{name}.png", title,
                                  font_scale=args.font_scale)
        if path:
            print(f"  {path}")
            made += 1

    path = region_group_figure(groups, figures_dir / "region_groups.png",
                               font_scale=args.font_scale)
    if path:
        print(f"  {path}")
        made += 1

    print(f"\nDone. {made} figures in {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
