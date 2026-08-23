"""LaTeX versions of the summary tables, written next to the CSVs.

The report reads these directly out of ``output/tables``, so rerunning step 5
updates the report. Nothing is copied into the report folder by hand -- a copied
table is a table that goes stale silently, and a stale number in a typeset PDF
is invisible.

Each file is a **complete** ``tabular`` environment. A fragment ending in a row
break would break the ``\\bottomrule`` of whatever includes it, which is a hard
error to read once it is three files deep.

Three commands are expected from the including document, so the category names
can be styled in one place:

    \\newcommand{\\CL}{\\textsc{cl}}
    \\newcommand{\\FS}{\\textsc{fs}}
    \\newcommand{\\LEX}{lexical}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from config import MIN_SIGNERS_FOR_STABLE_CI

#: Prefecture code -> the name printed in the table. Kept here rather than in
#: config.py because it is a presentation detail: the pipeline itself only ever
#: needs the code.
REGION_NAMES: Dict[str, str] = {
    "GM": "Gunma", "NR": "Nara", "NS": "Nagasaki", "FO": "Fukuoka",
    "IS": "Ishikawa", "TY": "Toyama", "IK": "Ibaraki",
}

#: Order the categories appear in every table.
KEYWORD_ORDER: List[str] = ["CL", "FS", "LEXICAL_ITEM"]

#: Category -> the macro the document defines for it.
KEYWORD_MACRO: Dict[str, str] = {
    "CL": r"\CL", "FS": r"\FS", "LEXICAL_ITEM": r"\LEX",
}

HAND_LABEL = {"dominant": "dominant", "non_dominant": "non-dominant"}

#: Marks a cell whose CI rests on too few signers to be stable.
UNSTABLE_MARK = r"\,$\dagger$"


def _keyword_label(keyword: str) -> str:
    return KEYWORD_MACRO.get(str(keyword), str(keyword).replace("_", r"\_"))


def _signers(row: pd.Series) -> str:
    """Signer count, flagged when the interval rests on too few of them."""
    n = int(row["n_signers"])
    reliable = str(row.get("ci_reliable", "True")).strip().lower() in {"true", "1", "yes"}
    return f"{n}" if reliable and n >= MIN_SIGNERS_FOR_STABLE_CI else f"{n}{UNSTABLE_MARK}"


def _interval(row: pd.Series) -> str:
    return f"[{float(row['ci_low']):.2f}, {float(row['ci_high']):.2f}]"


def _write(path: Path, lines: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _by_role(group: pd.DataFrame) -> Dict[str, pd.Series]:
    return {str(r["hand_role"]): r for _, r in group.iterrows()}


def _level_key(level: object):
    """Sort age bands youngest first, everything else alphabetically.

    Plain string sorting puts "50+" before "<50", because "5" sorts before "<".
    The band is an ordered quantity, so it is read in the wrong order that way.
    """
    text = str(level).strip()
    digits = "".join(c for c in text if c.isdigit())
    if not digits:
        return (1, text)
    # "<50" opens the band that ends at 50; "50+" opens the one that starts there.
    return (0, int(digits), 0 if text.startswith("<") else 1)


# ===========================================================================
# TABLE 1 -- prefecture x category, both hands side by side
# ===========================================================================

def region_table(by_region: pd.DataFrame, path: Path) -> Optional[Path]:
    """Prefectures down the page, the two hand roles across it.

    Side by side rather than as separate rows because the comparison a reader
    makes is dominant against non-dominant within one cell of the design.
    """
    if by_region.empty:
        return None

    lines = [
        r"\begin{tabular}{@{}llrrcccc@{}}",
        r"\toprule",
        r"& & & & \multicolumn{2}{c}{\textbf{dominant hand}} "
        r"& \multicolumn{2}{c}{\textbf{non-dominant hand}} \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"\textbf{Prefecture} & \textbf{Category} & \textbf{clips} & "
        r"\textbf{signers} & mean & 95\,\% CI & mean & 95\,\% CI \\",
        r"\midrule",
    ]

    codes = sorted(by_region["region_code"].unique(),
                   key=lambda c: REGION_NAMES.get(str(c), str(c)))
    for position, code in enumerate(codes):
        if position:
            lines.append(r"\addlinespace")
        block = by_region[by_region["region_code"] == code]
        first = True
        for keyword in KEYWORD_ORDER:
            rows = _by_role(block[block["keyword"] == keyword])
            if not rows:
                continue
            dominant = rows.get("dominant")
            other = rows.get("non_dominant")
            if dominant is None:
                continue
            name = REGION_NAMES.get(str(code), str(code)) if first else ""
            first = False
            lines.append(
                f"{name} & {_keyword_label(keyword)} & "
                f"{int(dominant['n_annotations'])} & {_signers(dominant)} & "
                f"{float(dominant['avg_regions']):.2f} & {_interval(dominant)} & "
                + (f"{float(other['avg_regions']):.2f} & {_interval(other)}"
                   if other is not None else "-- & --")
                + r" \\"
            )

    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write(path, lines)


# ===========================================================================
# TABLE 2 -- age band and gender
# ===========================================================================

def age_gender_table(by_age_band: pd.DataFrame, by_gender: pd.DataFrame,
                     path: Path, hand_role: str = "dominant") -> Optional[Path]:
    """Age band above, gender below, one hand role.

    One role only: the point of this table is that neither grouping moves the
    number much, and doubling every row would bury that.
    """
    if by_age_band.empty and by_gender.empty:
        return None

    lines = [
        r"\begin{tabular}{@{}lllrrcc@{}}",
        r"\toprule",
        r"\textbf{Grouping} & \textbf{Level} & \textbf{Category} & "
        r"\textbf{clips} & \textbf{signers} & \textbf{mean} & \textbf{95\,\% CI} \\",
        r"\midrule",
    ]

    def block(frame: pd.DataFrame, column: str, title: str, first_block: bool) -> None:
        if frame.empty:
            return
        if not first_block:
            lines.append(r"\midrule")
        rows = frame[frame["hand_role"] == hand_role]
        first_level = True
        for level in sorted(rows[column].unique(), key=_level_key):
            if not first_level:
                lines.append(r"\addlinespace[1mm]")
            grouping = title if first_level else ""
            first_level = False
            first_keyword = True
            for keyword in KEYWORD_ORDER:
                match = rows[(rows[column] == level) & (rows["keyword"] == keyword)]
                if match.empty:
                    continue
                row = match.iloc[0]
                lines.append(
                    f"{grouping if first_keyword else ''} & "
                    f"{str(level).replace('<', '$<$') if first_keyword else ''} & "
                    f"{_keyword_label(keyword)} & {int(row['n_annotations'])} & "
                    f"{_signers(row)} & {float(row['avg_regions']):.2f} & "
                    f"{_interval(row)} \\\\"
                )
                first_keyword = False
                grouping = ""
        lines.append(r"\addlinespace[1mm]")

    block(by_age_band, "age_band", "Age band", True)
    block(by_gender, "gender", "Gender", False)

    if lines[-1] == r"\addlinespace[1mm]":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write(path, lines)


# ===========================================================================
# TABLE 3 -- central torso against the periphery
# ===========================================================================

def central_table(summary: pd.DataFrame, path: Path) -> Optional[Path]:
    """Where the hand spends its time, pooled over prefectures.

    Pooled by clip count, not by averaging the six prefecture percentages:
    an unweighted mean would give Gunma's 105 CL clips the same say as Toyama's
    1,664, which is not what "the corpus does X" means.
    """
    if summary.empty:
        return None

    frame = summary.copy()
    for column in ("central_percent", "periphery_percent", "extreme_percent", "n_clips"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["n_clips"])

    lines = [
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Hand} & \textbf{central torso (\%)} & "
        r"\textbf{periphery (\%)} & \textbf{extreme periphery (\%)} \\",
        r"\midrule",
    ]

    for position, keyword in enumerate(KEYWORD_ORDER):
        block = frame[frame["keyword"] == keyword]
        if block.empty:
            continue
        if position:
            lines.append(r"\addlinespace[1mm]")
        first = True
        for role in ("dominant", "non_dominant"):
            rows = block[block["hand_role"] == role]
            if rows.empty:
                continue
            weight = rows["n_clips"].sum() or 1
            central = (rows["central_percent"] * rows["n_clips"]).sum() / weight
            periphery = (rows["periphery_percent"] * rows["n_clips"]).sum() / weight
            extreme = (rows["extreme_percent"] * rows["n_clips"]).sum() / weight
            label = _keyword_label(keyword) if first else ""
            first = False
            lines.append(f"{label} & {HAND_LABEL[role]} & {central:.1f} & "
                         f"{periphery:.1f} & {extreme:.1f} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write(path, lines)


# ===========================================================================
# DRIVER
# ===========================================================================

#: Printed names for the grouping variables, in the order the report shows them.
GROUP_LABELS = [
    ("keyword", "Linguistic category"),
    ("region_code", "Prefecture"),
    ("age_band", "Age band"),
    ("gender", "Gender"),
]

#: What each grouping is compared *within*, for the separation table's caption.
CATEGORY_ORDER = ["CL", "LEXICAL_ITEM", "FS"]


def _level_name(group: str, level: object) -> str:
    """A level as the report prints it: prefecture names, category macros.

    Macros are emitted with a trailing ``{}`` so a following space survives --
    ``\\CL vs`` swallows it and typesets ``CLvs``.
    """
    if group == "region_code":
        return REGION_NAMES.get(str(level), str(level))
    if group == "keyword":
        label = _keyword_label(str(level))
        return label + "{}" if label.startswith("\\") else label
    # "<50" is a perfectly good age band and a perfectly bad LaTeX character.
    return str(level).replace("<", "$<$")


def separation_table(comparisons: pd.DataFrame, path: Path,
                     hand_role: str = "dominant") -> Optional[Path]:
    """How many level pairs separate, counted over reliable cells only.

    Generated rather than typed, because these counts move whenever the corpus,
    the exclusions or the region scheme change, and a hand-copied count in a
    typeset PDF goes stale without anything failing.
    """
    if comparisons.empty:
        return None

    rows = comparisons[comparisons["hand_role"] == hand_role]
    reliable = rows[(rows["n_signers_a"] >= MIN_SIGNERS_FOR_STABLE_CI)
                    & (rows["n_signers_b"] >= MIN_SIGNERS_FOR_STABLE_CI)]
    if reliable.empty:
        return None

    lines = ["\\begin{tabular}{@{}lrrl@{}}", "\\toprule",
             "\\textbf{Grouping variable} & \\textbf{separating} & "
             "\\textbf{pairs tested} & \\textbf{largest gap (regions/clip)} \\\\",
             "\\midrule"]
    for group, label in GROUP_LABELS:
        subset = reliable[reliable["group"] == group]
        if subset.empty:
            continue
        separating = int(subset["separates"].sum())
        widest = subset.loc[subset["difference"].abs().idxmax()]
        gap = (f"{abs(float(widest['difference'])):.2f} \\quad "
               f"{_level_name(group, widest['level_a'])} vs "
               f"{_level_name(group, widest['level_b'])}, "
               f"{_level_name('keyword', widest['compared_within'])}"
               if group != "keyword" else
               f"{abs(float(widest['difference'])):.2f} \\quad "
               f"{_level_name(group, widest['level_a'])} vs "
               f"{_level_name(group, widest['level_b'])}, "
               f"{REGION_NAMES.get(str(widest['compared_within']), widest['compared_within'])}")
        count = f"\\textbf{{{separating}}}" if separating > len(subset) / 2 else str(separating)
        lines.append(f"{label} & {count} & {len(subset)} & {gap} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return _write(path, lines)


def spread_table(spread: pd.DataFrame, path: Path,
                 hand_role: str = "dominant") -> Optional[Path]:
    """Highest level mean minus lowest, per grouping, in regions per clip."""
    if spread.empty:
        return None

    rows = spread[spread["hand_role"] == hand_role]
    if rows.empty:
        return None

    lines = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
             "\\textbf{What varies} & \\textbf{\\CL{}} & \\textbf{\\LEX{}} "
             "& \\textbf{\\FS{}} \\\\", "\\midrule"]

    # The category row spans the three columns: the categories are its levels,
    # so it cannot be split by category. Pooled over the corpus, it is the
    # CL-minus-FS gap, and it is the yardstick the other rows are read against.
    category = rows[rows["group"] == "keyword"]
    if not category.empty:
        pooled = float(category["spread"].mean())
        lines.append("Linguistic category & \\multicolumn{3}{c}"
                     f"{{\\textbf{{{pooled:.2f}}}}} \\\\")
        lines.append("\\midrule")

    for group, label in GROUP_LABELS:
        if group == "keyword":
            continue
        subset = rows[rows["group"] == group].set_index("compared_within")
        if subset.empty:
            continue
        cells = []
        for category_name in CATEGORY_ORDER:
            if category_name in subset.index:
                value = float(subset.loc[category_name, "spread"])
                cells.append(f"\\textbf{{{value:.2f}}}" if group == "region_code"
                             else f"{value:.2f}")
            else:
                cells.append("--")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return _write(path, lines)


def write_report_tables(tables_dir: Path, hand_role: str = "dominant") -> List[Path]:
    """Regenerate every .tex table from the CSVs already in ``tables_dir``."""
    tables_dir = Path(tables_dir)

    def read(name: str) -> pd.DataFrame:
        path = tables_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, encoding="utf-8-sig")

    written = [
        region_table(read("by_region_and_keyword.csv"), tables_dir / "tab_region.tex"),
        age_gender_table(read("by_age_band_and_keyword.csv"),
                         read("by_gender_and_keyword.csv"),
                         tables_dir / "tab_age_gender.tex", hand_role),
        central_table(read("central_periphery_summary.csv"),
                      tables_dir / "tab_central.tex"),
        separation_table(read("pairwise_differences.csv"),
                         tables_dir / "tab_separation.tex", hand_role),
        spread_table(read("effect_spread.csv"),
                     tables_dir / "tab_spread.tex", hand_role),
    ]
    return [path for path in written if path is not None]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_folder", type=Path,
                        help="the pipeline output folder (its tables/ is read)")
    parser.add_argument("--hand-role", default="dominant")
    args = parser.parse_args()

    for path in write_report_tables(Path(args.output_folder) / "tables",
                                    args.hand_role):
        print(f"  {path}")
