"""Figures for the corpus statistics.

Every figure is meant to go into the report or the poster unchanged, so they
share one visual identity: a validated light categorical palette, a recessive
grid, and **an exact number printed next to every bar, point and segment** --
these figures are read as tables as often as they are read as pictures.

Where a quantity splits into a confirmed and a contested part -- mouth labels
that other annotators agreed or disagreed with -- the contested part is drawn as
a lighter tint of the *same* hue with a hatch over it. Same hue because it is the
same quantity; hatch as well as tint so the distinction survives greyscale
printing and colour-vision deficiency.

"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch

from config import MOUTH_CATEGORIES
from io_utils import region_label
from mouth import LEXICAL_UNIT

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Slots 1-4 of a categorical palette validated for colour-vision deficiency
# (worst adjacent CVD dE 9.1, normal-vision dE 22.9). Three of the four sit
# below 3:1 against a white surface, which is acceptable here only because every
# mark in every figure carries a printed value beside it.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]

TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e5e4e0"
SURFACE = "#ffffff"

CATEGORY_COLOURS = {
    "Mouthing": BLUE,
    "MouthGesture": ORANGE,
    "Others": AQUA,
}

DPI = 200
HATCH = "///"

CJK_CANDIDATES = ["Noto Sans CJK JP", "Noto Serif CJK JP", "IPAexGothic",
                  "IPAGothic", "TakaoGothic", "VL Gothic", "Yu Gothic",
                  "Hiragino Sans", "MS Gothic"]


def _available_cjk_font() -> Optional[str]:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in installed:
            return name
    return None


CJK_FONT = _available_cjk_font()


def tint(colour: str, amount: float = 0.62) -> str:
    """Mix a colour towards white. Used for the contested half of a split bar."""
    rgb = np.array(to_rgb(colour))
    return to_hex(rgb + (np.array([1.0, 1.0, 1.0]) - rgb) * float(amount))


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
    })
    if CJK_FONT:
        plt.rcParams["font.family"] = ["DejaVu Sans", CJK_FONT]
        plt.rcParams["axes.unicode_minus"] = False


def _save(figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return path


def _tidy(axes) -> None:
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)


def _number(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:g}"


def _headroom(axes, values, factor: float = 1.22, axis: str = "x") -> None:
    """Leave room for the printed values so they never run off the canvas."""
    largest = float(max(values)) if len(values) else 0.0
    if largest <= 0:
        return
    if axis == "x":
        axes.set_xlim(0, largest * factor)
    else:
        axes.set_ylim(0, largest * factor)


def _label_h(axes, y_positions, widths, texts, pad: float, size: float = 7.5) -> None:
    """Print a value to the right of each horizontal bar."""
    for y, width, text in zip(y_positions, widths, texts):
        axes.text(width + pad, y, text, va="center", ha="left",
                  fontsize=size, color=TEXT)


# ===========================================================================
# OVERVIEW
# ===========================================================================

def plot_overview_tiles(summary: pd.DataFrame, path: Path) -> Optional[Path]:
    """Three headline numbers as tiles.

    Not a chart: three unrelated scalars have no shared scale, so any bar chart
    of them would invite a comparison that means nothing.
    """
    if summary.empty:
        return None
    row = summary[summary["tag"] == "GLOBAL"].iloc[0] if "GLOBAL" in set(summary["tag"]) \
        else summary.iloc[0]

    tiles = [(f"{int(row['n_files_parsed']):,}", "Files processed"),
             (str(row["total_recording_hms"]), "Recording time"),
             (f"{int(row['n_signers']):,}", "Signers")]

    figure, axes_row = plt.subplots(1, 3, figsize=(9.0, 2.0))
    figure.suptitle("Corpus overview", fontsize=11, color=TEXT, y=1.04)
    for axes, (value, caption) in zip(axes_row, tiles):
        axes.set_axis_off()
        axes.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.90, transform=axes.transAxes,
                                     facecolor=SURFACE, edgecolor=GRID, linewidth=1.0))
        axes.text(0.5, 0.62, value, transform=axes.transAxes, ha="center",
                  va="center", fontsize=26, fontweight="bold", color=TEXT)
        axes.text(0.5, 0.22, caption, transform=axes.transAxes, ha="center",
                  va="center", fontsize=9, color=MUTED)
    return _save(figure, path)


def plot_annotation_breakdown(summary: pd.DataFrame, path: Path) -> Optional[Path]:
    """The parser's outcome for the whole corpus, as counts and shares.

    Rows are nested rather than exclusive -- "with a lexical item" is a subset of
    "parsed" -- so this is deliberately a list of measurements against one
    denominator, not a partition. Every share is of the total.
    """
    if summary.empty:
        return None
    row = summary[summary["tag"] == "GLOBAL"].iloc[0] if "GLOBAL" in set(summary["tag"]) \
        else summary.iloc[0]

    total = int(row["n_annotations"]) or 1
    entries = [
        ("Total annotations", int(row["n_annotations"])),
        ("Successfully parsed", int(row["n_parsed"])),
        ("With a lexical item", int(row["n_with_lexical"])),
        ("Lexical item only (no marker)", int(row["n_lexical_only"])),
        ("Unique lexical items", int(row["n_unique_lexical_items"])),
        ("Compound", int(row["n_compound"])),
        ("Ambiguous", int(row["n_ambiguous"])),
        ("Marker only (no lexical item)", int(row["n_key_only"])),
    ]

    figure, axes = plt.subplots(figsize=(6.6, 0.36 * len(entries) + 0.9))
    y = np.arange(len(entries))[::-1]
    values = [count for _label, count in entries]
    axes.barh(y, values, color=BLUE, height=0.62)

    texts = [f"{count:,}" if index == 0 else f"{count:,}  ({100 * count / total:.1f}%)"
             for index, (_label, count) in enumerate(entries)]
    _headroom(axes, values, 1.30)
    _label_h(axes, y, values, texts, pad=total * 0.012, size=8)

    axes.set_yticks(y)
    axes.set_yticklabels([label for label, _count in entries], fontsize=8.5)
    axes.set_xlabel("annotations")
    axes.set_title("Annotation breakdown")
    axes.grid(axis="y", visible=False)
    _tidy(axes)
    return _save(figure, path)


# ===========================================================================
# VOCABULARY SHAPE
# ===========================================================================

def plot_coverage_curve(curve: pd.DataFrame, path: Path,
                        cutoffs: Sequence[int] = (100, 200, 500, 900)) -> Optional[Path]:
    """Cumulative token share against vocabulary rank.

    The marked cutoffs are the vocabularies in the coverage table, so the figure
    and that table can be read against each other. Log rank on the x-axis: all
    the interesting behaviour is in the first few hundred glosses, and a linear
    axis would compress it into the left margin.
    """
    if curve.empty:
        return None
    figure, axes = plt.subplots(figsize=(6.0, 3.6))
    axes.plot(curve["rank"], curve["cumulative_percent"], color=BLUE, linewidth=2.0)
    axes.fill_between(curve["rank"], curve["cumulative_percent"],
                      color=BLUE, alpha=0.10)

    total = len(curve)
    # On a log axis the marked cutoffs bunch together near the right-hand end,
    # so labelling each point in place guarantees collisions. The readings go in
    # a block in the upper-left instead -- the one region a rising cumulative
    # curve always leaves empty.
    readings = []
    for cutoff in cutoffs:
        if cutoff > total:
            continue
        value = float(curve.loc[curve["rank"] == cutoff, "cumulative_percent"].iloc[0])
        axes.vlines(cutoff, 0, value, color=MUTED, linewidth=0.7, linestyle=":")
        axes.plot([cutoff], [value], "o", color=ORANGE, markersize=6, zorder=5)
        readings.append(f"top {cutoff:,} glosses{'':<2} {value:5.1f}%")
    readings.append(f"all {total:,} glosses{'':<2} 100.0%")

    axes.text(0.03, 0.97, "\n".join(readings), transform=axes.transAxes,
              va="top", ha="left", fontsize=8.5, color=TEXT, linespacing=1.5,
              family="DejaVu Sans Mono" if not CJK_FONT else None,
              bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE,
                        edgecolor=GRID, linewidth=1.0))

    axes.set_xscale("log")
    axes.set_xlabel("vocabulary size (unique lexical items, ranked by frequency)\n"
                    f"the whole vocabulary is {total:,} items and covers 100%")
    axes.set_ylabel("share of all gloss tokens (%)")
    axes.set_ylim(0, 112)
    axes.set_title("How much of the corpus a fixed vocabulary covers")
    _tidy(axes)
    return _save(figure, path)


#: One-hue sequential ramp, light to dark, from the same palette as the
#: categorical slots. Used only for magnitude, never for identity.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
RAMP_DARK_TEXT_LIMIT = 6          # steps at or above this index need white text


def _ramp_colour(value: float, largest: float) -> tuple:
    """A ramp step for a count, on a log scale, plus the text colour to use.

    Log rather than linear because the counts span three orders of magnitude:
    on a linear ramp every cell but the largest would be the palest step.
    """
    if value <= 0 or largest <= 0:
        return "#f4f4f2", MUTED
    position = np.log10(value) / np.log10(largest)
    index = int(round(position * (len(BLUE_RAMP) - 1)))
    index = max(0, min(len(BLUE_RAMP) - 1, index))
    return BLUE_RAMP[index], ("white" if index >= RAMP_DARK_TEXT_LIMIT else TEXT)


def plot_class_sizes(class_sizes: pd.DataFrame, path: Path) -> Optional[Path]:
    """How many glosses could serve as classes, at each pair of floors.

    A grid, not a line chart. The question is a lookup -- "at 20 examples and 5
    signers, how many classes do I have?" -- and the four signer floors converge
    onto one another above about 20 examples, so plotted as lines three of them
    are hidden under the fourth exactly where a reader would try to read them.

    Each cell carries the gloss count and, beneath it, the share of all gloss
    tokens those glosses account for: a vocabulary can look small and still cover
    most of the data, and that pair of numbers is the whole trade-off.
    """
    if class_sizes.empty:
        return None
    if "tag" in class_sizes.columns and "GLOBAL" in set(class_sizes["tag"]):
        class_sizes = class_sizes[class_sizes["tag"] == "GLOBAL"]
    if class_sizes.empty:
        return None

    examples = sorted(class_sizes["min_examples"].unique())
    signers = sorted(class_sizes["min_signers"].unique())
    counts = class_sizes.set_index(["min_signers", "min_examples"])

    figure, axes = plt.subplots(figsize=(1.15 * len(examples) + 1.9,
                                         0.86 * len(signers) + 2.0))
    largest = float(class_sizes["n_glosses"].max()) or 1.0

    for row, floor in enumerate(signers):
        for column, threshold in enumerate(examples):
            try:
                entry = counts.loc[(floor, threshold)]
            except KeyError:
                continue
            value = int(entry["n_glosses"])
            share = float(entry.get("percent_of_tokens", float("nan")))
            colour, ink = _ramp_colour(value, largest)

            axes.add_patch(plt.Rectangle((column, row), 0.94, 0.90,
                                         facecolor=colour, edgecolor=SURFACE,
                                         linewidth=2.0))
            axes.text(column + 0.47, row + 0.56, f"{value:,}", ha="center",
                      va="center", fontsize=11, fontweight="bold", color=ink)
            if not np.isnan(share):
                axes.text(column + 0.47, row + 0.26, f"{share:.0f}% of tokens",
                          ha="center", va="center", fontsize=7, color=ink)

    axes.set_xlim(-0.06, len(examples))
    axes.set_ylim(-0.06, len(signers))
    axes.set_xticks([c + 0.47 for c in range(len(examples))])
    axes.set_xticklabels([f"≥ {e:,}" for e in examples], fontsize=9)
    axes.set_yticks([r + 0.45 for r in range(len(signers))])
    axes.set_yticklabels([f"≥ {s} signer{'s' if s != 1 else ''}" for s in signers],
                         fontsize=9)
    axes.set_xlabel("examples of the gloss in the corpus")
    axes.set_ylabel("different signers who produced it")
    axes.set_title("How many glosses could be used as classes\n"
                   "each cell: glosses meeting both floors, and the share of all "
                   "gloss tokens they cover", fontsize=10.5)
    axes.grid(False)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(length=0)
    return _save(figure, path)


def plot_vocabulary_coverage(coverage: pd.DataFrame, path: Path) -> Optional[Path]:
    """The coverage table as a picture: what each candidate vocabulary buys.

    One row per candidate vocabulary. The pair of bars is the same quantity
    twice -- all the tokens those glosses account for, and the tokens left after
    capping each gloss at ``cap`` -- so the capped bar is a lighter tint of the
    same hue rather than a second colour. The gap between them is the imbalance
    a training set would have to absorb.

    The row label carries the number that decides whether the vocabulary is real:
    how many glosses were actually found. "Top 500" that yields 496 means the
    corpus ran out of glosses meeting the signer floor before the cutoff.
    """
    if coverage.empty:
        return None
    frame = coverage[coverage["scope"] == "GLOBAL"] if "scope" in coverage.columns \
        and "GLOBAL" in set(coverage["scope"]) else coverage
    if frame.empty:
        return None
    frame = frame.iloc[::-1]                      # first spec at the top

    figure, axes = plt.subplots(figsize=(9.0, 0.80 * len(frame) + 1.8))
    y = np.arange(len(frame), dtype=float)
    height = 0.34

    raw = frame["total_occurrences"].to_numpy(dtype=float)
    capped = frame["total_occurrences_capped"].to_numpy(dtype=float)
    cap = int(frame["cap"].iloc[0]) if "cap" in frame.columns else 0

    axes.barh(y + height / 2, raw, height=height, color=BLUE,
              label="all occurrences")
    axes.barh(y - height / 2, capped, height=height, color=tint(BLUE, 0.55),
              edgecolor=BLUE, linewidth=0.8,
              label=f"capped at {cap:,} per gloss" if cap else "capped")

    pad = raw.max() * 0.012
    for position, (_, row) in zip(y, frame.iterrows()):
        axes.text(row["total_occurrences"] + pad, position + height / 2,
                  f"{int(row['total_occurrences']):,}"
                  f"   ({row['percent_of_corpus_tokens']:.1f}% of all gloss tokens)",
                  va="center", fontsize=8, color=TEXT)
        axes.text(row["total_occurrences_capped"] + pad, position - height / 2,
                  f"{int(row['total_occurrences_capped']):,}", va="center",
                  fontsize=8, color=MUTED)

    labels = []
    for _, row in frame.iterrows():
        requested = "".join(c for c in str(row["group"]) if c.isdigit())
        found = int(row["n_glosses"])
        if requested and found < int(requested):
            got = f"{found:,} of {int(requested):,} requested"
        else:
            got = f"{found:,} glosses"
        labels.append(f"{row['group']}\n≥ {int(row['min_signers'])} signers\n{got}")

    axes.set_yticks(y)
    axes.set_yticklabels(labels, fontsize=8.5)
    _headroom(axes, raw, 1.55)
    axes.set_xlabel("gloss tokens covered by the vocabulary")
    axes.set_title("What each candidate vocabulary covers", pad=28)
    axes.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.005))
    axes.grid(axis="y", visible=False)
    _tidy(axes)
    return _save(figure, path)


def plot_top_glosses(top: pd.DataFrame, path: Path, n: int = 50) -> Optional[Path]:
    """The most frequent glosses, with occurrences and signer counts printed."""
    if top.empty or not CJK_FONT:
        return None
    subset = top.head(int(n)).iloc[::-1]
    figure, axes = plt.subplots(figsize=(6.2, 0.26 * len(subset) + 1.2))

    values = subset["occurrences"].to_numpy(dtype=float)
    y = np.arange(len(subset))
    axes.barh(y, values, color=BLUE, height=0.70)

    texts = [f"{int(v):,}  ({int(s)} signers)"
             for v, s in zip(values, subset["n_signers"])]
    _headroom(axes, values, 1.34)
    _label_h(axes, y, values, texts, pad=values.max() * 0.012)

    axes.set_yticks(y)
    axes.set_yticklabels(subset["gloss"], fontsize=8.5)
    axes.set_xlabel("occurrences")
    axes.set_title(f"{len(subset)} most frequent lexical items")
    axes.grid(axis="y", visible=False)
    _tidy(axes)
    return _save(figure, path)


# ===========================================================================
# PARSING OUTCOME
# ===========================================================================

def plot_region_outcome(summary: pd.DataFrame, path: Path) -> Optional[Path]:
    """Parsed / ambiguous / compound share, one stacked bar per prefecture.

    These three do partition the annotations, so stacking is honest here.
    """
    if summary.empty:
        return None
    regions = summary[summary["tag"] != "GLOBAL"].copy()
    if regions.empty:
        regions = summary.copy()
    regions = regions.sort_values("n_annotations", ascending=False)
    labels = [f"{region_label(t)}\n{int(n):,} annot." for t, n
              in zip(regions["tag"], regions["n_annotations"])]

    figure, axes = plt.subplots(figsize=(1.15 * len(regions) + 2.2, 3.8))
    x = np.arange(len(regions))
    bottom = np.zeros(len(regions))
    for column, colour, name in [("parsed_percent", BLUE, "parsed"),
                                 ("ambiguous_percent", ORANGE, "ambiguous"),
                                 ("compound_percent", AQUA, "compound")]:
        values = regions[column].to_numpy(dtype=float)
        axes.bar(x, values, bottom=bottom, color=colour, label=name, width=0.60)
        for position, value, base in zip(x, values, bottom):
            if value >= 4:
                axes.text(position, base + value / 2, f"{value:.1f}%",
                          ha="center", va="center", fontsize=7.5, color="white")
            elif value > 0:
                axes.text(position + 0.34, base + value / 2, f"{value:.1f}%",
                          ha="left", va="center", fontsize=7, color=colour)
        bottom += values

    axes.set_xticks(x)
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_ylabel("share of annotations (%)")
    axes.set_ylim(0, 104)
    axes.set_title("Parsing outcome by prefecture", pad=12)
    axes.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    axes.grid(axis="x", visible=False)
    _tidy(axes)
    return _save(figure, path)


def plot_key_frequency(keys: pd.DataFrame, path: Path) -> Optional[Path]:
    """Marker frequency, as a count and as a share of parsed annotations.

    Two panels rather than one axis with two scales: a dual-axis chart invites
    the reader to compare two quantities that have no common unit. The panels
    share a row order, so a marker is on the same line in both.

    Blocking markers are drawn in a different hue: they say the annotation could
    not be read, which is a fact about the recording rather than the language.
    """
    if keys.empty:
        return None
    subset = keys[keys["tag"] == "GLOBAL"] if "GLOBAL" in set(keys["tag"]) else keys
    unused = sorted(subset.loc[subset["n_annotations"] == 0, "key"].str.upper())
    subset = subset[subset["n_annotations"] > 0].sort_values("n_annotations")
    if subset.empty:
        return None

    colours = [ORANGE if kind == "blocking" else BLUE for kind in subset["kind"]]
    y = np.arange(len(subset))
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(8.4, 0.26 * len(subset) + 1.6), sharey=True)

    counts = subset["n_annotations"].to_numpy(dtype=float)
    left.barh(y, counts, color=colours, height=0.70)
    _headroom(left, counts, 1.22)
    _label_h(left, y, counts, [f"{int(v):,}" for v in counts], pad=counts.max() * 0.015)
    left.set_yticks(y)
    left.set_yticklabels([k.upper() for k in subset["key"]], fontsize=8)
    left.set_xlabel("annotations carrying the marker")
    left.set_title("Count", fontsize=9.5)
    left.grid(axis="y", visible=False)
    _tidy(left)

    shares = subset["percent_of_parsed"].to_numpy(dtype=float)
    right.barh(y, shares, color=colours, height=0.70)
    _headroom(right, shares, 1.22)
    _label_h(right, y, shares, [f"{v:.2f}%" for v in shares], pad=shares.max() * 0.015)
    right.set_xlabel("share of successfully parsed annotations")
    right.set_title("Percentage", fontsize=9.5)
    right.grid(axis="y", visible=False)
    _tidy(right)

    title = "Marker frequency"
    if unused:
        title += "\n" + textwrap.fill(f"never used: {', '.join(unused)}", width=78)
    figure.suptitle(title, fontsize=11, color=TEXT, y=1.0)
    figure.legend(handles=[Patch(facecolor=BLUE, label="linguistic marker"),
                           Patch(facecolor=ORANGE, label="unreadable-annotation marker")],
                  ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    return _save(figure, path)


# ===========================================================================
# MOUTH ACTION
# ===========================================================================

def _split_bar_legend(axes, **kwargs) -> None:
    axes.legend(handles=[
        Patch(facecolor=MUTED, label="agreed by other annotators"),
        Patch(facecolor=tint(MUTED), hatch=HATCH, edgecolor=TEXT,
              label="disagreed"),
    ], **kwargs)


def plot_mouth_category_counts(categories: pd.DataFrame, path: Path,
                               n_files: Optional[int] = None) -> Optional[Path]:
    """Mouth labels per category, split into agreed and disagreed.

    The split is over *labels*, not annotations, so the bar heights are the raw
    inventory of MouthAction annotations in the corpus.
    """
    if categories.empty:
        return None
    subset = categories[categories["tag"] == "GLOBAL"] if "GLOBAL" in set(categories["tag"]) \
        else categories
    if subset.empty:
        return None

    figure, axes = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(subset))
    for position, (_, row) in zip(x, subset.iterrows()):
        colour = CATEGORY_COLOURS.get(row["category"], BLUE)
        agreed = float(row["n_agreed"])
        disagreed = float(row["n_disagreed"])
        axes.bar(position, agreed, color=colour, width=0.56)
        axes.bar(position, disagreed, bottom=agreed, color=tint(colour),
                 width=0.56, hatch=HATCH, edgecolor=colour, linewidth=0.8)
        axes.text(position, agreed + disagreed,
                  f"{int(row['n_labels']):,} labels\n"
                  f"{row['percent_agreed']:.1f}% agreed\n"
                  f"{row['percent_disagreed']:.1f}% disagreed",
                  ha="center", va="bottom", fontsize=8, color=TEXT)

    axes.set_xticks(x)
    axes.set_xticklabels(subset["category"], fontsize=9.5)
    axes.set_ylabel("MouthAction labels")
    _headroom(axes, subset["n_labels"], 1.42, axis="y")
    title = "MouthAction labels by category"
    if n_files:
        title += f" ({n_files} files with MouthAction tiers)"
    axes.set_title(title)
    _split_bar_legend(axes, loc="upper right")
    axes.grid(axis="x", visible=False)
    _tidy(axes)
    return _save(figure, path)


def _unit_label(unit: str) -> str:
    """Marker names read better upper-cased; prose row names are left alone."""
    if unit.startswith("lexical"):
        return "lexical item, no marker"
    if unit == "any annotation":
        return "all annotations"
    return unit if " " in unit else unit.upper()


def plot_mouth_key_category(key_categories: pd.DataFrame, path: Path,
                            min_labels: int = 20,
                            max_units: int = 12) -> Optional[Path]:
    """For each marker, what the mouth was doing, and whether annotators agreed.

    Three rows per marker -- one per mouth category -- with the contested part of
    each bar hatched. The printed text carries what the bar cannot: the agreed
    and disagreed split, and how many of the marker's own annotations that
    category reached.
    """
    if key_categories.empty:
        return None
    frame = key_categories[key_categories["tag"] == "GLOBAL"] \
        if "GLOBAL" in set(key_categories["tag"]) else key_categories
    # Markers only. The two summary units -- every annotation, and bare lexical
    # items -- are five to ten times larger than any single marker, and putting
    # them on the same axis squeezes every marker bar into the left margin. They
    # stay in mouth_overlap.csv and in the LaTeX table, where the numbers are
    # read rather than compared by length.
    frame = frame[~frame["unit"].isin(["any annotation", LEXICAL_UNIT])]
    if frame.empty:
        return None

    ranked = (frame.groupby("unit")["n_labels"].sum()
              .sort_values(ascending=False))
    units = [u for u in ranked.index if ranked[u] >= int(min_labels)][:int(max_units)]
    if not units:
        return None

    rows: List[tuple] = []
    for unit in units:
        for category in MOUTH_CATEGORIES:
            match = frame[(frame["unit"] == unit) & (frame["category"] == category)]
            if match.empty:
                continue
            rows.append((unit, category, match.iloc[0]))

    figure, axes = plt.subplots(figsize=(9.0, 0.30 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    widest = max(float(r["n_labels"]) for _u, _c, r in rows) or 1.0

    for position, (unit, category, row) in zip(y, rows):
        colour = CATEGORY_COLOURS.get(category, BLUE)
        agreed = float(row["n_labels_agreed"])
        disagreed = float(row["n_labels_disagreed"])
        axes.barh(position, agreed, color=colour, height=0.68)
        axes.barh(position, disagreed, left=agreed, color=tint(colour),
                  height=0.68, hatch=HATCH, edgecolor=colour, linewidth=0.8)
        axes.text(agreed + disagreed + widest * 0.012, position,
                  f"{int(row['n_labels']):,} labels "
                  f"({int(agreed):,} agreed / {int(disagreed):,} disagreed); "
                  f"{int(row['n_rows']):,} of {int(row['n_unit_rows']):,} rows "
                  f"({row['percent_of_rows']:.1f}%)",
                  va="center", ha="left", fontsize=7.5, color=TEXT)

    # A faint rule between markers, so the three category rows read as a group.
    for index in range(3, len(rows), 3):
        axes.axhline(y[index] + 0.5, color=GRID, linewidth=0.8)

    axes.set_yticks(y)
    axes.set_yticklabels([f"{_unit_label(u)} — {c}" for u, c, _r in rows], fontsize=8)
    axes.set_xlim(0, widest * 2.20)
    axes.set_xlabel("overlapping MouthAction labels")
    axes.set_title("Mouth action co-occurring with each marker")
    _split_bar_legend(axes, loc="lower right")
    axes.grid(axis="y", visible=False)
    _tidy(axes)
    return _save(figure, path)


# ===========================================================================
# TIMING AND SIGNERS
# ===========================================================================

def plot_duration_distribution(durations: pd.DataFrame, path: Path,
                               min_n: int = 30) -> Optional[Path]:
    """Median duration with the 25-75 and 5-95 ranges, per marker.

    Drawn from percentiles rather than as a box plot because the underlying
    counts differ by orders of magnitude between markers; a box plot would invite
    the reader to compare spreads that rest on very different evidence. The count
    behind each row is printed for the same reason.
    """
    if durations.empty:
        return None
    if "tag" in durations.columns and "GLOBAL" in set(durations["tag"]):
        durations = durations[durations["tag"] == "GLOBAL"]
    subset = durations[durations["n"] >= int(min_n)].sort_values("p50_ms")
    if subset.empty:
        return None

    figure, axes = plt.subplots(figsize=(7.0, 0.30 * len(subset) + 1.3))
    y = np.arange(len(subset))
    axes.hlines(y, subset["p5_ms"], subset["p95_ms"], color=tint(BLUE, 0.45),
                linewidth=1.6)
    axes.hlines(y, subset["p25_ms"], subset["p75_ms"], color=BLUE, linewidth=5.0)
    axes.plot(subset["p50_ms"], y, "o", color=SURFACE, markersize=4.0,
              markeredgecolor=BLUE, markeredgewidth=1.4, zorder=5)

    for position, (_, row) in zip(y, subset.iterrows()):
        axes.text(row["p95_ms"] + subset["p95_ms"].max() * 0.015, position,
                  f"median {int(row['p50_ms']):,} ms   (n = {int(row['n']):,})",
                  va="center", ha="left", fontsize=7.5, color=TEXT)

    axes.set_yticks(y)
    axes.set_yticklabels([_unit_label(u) for u in subset["unit"]], fontsize=8)
    _headroom(axes, subset["p95_ms"], 1.62)
    axes.set_xlabel("annotation duration (ms) — median, 25–75%, 5–95%")
    axes.set_title("How long an annotated segment lasts")
    axes.grid(axis="y", visible=False)
    _tidy(axes)
    return _save(figure, path)


def plot_signer_balance(balance: pd.DataFrame, path: Path) -> Optional[Path]:
    """Cumulative token share by signer, against an even-contribution line."""
    if balance.empty:
        return None
    figure, axes = plt.subplots(figsize=(6.0, 3.6))
    n = len(balance)
    axes.plot(balance["rank"], balance["cumulative_percent"],
              color=BLUE, linewidth=2.0, label="observed")
    axes.plot([1, n], [100 / n, 100], color=MUTED, linewidth=1.0,
              linestyle="--", label="even contribution")

    for fraction in (0.25, 0.5):
        rank = max(1, int(round(n * fraction)))
        value = float(balance.loc[balance["rank"] == rank, "cumulative_percent"].iloc[0])
        axes.plot([rank], [value], "o", color=ORANGE, markersize=6, zorder=5)
        axes.annotate(f"top {rank} of {n} signers\n{value:.1f}% of tokens",
                      (rank, value), textcoords="offset points", xytext=(8, -18),
                      fontsize=8, color=TEXT)

    axes.set_xlabel("signers, most productive first")
    axes.set_ylabel("cumulative share of gloss tokens (%)")
    axes.set_ylim(0, 104)
    axes.set_title("How evenly the corpus is sampled across signers")
    axes.legend(loc="lower right")
    _tidy(axes)
    return _save(figure, path)


def plot_region_vocabulary(summary: pd.DataFrame, path: Path) -> Optional[Path]:
    """Lexical variety per prefecture, normalised by how much each contributed.

    Vocabulary size on its own only restates corpus size -- the prefecture with
    the most annotations will have the largest vocabulary. Dividing by tokens
    gives the quantity that actually differs between prefectures, and it is the
    one a modeller feels directly: more variety per token means fewer examples
    per class.

    A scatter of vocabulary against tokens says the same thing, but seven
    prefectures of similar size put their labels on top of each other; bars keep
    every number legible.
    """
    regions = summary[summary["tag"] != "GLOBAL"].copy()
    if regions.empty:
        return None

    regions["per_1000"] = (1000 * regions["n_unique_lexical_items"]
                           / regions["n_parsed"].replace(0, np.nan))
    regions = regions.dropna(subset=["per_1000"]).sort_values("per_1000")
    if regions.empty:
        return None

    figure, axes = plt.subplots(figsize=(6.8, 0.38 * len(regions) + 1.3))
    y = np.arange(len(regions))
    values = regions["per_1000"].to_numpy(dtype=float)
    axes.barh(y, values, color=BLUE, height=0.64)

    texts = [f"{value:.1f}   ({int(v):,} glosses in {int(t):,} parsed annotations)"
             for value, v, t in zip(values, regions["n_unique_lexical_items"],
                                    regions["n_parsed"])]
    _headroom(axes, values, 1.95)
    _label_h(axes, y, values, texts, pad=values.max() * 0.02)

    axes.set_yticks(y)
    axes.set_yticklabels([region_label(t) for t in regions["tag"]], fontsize=9)
    axes.set_xlabel("unique lexical items per 1,000 parsed annotations")
    axes.set_title("Lexical variety by prefecture")
    axes.grid(axis="y", visible=False)
    _tidy(axes)
    return _save(figure, path)


def missing_font_note() -> Optional[str]:
    """Warning text when no CJK font is installed, else None."""
    if CJK_FONT:
        return None
    return ("No CJK font found, so figures containing Japanese glosses were "
            "skipped. Install one, e.g. `apt-get install fonts-noto-cjk` or "
            "`pip install japanize-matplotlib`, and rerun step 5.")
