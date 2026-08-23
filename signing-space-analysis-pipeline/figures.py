"""Figures, all carrying signer-level uncertainty.

Two kinds are produced:

* a **body map** per keyword: the signing space drawn to scale, each region
  shaded by the share of hand points it received, annotated with the share and
  its 95% bootstrap interval;
* **comparison bars** of the average number of regions used per clip, by
  geographical region and by age group, with interval whiskers.

Matplotlib rather than hand-placed OpenCV pixels: error bars, log-free axes and
vector output come for free, and the figures are editable for a poster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import MIN_SIGNERS_FOR_STABLE_CI, REGION_KEYS, REGION_LABELS

# A schematic body map, drawn in figure-normalised units. This mirrors the
# anatomical bands used by the classifier without depending on any one clip's
# measurements, so every keyword is drawn on the same frame of reference.
X = {"xl": 0.00, "pl": 0.28, "cl": 0.40, "cr": 0.60, "pr": 0.72, "xr": 1.00}
Y = {"yt": 0.00, "head": 0.16, "chin": 0.33, "mid": 0.515, "hip": 0.70,
     "low": 0.84, "yb": 1.00}

MAP_RECTS: Dict[str, Tuple[float, float, float, float]] = {
    "ep_upper_left": (X["xl"], Y["yt"], X["cl"], Y["head"]),
    "ep_upper_center": (X["cl"], Y["yt"], X["cr"], Y["head"]),
    "ep_upper_right": (X["cr"], Y["yt"], X["xr"], Y["head"]),
    "ep_left_upper": (X["xl"], Y["head"], X["pl"], Y["chin"]),
    "ep_left_upper_torso": (X["xl"], Y["chin"], X["pl"], Y["mid"]),
    "ep_left_lower_torso": (X["xl"], Y["mid"], X["pl"], Y["hip"]),
    "ep_left_lower": (X["xl"], Y["hip"], X["pl"], Y["low"]),
    "p_upper_left": (X["pl"], Y["head"], X["cl"], Y["chin"]),
    "p_upper_center": (X["cl"], Y["head"], X["cr"], Y["chin"]),
    "p_upper_right": (X["cr"], Y["head"], X["pr"], Y["chin"]),
    "p_left_upper_torso": (X["pl"], Y["chin"], X["cl"], Y["mid"]),
    "p_left_lower_torso": (X["pl"], Y["mid"], X["cl"], Y["hip"]),
    "upper_torso": (X["cl"], Y["chin"], X["cr"], Y["mid"]),
    "lower_torso": (X["cl"], Y["mid"], X["cr"], Y["hip"]),
    "p_right_upper_torso": (X["cr"], Y["chin"], X["pr"], Y["mid"]),
    "p_right_lower_torso": (X["cr"], Y["mid"], X["pr"], Y["hip"]),
    "p_lower_left": (X["pl"], Y["hip"], X["cl"], Y["low"]),
    "p_lower_center": (X["cl"], Y["hip"], X["cr"], Y["low"]),
    "p_lower_right": (X["cr"], Y["hip"], X["pr"], Y["low"]),
    "ep_right_upper": (X["pr"], Y["head"], X["xr"], Y["chin"]),
    "ep_right_upper_torso": (X["pr"], Y["chin"], X["xr"], Y["mid"]),
    "ep_right_lower_torso": (X["pr"], Y["mid"], X["xr"], Y["hip"]),
    "ep_right_lower": (X["pr"], Y["hip"], X["xr"], Y["low"]),
    "ep_lower_left": (X["xl"], Y["low"], X["cl"], Y["yb"]),
    "ep_lower_center": (X["cl"], Y["low"], X["cr"], Y["yb"]),
    "ep_lower_right": (X["cr"], Y["low"], X["xr"], Y["yb"]),
}

# Region names are in the SIGNER's frame, but a body map is conventionally
# drawn as if facing the signer, so the signer's right belongs on the
# viewer's left. Mirroring the x extents here does that without touching a
# single name, so the figure and the CSVs can never disagree.
MAP_RECTS = {
    name: (1.0 - x2, y1, 1.0 - x1, y2)
    for name, (x1, y1, x2, y2) in MAP_RECTS.items()
}

ROLE_COLOURS = {"dominant": "#2A4B7C", "non_dominant": "#B07A1E"}

#: How far the busiest region is shaded toward the role colour. Near 1.0 the map
#: uses the full range from white to saturated, which is what makes the pattern
#: readable across a room. Dark cells would swallow black text, so the label
#: colour flips with the background (see :func:`_text_colour`) rather than the
#: shading being held back.
MAX_SHADE = 0.92

#: Below 1.0 this lifts the low end of the scale, so a region with a few percent
#: is visibly not empty. Without it a single dominant region flattens everything
#: else to white and the map looks emptier than the data is.
SHADE_GAMMA = 0.65

#: Luminance below which a cell counts as dark and its text is drawn white.
DARK_TEXT_THRESHOLD = 0.55
ROLE_LABELS = {"dominant": "dominant hand", "non_dominant": "non-dominant hand"}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _shade(value: float, vmax: float, colour: str) -> tuple:
    """Blend white toward the role colour in proportion to the share."""
    rgb = np.array(matplotlib.colors.to_rgb(colour))
    fraction = 0.0 if vmax <= 0 else float(np.clip(value / vmax, 0.0, 1.0))
    weight = (fraction ** SHADE_GAMMA) * MAX_SHADE
    return tuple(np.ones(3) * (1 - weight) + rgb * weight)


def _text_colour(background: tuple) -> tuple:
    """Black on a light cell, white on a dark one.

    Perceived luminance, not the mean of the channels: blue reads far darker
    than green at the same numeric value, and the role colours are a blue and
    an amber.
    """
    red, green, blue = background[:3]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return (1.0, 1.0, 1.0) if luminance < DARK_TEXT_THRESHOLD else (0.0, 0.0, 0.0)


def draw_body_map(
    ax,
    distribution: pd.DataFrame,
    role: str,
    title: str,
    font_scale: float = 1.0,
) -> None:
    """Shade each region by its share, annotated with the bootstrap interval."""
    subset = distribution[distribution["hand_role"] == role].set_index("region")
    vmax = float(subset["percent"].max()) if len(subset) else 1.0
    colour = ROLE_COLOURS.get(role, "#2A4B7C")

    for region, (x1, y1, x2, y2) in MAP_RECTS.items():
        percent = float(subset.loc[region, "percent"]) if region in subset.index else 0.0
        low = float(subset.loc[region, "ci_low"]) if region in subset.index else np.nan
        high = float(subset.loc[region, "ci_high"]) if region in subset.index else np.nan

        face = _shade(percent, vmax, colour)
        ink = _text_colour(face)
        muted = tuple(0.35 + 0.65 * channel for channel in ink)   # softer, same side

        ax.add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            facecolor=face, edgecolor="#333333", linewidth=0.7,
        ))

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # Wrap to the cell: the side-periphery cells are a third the width of
        # the central column, and "P right upper torso" does not fit on one
        # line in one of them.
        label = "\n".join(textwrap.wrap(
            REGION_LABELS.get(region, region),
            width=max(8, int((x2 - x1) / 0.0115)),
        )) or region

        ax.text(cx, cy - 0.022, label, ha="center", va="center",
                fontsize=8.5 * font_scale, color=muted, linespacing=1.25)
        ax.text(cx, cy + 0.020, f"{percent:.1f}%", ha="center", va="center",
                fontsize=13 * font_scale, fontweight="bold", color=ink)
        if np.isfinite(low) and np.isfinite(high):
            ax.text(cx, cy + 0.049, f"[{low:.1f}, {high:.1f}]", ha="center",
                    va="center", fontsize=7.5 * font_scale, color=muted)

    # Anatomical guides. The mid-torso line is now a real boundary all the way
    # across: the side periphery and the extreme periphery are both split at it,
    # so the rule no longer cuts through the middle of any cell.
    guides = [
        ("head", "head top", 0.0, 1.0),
        ("chin", "chin", 0.0, 1.0),
        ("mid", "mid torso", 0.0, 1.0),
        ("hip", "torso bottom", 0.0, 1.0),
        ("low", "below torso", 0.0, 1.0),
    ]
    for key, label, x_start, x_end in guides:
        ax.plot([x_start, x_end], [Y[key], Y[key]],
                color="#999999", linewidth=0.5, zorder=3)
        # A white outline round the guide labels: now that a busy region can be
        # near-saturated, grey-on-dark would be unreadable exactly where the
        # interesting cell is.
        ax.text(x_start + 0.006, Y[key] - 0.006, label, fontsize=8 * font_scale,
                color="#444444", va="bottom", style="italic", zorder=4,
                path_effects=[path_effects.withStroke(linewidth=2.2,
                                                      foreground="white")])

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)          # y grows downward, as in the coordinate system
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16 * font_scale, color=colour, pad=10,
                 fontweight="bold")
    ax.set_xlabel("viewer's left = signer's right", fontsize=8.5 * font_scale,
                  color="#666666", labelpad=4)


def body_map_figure(
    distribution: pd.DataFrame,
    keyword: str,
    region_code: str,
    n_clips: int,
    n_signers: int,
    out_path: Path,
    font_scale: float = 1.0,
) -> Path:
    """One figure per keyword: dominant and non-dominant body maps side by side.

    ``font_scale`` multiplies every type size; raise it for a poster, where the
    figure is read from two metres rather than on screen.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15 * font_scale, 9.6 * font_scale))

    for ax, role in zip(axes, ("dominant", "non_dominant")):
        draw_body_map(ax, distribution, role, ROLE_LABELS[role], font_scale)

    caption = (
        f"{keyword} / {region_code}  -  share of hand points per signing-space region\n"
        f"{n_clips} clips from {n_signers} signers; "
        f"[low, high] = 95% CI bootstrapped over signers"
    )
    if n_signers < MIN_SIGNERS_FOR_STABLE_CI:
        caption += f"  -  WARNING: fewer than {MIN_SIGNERS_FOR_STABLE_CI} signers, interval unstable"

    fig.suptitle(caption, fontsize=13 * font_scale)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def grouped_bar_figure(
    table: pd.DataFrame,
    group_column: str,
    out_path: Path,
    title: str,
    value_column: str = "avg_regions",
    x_label: str = "average number of signing-space regions per clip",
    font_scale: float = 1.0,
) -> Optional[Path]:
    """Average regions used, by group and keyword, with bootstrap whiskers."""
    if table.empty:
        return None

    groups = sorted(table[group_column].astype(str).unique())
    keywords = sorted(table["keyword"].astype(str).unique())

    fig_height = max(3.6, 0.52 * len(groups) * len(keywords) + 2.0) * font_scale
    fig, ax = plt.subplots(figsize=(11.5 * font_scale, fig_height))

    palette = plt.get_cmap("tab10")
    height = 0.8 / max(len(keywords), 1)
    positions: List[float] = []
    labels: List[str] = []

    for group_index, group in enumerate(groups):
        for keyword_index, keyword in enumerate(keywords):
            row = table[
                (table[group_column].astype(str) == group)
                & (table["keyword"].astype(str) == keyword)
            ]
            if row.empty:
                continue
            row = row.iloc[0]

            value = float(row[value_column])
            low = float(row.get("ci_low", np.nan))
            high = float(row.get("ci_high", np.nan))
            error = (
                [[max(0.0, value - low)], [max(0.0, high - value)]]
                if np.isfinite(low) and np.isfinite(high) else None
            )

            y = group_index + keyword_index * height
            positions.append(y)
            # n goes in the tick label, not next to the bar, so it can never
            # collide with the error whisker.
            labels.append(f"{group}  {keyword}  (n={int(row['n_signers'])})")

            unreliable = not bool(row.get("ci_reliable", True))
            ax.barh(
                y, value, height=height * 0.9,
                color=palette(keyword_index % 10),
                alpha=0.55 if unreliable else 0.95,
                hatch="//" if unreliable else None,
                edgecolor="white", linewidth=0.5,
                label=keyword if group_index == 0 else None,
            )
            if error is not None:
                ax.errorbar(value, y, xerr=error, fmt="none",
                            ecolor="#333333", elinewidth=1.0, capsize=3)


    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=10 * font_scale)
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=12 * font_scale)
    ax.tick_params(axis="x", labelsize=10 * font_scale)
    ax.set_title(title, fontsize=15 * font_scale, fontweight="bold")
    ax.legend(title="keyword", fontsize=10 * font_scale,
              title_fontsize=10 * font_scale, loc="lower right")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.text(
        0.01, 0.005,
        "Whiskers: 95% CI bootstrapped over signers. n = unique signers. "
        f"Hatched bars have fewer than {MIN_SIGNERS_FOR_STABLE_CI} signers.",
        fontsize=9 * font_scale, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def region_group_figure(
    group_distribution: pd.DataFrame,
    out_path: Path,
    title: str = "Signing-space use by anatomical group",
    font_scale: float = 1.0,
) -> Optional[Path]:
    """Stacked shares across the coarse region groups, per keyword and hand role."""
    if group_distribution.empty:
        return None

    frame = group_distribution.copy()
    frame["label"] = frame["keyword"].astype(str) + "  " + frame["hand_role"].astype(str)
    pivot = frame.pivot_table(index="label", columns="group", values="percent",
                              aggfunc="mean").fillna(0.0)

    fig, ax = plt.subplots(figsize=(11.5 * font_scale,
                                    max(3.2, 0.55 * len(pivot) + 1.8) * font_scale))
    left = np.zeros(len(pivot))
    palette = plt.get_cmap("Set2")

    for index, column in enumerate(pivot.columns):
        values = pivot[column].to_numpy()
        ax.barh(pivot.index, values, left=left, label=column,
                color=palette(index % 8), edgecolor="white", linewidth=0.6)
        left += values

    ax.set_xlabel("share of hand points (%)", fontsize=12 * font_scale)
    ax.tick_params(axis="x", labelsize=10 * font_scale)
    ax.set_title(title, fontsize=15 * font_scale, fontweight="bold")
    ax.legend(fontsize=10 * font_scale, ncol=2, loc="lower right")
    ax.tick_params(axis="y", labelsize=10 * font_scale)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
