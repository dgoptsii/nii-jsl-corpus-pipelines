"""Aggregation and uncertainty for signing-space statistics.

Every interval here is a bootstrap CI resampled over **signers**, not over
clips. One signer contributes many clips, so clips are not independent
observations and a clip-level interval would come out far too narrow. Every
table carries ``n_signers`` next to its interval, and cells below
:data:`config.MIN_SIGNERS_FOR_STABLE_CI` are flagged, since a bootstrap over
very few signers is itself unstable.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    CENTRAL_REGIONS,
    CI_QUALITY_TIERS,
    MIN_SIGNERS_FOR_ANY_CI,
    EXTREME_REGIONS,
    HAND_ROLES,
    MIN_SIGNERS_FOR_STABLE_CI,
    PERIPHERY_REGIONS,
    REGION_GROUPS,
    REGION_KEYS,
)


# ===========================================================================
# BOOTSTRAP
# ===========================================================================

def bootstrap_ci_over_signers(
    values: Sequence[float],
    signers: Sequence[str],
    statistic: Callable[[np.ndarray], float] = np.mean,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """Confidence interval for ``statistic``, resampling signers with replacement.

    ``values`` and ``signers`` are parallel: one entry per clip, naming the
    signer who produced it. Each bootstrap round draws as many signers as there
    are distinct signers, pools all clips belonging to the drawn signers, and
    recomputes the statistic.

    Returns ``(nan, nan)`` when there is nothing to resample - a single signer
    carries no information about between-signer variation.
    """
    values = np.asarray(list(values), dtype=np.float64)
    signers = np.asarray(list(signers), dtype=object)

    if len(values) == 0 or len(values) != len(signers):
        return (float("nan"), float("nan"))

    unique = np.unique(signers)
    if len(unique) < 2:
        return (float("nan"), float("nan"))

    by_signer = {signer: values[signers == signer] for signer in unique}
    rng = np.random.default_rng(seed)

    estimates = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        pooled = np.concatenate([by_signer[s] for s in drawn])
        estimates[i] = statistic(pooled) if len(pooled) else np.nan

    estimates = estimates[np.isfinite(estimates)]
    if len(estimates) == 0:
        return (float("nan"), float("nan"))

    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.percentile(estimates, 100 * alpha)),
        float(np.percentile(estimates, 100 * (1 - alpha))),
    )


def ci_quality(n_signers: int) -> str:
    """How much to trust an interval built from this many signers."""
    for floor, label in CI_QUALITY_TIERS:
        if n_signers >= floor:
            return label
    return "none"


def bootstrap_difference(
    values_a: Sequence[float], signers_a: Sequence[str],
    values_b: Sequence[float], signers_b: Sequence[str],
    statistic: Callable[[np.ndarray], float] = np.mean,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Resample two groups independently and describe their difference.

    Asking whether two confidence intervals overlap is the wrong test in both
    directions: too strict, since two 95% intervals can overlap while the
    difference is reliably non-zero, and lossy, since overlap is a yes/no answer
    to a question that has a magnitude.

    Each round instead draws signers with replacement from each group separately
    and keeps the difference. Returns the observed difference, its interval,
    ``p_a_greater`` (the share of resamples with A above B), and ``separates``,
    whether that interval excludes zero.
    """
    empty = {"difference": float("nan"), "diff_ci_low": float("nan"),
             "diff_ci_high": float("nan"), "p_a_greater": float("nan"),
             "separates": False, "n_signers_a": 0, "n_signers_b": 0}

    values_a = np.asarray(list(values_a), dtype=np.float64)
    values_b = np.asarray(list(values_b), dtype=np.float64)
    signers_a = np.asarray(list(signers_a), dtype=object)
    signers_b = np.asarray(list(signers_b), dtype=object)

    if len(values_a) != len(signers_a) or len(values_b) != len(signers_b):
        return empty

    unique_a, unique_b = np.unique(signers_a), np.unique(signers_b)
    empty["n_signers_a"], empty["n_signers_b"] = len(unique_a), len(unique_b)
    if len(unique_a) < MIN_SIGNERS_FOR_ANY_CI or len(unique_b) < MIN_SIGNERS_FOR_ANY_CI:
        return empty

    by_a = {s: values_a[signers_a == s] for s in unique_a}
    by_b = {s: values_b[signers_b == s] for s in unique_b}
    rng = np.random.default_rng(seed)

    differences = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        pooled_a = np.concatenate([by_a[s] for s in
                                   rng.choice(unique_a, len(unique_a), replace=True)])
        pooled_b = np.concatenate([by_b[s] for s in
                                   rng.choice(unique_b, len(unique_b), replace=True)])
        differences[index] = statistic(pooled_a) - statistic(pooled_b)

    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return empty

    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(differences, 100 * alpha))
    high = float(np.percentile(differences, 100 * (1 - alpha)))

    return {
        "difference": round(float(statistic(values_a) - statistic(values_b)), 4),
        "diff_ci_low": round(low, 4),
        "diff_ci_high": round(high, 4),
        "p_a_greater": round(float((differences > 0).mean()), 4),
        "separates": bool(low > 0 or high < 0),
        "n_signers_a": int(len(unique_a)),
        "n_signers_b": int(len(unique_b)),
    }


def pairwise_comparisons(clip_table: pd.DataFrame,
                         group_column: str,
                         within_column: str = "keyword",
                         hand_role: str = "dominant",
                         value_column: str = "regions_used") -> pd.DataFrame:
    """Every pair of levels of ``group_column``, compared within each category.

    This is the table behind a claim like "category explains more than
    prefecture": it says by how much, with what uncertainty, and how often the
    ordering held --- rather than only how many intervals happened to overlap.
    """
    frame = clip_table[clip_table["hand_role"] == hand_role]
    if "hand_present" in frame.columns:
        frame = frame[frame["hand_present"].astype(str).str.lower().isin({"true", "1"})]

    rows: List[Dict[str, object]] = []
    for within in sorted(frame[within_column].dropna().unique()):
        block = frame[frame[within_column] == within]
        levels = sorted(block[group_column].dropna().unique())
        for level_a, level_b in itertools.combinations(levels, 2):
            side_a = block[block[group_column] == level_a]
            side_b = block[block[group_column] == level_b]
            if side_a.empty or side_b.empty:
                continue
            result = bootstrap_difference(
                side_a[value_column], side_a["signer_id"],
                side_b[value_column], side_b["signer_id"])
            rows.append({
                "compared_within": within,
                "group": group_column,
                "level_a": level_a,
                "level_b": level_b,
                "hand_role": hand_role,
                "mean_a": round(float(pd.to_numeric(side_a[value_column]).mean()), 4),
                "mean_b": round(float(pd.to_numeric(side_b[value_column]).mean()), 4),
                "n_clips_a": int(len(side_a)),
                "n_clips_b": int(len(side_b)),
                **result,
                "quality_a": ci_quality(result["n_signers_a"]),
                "quality_b": ci_quality(result["n_signers_b"]),
            })

    columns = ["compared_within", "group", "level_a", "level_b", "hand_role",
               "mean_a", "mean_b", "difference", "diff_ci_low", "diff_ci_high",
               "p_a_greater", "separates", "n_clips_a", "n_clips_b",
               "n_signers_a", "n_signers_b", "quality_a", "quality_b"]
    return pd.DataFrame(rows, columns=columns)


def effect_spread(clip_table: pd.DataFrame,
                  group_column: str,
                  within_column: str = "keyword",
                  hand_role: str = "dominant",
                  value_column: str = "regions_used",
                  min_signers: int = MIN_SIGNERS_FOR_STABLE_CI) -> pd.DataFrame:
    """How far apart a grouping's levels are, in the units of the measure.

    Counting non-overlapping intervals answers "can we tell these apart?", which
    collapses to zero for any grouping with few levels and hides a real difference
    in size. ``spread``, the gap between the highest and lowest level mean, says
    how much the grouping moves the answer instead.

    Deliberately not an overlap percentage: interval width is driven by how many
    signers a level happens to have, so an overlap measure makes a thinly-sampled
    level look more similar to everything else. Levels below ``min_signers`` are
    excluded, since one resting on three signers can sit anywhere.
    """
    frame = clip_table[clip_table["hand_role"] == hand_role]
    if "hand_present" in frame.columns:
        frame = frame[frame["hand_present"].astype(str).str.lower().isin({"true", "1"})]

    rows: List[Dict[str, object]] = []
    for within in sorted(frame[within_column].dropna().unique()):
        block = frame[frame[within_column] == within]
        kept, means = [], []
        for level in sorted(block[group_column].dropna().unique()):
            side = block[block[group_column] == level]
            if side["signer_id"].nunique() < min_signers:
                continue
            kept.append(level)
            means.append(float(pd.to_numeric(side[value_column]).mean()))
        if len(means) < 2:
            continue
        highest, lowest = int(np.argmax(means)), int(np.argmin(means))
        rows.append({
            "group": group_column,
            "compared_within": within,
            "hand_role": hand_role,
            "n_levels": len(kept),
            "levels_excluded_few_signers":
                int(block[group_column].nunique() - len(kept)),
            "spread": round(max(means) - min(means), 4),
            "sd_of_level_means": round(float(np.std(means, ddof=1)), 4),
            "highest_level": kept[highest],
            "highest_mean": round(means[highest], 4),
            "lowest_level": kept[lowest],
            "lowest_mean": round(means[lowest], 4),
        })
    return pd.DataFrame(rows)


def comparison_scoreboard(comparisons: pd.DataFrame) -> pd.DataFrame:
    """How many pairs separate, per grouping, the summary the poster prints.

    Counted from the interval of the *difference*, not from interval overlap, so
    the number is not the artificially conservative one.
    """
    if comparisons.empty:
        return pd.DataFrame(columns=["group", "n_pairs", "n_separating",
                                     "percent_separating", "n_pairs_reliable",
                                     "n_separating_reliable"])
    rows = []
    for group, block in comparisons.groupby("group", sort=True):
        reliable = block[(block["n_signers_a"] >= MIN_SIGNERS_FOR_STABLE_CI)
                         & (block["n_signers_b"] >= MIN_SIGNERS_FOR_STABLE_CI)]
        rows.append({
            "group": group,
            "n_pairs": int(len(block)),
            "n_separating": int(block["separates"].sum()),
            "percent_separating": round(100 * block["separates"].mean(), 1),
            "n_pairs_reliable": int(len(reliable)),
            "n_separating_reliable": int(reliable["separates"].sum()),
        })
    return pd.DataFrame(rows)


def _summary_stats(frame: pd.DataFrame, value_column: str) -> Dict[str, float]:
    """Mean, spread and a signer-level interval for one group of clips."""
    values = pd.to_numeric(frame[value_column], errors="coerce").dropna()
    signers = frame.loc[values.index, "signer_id"].astype(str)

    n_signers = int(signers.nunique())
    low, high = bootstrap_ci_over_signers(values.to_numpy(), signers.to_numpy())

    return {
        "n_annotations": int(len(frame)),
        "n_clips_with_hand": int(len(values)),
        "n_signers": n_signers,
        "mean": round(float(values.mean()), 4) if len(values) else float("nan"),
        "sd": round(float(values.std(ddof=1)), 4) if len(values) > 1 else float("nan"),
        "median": round(float(values.median()), 4) if len(values) else float("nan"),
        "ci_low": round(low, 4) if np.isfinite(low) else float("nan"),
        "ci_high": round(high, 4) if np.isfinite(high) else float("nan"),
        "ci_reliable": bool(n_signers >= MIN_SIGNERS_FOR_STABLE_CI),
        "ci_quality": ci_quality(n_signers),
    }


# ===========================================================================
# CLIP-LEVEL TABLE
# ===========================================================================

def build_clip_table(frames: pd.DataFrame, hand_roles: Sequence[str]) -> pd.DataFrame:
    """Collapse per-frame region counts into one row per clip and hand role.

    ``regions_used`` - the number of distinct signing-space regions the hand
    visited during the clip - is the "average number of signing regions" measure
    requested for the summary tables.
    """
    rows: List[Dict[str, object]] = []
    group_columns = ["keyword", "region_code", "clip_id"]

    for keys, clip in frames.groupby(group_columns, dropna=False):
        keyword, region_code, clip_id = keys
        first = clip.iloc[0]

        for role in hand_roles:
            totals = {
                region: float(clip[f"{role}_{region}"].sum())
                for region in REGION_KEYS
                if f"{role}_{region}" in clip.columns
            }
            counted = {r: v for r, v in totals.items() if r != "missing"}
            used = [region for region, value in counted.items() if value > 0]
            total_points = sum(counted.values())

            row: Dict[str, object] = {
                "keyword": keyword,
                "region_code": region_code,
                "clip_id": clip_id,
                "signer_id": first.get("signer_id", ""),
                "age_group": first.get("age_group", "unknown"),
                "age_band": first.get("age_band", "unknown"),
                "gender": first.get("gender", "unknown"),
                "handedness": first.get("handedness", "right"),
                "hand_role": role,
                "hand_present": total_points > 0,
                "regions_used": len(used),
                "used_regions": ";".join(sorted(used)),
                "total_points": int(total_points),
                "n_frames": int(len(clip)),
            }
            for region in REGION_KEYS:
                row[region] = int(totals.get(region, 0))
            rows.append(row)

    return pd.DataFrame(rows)


# ===========================================================================
# REQUESTED SUMMARY TABLES
# ===========================================================================

def summarise_by(
    clip_table: pd.DataFrame,
    group_columns: Sequence[str],
    hand_roles: Sequence[str] = tuple(HAND_ROLES),
    present_only: bool = True,
) -> pd.DataFrame:
    """Number of annotations, unique signers and average regions per group.

    The shape of both requested tables; only ``group_columns`` differs
    (``["region_code", "keyword"]`` versus ``["age_group", "keyword"]``).

    **The two hands are summarised separately.** ``hand_role`` is always part of
    the grouping, so every cell appears twice. Pooling would be meaningless: the
    dominant hand carries the sign while the non-dominant one is often idle, and
    it would mix a hand present in every clip with one that is not.

    ``n_annotations`` therefore counts only the clips in which *that* hand was
    detected; ``n_clips`` is the cell's full denominator and
    ``hand_present_percent`` the ratio, a direct measure of how two-handed the
    keyword is.
    """
    group_columns = list(group_columns)
    if isinstance(hand_roles, str):
        hand_roles = [hand_roles]
    hand_roles = list(hand_roles)

    frame = clip_table
    if hand_roles and "hand_role" in frame.columns:
        frame = frame[frame["hand_role"].isin(hand_roles)]

    rows: List[Dict[str, object]] = []
    for keys, cell in frame.groupby(group_columns + ["hand_role"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        n_clips = int(len(cell))
        group = cell[cell["hand_present"]] if present_only else cell
        if group.empty:
            continue

        row: Dict[str, object] = dict(zip(group_columns + ["hand_role"], keys))
        row.update(_summary_stats(group, "regions_used"))
        row["avg_regions"] = row.pop("mean")
        row["sd_regions"] = row.pop("sd")
        row["median_regions"] = row.pop("median")
        row["n_clips"] = n_clips
        row["hand_present_percent"] = round(100.0 * row["n_annotations"] / n_clips, 2)
        row["n_left_handed_signers"] = int(
            group[group["handedness"] == "left"]["signer_id"].nunique()
        )
        rows.append(row)

    columns = group_columns + [
        "hand_role", "n_annotations", "n_clips", "hand_present_percent",
        "n_signers", "n_left_handed_signers",
        "avg_regions", "ci_low", "ci_high", "ci_reliable", "ci_quality",
        "sd_regions", "median_regions",
    ]

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # Keep the roles in the order they were asked for, not alphabetically.
    order = {role: index for index, role in enumerate(hand_roles)}
    table["_role_order"] = table["hand_role"].map(order).fillna(len(order))
    table = table.sort_values(group_columns + ["_role_order"])
    return table[columns].reset_index(drop=True)


def region_by_keyword_table(
    clip_table: pd.DataFrame,
    hand_roles: Sequence[str] = tuple(HAND_ROLES),
) -> pd.DataFrame:
    """Table 1: geographical region x keyword x hand role."""
    return summarise_by(clip_table, ["region_code", "keyword"], hand_roles=hand_roles)


def age_group_by_keyword_table(
    clip_table: pd.DataFrame,
    hand_roles: Sequence[str] = tuple(HAND_ROLES),
) -> pd.DataFrame:
    """Table 2: age group x keyword x hand role."""
    return summarise_by(clip_table, ["age_group", "keyword"], hand_roles=hand_roles)


def age_band_by_keyword_table(
    clip_table: pd.DataFrame,
    hand_roles: Sequence[str] = tuple(HAND_ROLES),
) -> pd.DataFrame:
    """The coarse ``<50`` / ``50+`` split x keyword x hand role.

    Two bands rather than seven decades, so each cell holds enough signers for
    the bootstrap to say something. This is the age comparison to report.
    """
    return summarise_by(clip_table, ["age_band", "keyword"], hand_roles=hand_roles)


def gender_by_keyword_table(
    clip_table: pd.DataFrame,
    hand_roles: Sequence[str] = tuple(HAND_ROLES),
) -> pd.DataFrame:
    """Table 3: gender x keyword x hand role."""
    return summarise_by(clip_table, ["gender", "keyword"], hand_roles=hand_roles)


def cross_table(
    clip_table: pd.DataFrame,
    group_columns: Sequence[str] = ("region_code", "age_band", "gender", "keyword"),
    hand_roles: Sequence[str] = tuple(HAND_ROLES),
) -> pd.DataFrame:
    """Every combination of the grouping variables at once.

    This is the table to look at for interactions - whether an age effect holds
    in every prefecture, whether it differs by gender - which the one-variable
    tables cannot show.

    It uses the coarse ``age_band`` rather than the decades on purpose: seven
    decades x seven prefectures x two genders would leave most cells with one or
    two signers. Even at two bands it fragments, so check ``ci_reliable`` before
    reading any cell, and treat a False as "not enough people" rather than a
    result.
    """
    return summarise_by(clip_table, list(group_columns), hand_roles=hand_roles)


# ===========================================================================
# REGION DISTRIBUTIONS (for the body-map figures)
# ===========================================================================

def region_distribution(
    clip_table: pd.DataFrame,
    group_columns: Sequence[str] = ("keyword", "region_code"),
    hand_roles: Sequence[str] = ("dominant", "non_dominant"),
) -> pd.DataFrame:
    """Share of hand points in each signing-space region, with signer-level CIs.

    The per-clip share is computed first and then averaged over clips, so a long
    clip does not dominate a short one, and the interval is resampled over
    signers exactly as elsewhere.
    """
    rows: List[Dict[str, object]] = []
    # Tolerate a table that does not carry every region column.
    counted = [region for region in REGION_KEYS
               if region != "missing" and region in clip_table.columns]
    if not counted:
        return pd.DataFrame()

    for role in hand_roles:
        frame = clip_table[(clip_table["hand_role"] == role) & (clip_table["hand_present"])]
        if frame.empty:
            continue

        for keys, group in frame.groupby(list(group_columns), dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(group_columns, keys))
            totals = group[counted].to_numpy(dtype=float)
            per_clip_total = totals.sum(axis=1)
            per_clip_total[per_clip_total == 0] = np.nan
            shares = 100.0 * totals / per_clip_total[:, None]
            signers = group["signer_id"].astype(str).to_numpy()

            for column_index, region in enumerate(counted):
                values = shares[:, column_index]
                ok = np.isfinite(values)
                low, high = bootstrap_ci_over_signers(values[ok], signers[ok])
                rows.append({
                    **base,
                    "hand_role": role,
                    "region": region,
                    "percent": round(float(np.nanmean(values)), 4) if ok.any() else 0.0,
                    "ci_low": round(low, 4) if np.isfinite(low) else float("nan"),
                    "ci_high": round(high, 4) if np.isfinite(high) else float("nan"),
                    "points": int(group[region].sum()),
                    "n_clips": int(len(group)),
                    "n_signers": int(group["signer_id"].nunique()),
                })

    return pd.DataFrame(rows)


def region_group_distribution(distribution: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-region shares into the coarse anatomical groups."""
    if distribution.empty:
        return distribution

    keys = [c for c in distribution.columns
            if c in {"keyword", "region_code", "age_group", "hand_role"}]
    rows: List[Dict[str, object]] = []

    for values, group in distribution.groupby(keys, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        base = dict(zip(keys, values))
        for name, regions in REGION_GROUPS.items():
            subset = group[group["region"].isin(regions)]
            rows.append({
                **base,
                "group": name,
                "percent": round(float(subset["percent"].sum()), 4),
                "points": int(subset["points"].sum()),
            })

    return pd.DataFrame(rows)


def central_periphery_summary(distribution: pd.DataFrame) -> pd.DataFrame:
    """Central / periphery / extreme split, plus the dominant single region."""
    if distribution.empty:
        return distribution

    keys = [c for c in distribution.columns
            if c in {"keyword", "region_code", "age_group", "hand_role"}]
    rows: List[Dict[str, object]] = []

    for values, group in distribution.groupby(keys, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        top = group.sort_values("percent", ascending=False).iloc[0]
        rows.append({
            **dict(zip(keys, values)),
            "dominant_region": str(top["region"]),
            "dominant_percent": float(top["percent"]),
            "central_percent": round(float(group[group["region"].isin(CENTRAL_REGIONS)]["percent"].sum()), 4),
            "periphery_percent": round(float(group[group["region"].isin(PERIPHERY_REGIONS)]["percent"].sum()), 4),
            "extreme_percent": round(float(group[group["region"].isin(EXTREME_REGIONS)]["percent"].sum()), 4),
            "n_clips": int(top["n_clips"]),
            "n_signers": int(top["n_signers"]),
        })

    return pd.DataFrame(rows)
