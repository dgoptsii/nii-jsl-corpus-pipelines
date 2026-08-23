"""Top-N gloss selection and the vocabulary coverage table.

The functions here are the ones meant to be called directly, from a script or
an interactive session:

    from topn import top_glosses, write_top_glosses, coverage_table

    top_glosses(stats, top_n=100, min_signers=5)
    top_glosses(stats, top_n=100, min_signers=5, regions=["FO", "GM"])
    write_top_glosses(annotations, "out/top100.csv", top_n=100, min_signers=5)
    coverage_table(stats)

Two ideas run through all of them.

**A gloss needs several signers to be usable.** A gloss seen 400 times from one
person teaches a model that person's idiolect. ``min_signers`` is therefore a
first-class argument everywhere, not a filter applied afterwards.

**Totals are capped.** A handful of very frequent glosses would otherwise
dominate any total, and a training set built from them would be just as
imbalanced. The capped total answers the more useful question: how many
examples would you actually keep if you took at most ``cap`` per gloss?
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from config import (
    GLOBAL_TAG,
    MIN_SIGNERS_DEFAULT,
    OCCURRENCE_CAP,
    TOP_N_SPECS,
)
from io_utils import write_csv
from metrics import add_flags, gloss_statistics


# ===========================================================================
# SELECTION
# ===========================================================================

def gloss_statistics_for_regions(annotations: pd.DataFrame,
                                 regions: Optional[Sequence[str]] = None
                                 ) -> pd.DataFrame:
    """Gloss statistics restricted to a set of prefectures.

    ``regions=None`` means the whole corpus. Passing a list recomputes the
    counts inside those prefectures only -- it does not filter a global table,
    which would leave the signer and occurrence counts describing the whole
    corpus while claiming to describe a subset.
    """
    frame = annotations
    if "is_parsed" not in frame.columns:
        frame = add_flags(frame)
    if regions:
        wanted = {str(r).upper() for r in regions}
        frame = frame[frame["region_code"].astype(str).str.upper().isin(wanted)]
    return gloss_statistics(frame)


def top_glosses(gloss_stats: pd.DataFrame,
                top_n: int = 100,
                min_signers: int = MIN_SIGNERS_DEFAULT,
                min_occurrences: int = 1) -> pd.DataFrame:
    """The ``top_n`` most frequent glosses that clear ``min_signers``.

    The signer filter is applied *before* the cutoff, so "top 100 with at least
    5 signers" means the 100 most frequent glosses among those that qualify --
    not the qualifying members of the overall top 100, which would silently
    return fewer than 100 rows.
    """
    eligible = gloss_stats[
        (gloss_stats["n_signers"] >= int(min_signers))
        & (gloss_stats["occurrences"] >= int(min_occurrences))
    ].reset_index(drop=True)

    selected = eligible if not top_n else eligible.head(int(top_n))
    selected = selected.copy()
    selected.insert(0, "rank", range(1, len(selected) + 1))
    return selected


def write_top_glosses(annotations: pd.DataFrame,
                      path: Path,
                      top_n: int = 100,
                      min_signers: int = MIN_SIGNERS_DEFAULT,
                      regions: Optional[Sequence[str]] = None,
                      min_occurrences: int = 1) -> pd.DataFrame:
    """Build a top-N gloss list for chosen prefectures and write it to CSV.

    This is the entry point for "give me the top 200 glosses in Fukuoka and
    Nagasaki with at least 5 signers each".
    """
    stats = gloss_statistics_for_regions(annotations, regions)
    selected = top_glosses(stats, top_n=top_n, min_signers=min_signers,
                           min_occurrences=min_occurrences)
    selected = selected.copy()
    selected.insert(1, "scope", ";".join(regions) if regions else GLOBAL_TAG)
    selected.insert(2, "min_signers_required", int(min_signers))
    write_csv(Path(path), selected)
    return selected


# ===========================================================================
# THE COVERAGE TABLE
# ===========================================================================

def coverage_row(gloss_stats: pd.DataFrame,
                 label: str,
                 top_n: int,
                 min_signers: int,
                 cap: int = OCCURRENCE_CAP,
                 scope: str = GLOBAL_TAG) -> dict:
    """One row of the coverage table: the shape of one candidate vocabulary."""
    selected = top_glosses(gloss_stats, top_n=top_n, min_signers=min_signers)

    if selected.empty:
        return {"scope": scope, "group": label, "min_signers": min_signers,
                "n_glosses": 0, "occurrences_min": 0, "occurrences_max": 0,
                "occurrences_max_capped": 0, "total_occurrences": 0,
                "total_occurrences_capped": 0, "cap": cap,
                "signers_min": 0, "signers_max": 0, "percent_of_corpus_tokens": 0.0}

    occurrences = selected["occurrences"]
    capped = occurrences.clip(upper=int(cap))
    corpus_tokens = int(gloss_stats["occurrences"].sum()) or 1

    return {
        "scope": scope,
        "group": label,
        "min_signers": int(min_signers),
        "n_glosses": int(len(selected)),
        "occurrences_min": int(occurrences.min()),
        "occurrences_max": int(occurrences.max()),
        "occurrences_max_capped": int(capped.max()),
        "total_occurrences": int(occurrences.sum()),
        "total_occurrences_capped": int(capped.sum()),
        "cap": int(cap),
        "signers_min": int(selected["n_signers"].min()),
        "signers_max": int(selected["n_signers"].max()),
        "percent_of_corpus_tokens": round(100 * occurrences.sum() / corpus_tokens, 2),
    }


def coverage_table(gloss_stats: pd.DataFrame,
                   specs: Sequence[Tuple[str, int, int]] = tuple(TOP_N_SPECS),
                   cap: int = OCCURRENCE_CAP,
                   scope: str = GLOBAL_TAG) -> pd.DataFrame:
    """The headline vocabulary table.

    One row per candidate vocabulary size, each with its own signer floor.
    Reading down the ``occurrences_min`` column shows how quickly the tail
    thins: it is the number that decides whether a vocabulary of that size is
    trainable at all.
    """
    return pd.DataFrame([
        coverage_row(gloss_stats, label, top_n, min_signers, cap=cap, scope=scope)
        for label, top_n, min_signers in specs
    ])


def coverage_table_by_region(annotations: pd.DataFrame,
                             regions: Optional[Iterable[str]] = None,
                             specs: Sequence[Tuple[str, int, int]] = tuple(TOP_N_SPECS),
                             cap: int = OCCURRENCE_CAP) -> pd.DataFrame:
    """The coverage table for the whole corpus and for each prefecture."""
    frame = annotations if "is_parsed" in annotations.columns else add_flags(annotations)
    codes = sorted(frame["region_code"].dropna().unique()) if regions is None else list(regions)

    tables = [coverage_table(gloss_statistics(frame), specs, cap, scope=GLOBAL_TAG)]
    for code in codes:
        stats = gloss_statistics_for_regions(frame, [code])
        if not stats.empty:
            tables.append(coverage_table(stats, specs, cap, scope=str(code)))
    return pd.concat(tables, ignore_index=True)
