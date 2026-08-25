"""What a machine-learning user needs to know before touching the corpus.

:mod:`metrics` describes the corpus as a linguistic object; this describes it
as a dataset:

* :func:`class_size_table`: how the vocabulary shrinks as the minimum example
  count and signer floor rise, and how many tokens survive with it.
* :func:`duration_distribution`: segment durations, which set the input window
  and the frame budget.
* :func:`signing_rate`: pace varies between signers, so a model trained on one
  may not transfer.
* :func:`split_feasibility`: the important one. A random split leaks the same
  signer into training and test, and the accuracy then measures memorisation of
  a person rather than recognition of a sign. This reports what a proper
  held-out-signer split costs in vocabulary and in tokens.
* :func:`coverage_curve`: cumulative token share by rank.
* :func:`marker_cooccurrence`: which markers travel together, before deciding
  which deserve a head in a multi-task model.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from config import (
    CLASS_SIZE_SIGNER_FLOORS,
    CLASS_SIZE_THRESHOLDS,
    COVERAGE_CUTOFFS,
    DEFAULT_TEST_FRACTION,
    GLOBAL_TAG,
    KEY_COLUMNS,
    LEXICAL_COLUMN,
    OCCURRENCE_CAP,
    SIGNER_COLUMN,
)
from io_utils import nonempty
from metrics import gloss_statistics

DURATION_PERCENTILES = [5, 25, 50, 75, 95, 99]


# ===========================================================================
# HOW MANY CLASSES ARE TRAINABLE
# ===========================================================================

def class_size_table(gloss_stats: pd.DataFrame,
                     thresholds: Sequence[int] = tuple(CLASS_SIZE_THRESHOLDS),
                     min_signers: Sequence[int] = tuple(CLASS_SIZE_SIGNER_FLOORS),
                     cap: int = OCCURRENCE_CAP,
                     label: str = GLOBAL_TAG) -> pd.DataFrame:
    """Vocabulary and token counts surviving each (examples, signers) floor.

    The two floors matter for different reasons. Too few *examples* and the
    class cannot be learned; too few *signers* and it can be learned only as one
    person's production. Reporting them as a grid rather than two lists is what
    makes the trade-off visible: raising the signer floor from 1 to 3 usually
    costs far more vocabulary than raising the example floor does.

    Tokens are reported both raw and capped at ``cap`` per gloss, for the same
    reason the coverage table caps them: the raw total is dominated by a handful
    of very frequent glosses, and the capped total is the number of examples a
    balanced training set would actually keep.
    """
    if gloss_stats.empty:
        return pd.DataFrame(columns=["tag", "min_examples", "min_signers",
                                     "n_glosses", "n_tokens", "percent_of_tokens",
                                     "n_tokens_capped", "percent_of_tokens_capped",
                                     "cap"])

    total_tokens = int(gloss_stats["occurrences"].sum()) or 1
    total_capped = int(gloss_stats["occurrences"].clip(upper=int(cap)).sum()) or 1

    rows = []
    for signers in min_signers:
        for threshold in thresholds:
            kept = gloss_stats[(gloss_stats["occurrences"] >= int(threshold))
                               & (gloss_stats["n_signers"] >= int(signers))]
            tokens = int(kept["occurrences"].sum())
            capped = int(kept["occurrences"].clip(upper=int(cap)).sum())
            rows.append({
                "tag": label,
                "min_examples": int(threshold),
                "min_signers": int(signers),
                "n_glosses": int(len(kept)),
                "n_tokens": tokens,
                "percent_of_tokens": round(100 * tokens / total_tokens, 2),
                "n_tokens_capped": capped,
                "percent_of_tokens_capped": round(100 * capped / total_capped, 2),
                "cap": int(cap),
            })
    return pd.DataFrame(rows)


def coverage_curve(gloss_stats: pd.DataFrame,
                   cutoffs: Sequence[int] = tuple(COVERAGE_CUTOFFS),
                   label: str = GLOBAL_TAG) -> pd.DataFrame:
    """Cumulative token share of the ``n`` most frequent glosses.

    Also reported as the complementary out-of-vocabulary rate, which is the
    number a recogniser restricted to that vocabulary would actually face.
    """
    if gloss_stats.empty:
        return pd.DataFrame(columns=["tag", "vocabulary_size", "n_tokens_covered",
                                     "percent_covered", "oov_percent",
                                     "min_occurrences_at_cutoff"])

    ordered = gloss_stats.sort_values("occurrences", ascending=False)
    occurrences = ordered["occurrences"].to_numpy()
    cumulative = np.cumsum(occurrences)
    total = int(cumulative[-1]) or 1

    rows = []
    for cutoff in list(cutoffs) + [len(ordered)]:
        size = min(int(cutoff), len(ordered))
        if size <= 0:
            continue
        covered = int(cumulative[size - 1])
        rows.append({
            "tag": label,
            "vocabulary_size": size,
            "n_tokens_covered": covered,
            "percent_covered": round(100 * covered / total, 2),
            "oov_percent": round(100 * (total - covered) / total, 2),
            "min_occurrences_at_cutoff": int(occurrences[size - 1]),
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["vocabulary_size"])


def full_coverage_curve(gloss_stats: pd.DataFrame) -> pd.DataFrame:
    """Rank against cumulative share, one row per gloss, the plotting input."""
    if gloss_stats.empty:
        return pd.DataFrame(columns=["rank", "gloss", "occurrences",
                                     "cumulative_tokens", "cumulative_percent"])
    ordered = gloss_stats.sort_values("occurrences", ascending=False).reset_index(drop=True)
    cumulative = ordered["occurrences"].cumsum()
    total = int(cumulative.iloc[-1]) or 1
    return pd.DataFrame({
        "rank": range(1, len(ordered) + 1),
        "gloss": ordered["gloss"],
        "occurrences": ordered["occurrences"],
        "n_signers": ordered["n_signers"],
        "cumulative_tokens": cumulative,
        "cumulative_percent": (100 * cumulative / total).round(3),
    })


# ===========================================================================
# TIMING
# ===========================================================================

def _durations_ms(annotations: pd.DataFrame) -> pd.Series:
    start = pd.to_numeric(annotations.get("time_start"), errors="coerce")
    end = pd.to_numeric(annotations.get("time_end"), errors="coerce")
    duration = end - start
    return duration[duration.notna() & (duration > 0)]


def duration_distribution(annotations: pd.DataFrame,
                          label: str = GLOBAL_TAG) -> pd.DataFrame:
    """Annotation duration percentiles, overall and for each marker.

    Percentiles rather than a mean: sign durations are strongly right-skewed,
    a held sign or a fingerspelled name can run several times the median: so a
    mean would describe no actual sign.
    """
    parsed = annotations[annotations["is_parsed"]] if "is_parsed" in annotations.columns \
        else annotations

    def row(name: str, subset: pd.DataFrame) -> Optional[dict]:
        durations = _durations_ms(subset)
        if durations.empty:
            return None
        entry = {"tag": label, "unit": name, "n": int(len(durations)),
                 "mean_ms": round(float(durations.mean()), 1)}
        for percentile in DURATION_PERCENTILES:
            entry[f"p{percentile}_ms"] = round(float(np.percentile(durations, percentile)), 1)
        entry["total_ms"] = round(float(durations.sum()), 1)
        return entry

    rows = [row("all parsed", parsed)]
    if "lexical_only" in parsed.columns:
        rows.append(row("lexical_item (no marker)", parsed[parsed["lexical_only"]]))
    for key in KEY_COLUMNS:
        if key in parsed.columns:
            rows.append(row(key, parsed[nonempty(parsed[key])]))

    return pd.DataFrame([r for r in rows if r])


def signing_rate(annotations: pd.DataFrame,
                 elan_index: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Annotations per minute, per signer.

    Computed from the signer's own annotated span rather than from the file
    duration: in a dialogue each participant is only signing for part of the
    recording, so dividing by file length would halve every rate.
    """
    rows = []
    for signer, group in annotations.groupby(SIGNER_COLUMN, sort=True):
        durations = _durations_ms(group)
        start = pd.to_numeric(group.get("time_start"), errors="coerce")
        end = pd.to_numeric(group.get("time_end"), errors="coerce")
        if start.notna().any() and end.notna().any():
            span_ms = float(end.max() - start.min())
        else:
            span_ms = 0.0
        parsed = group[group["is_parsed"]] if "is_parsed" in group.columns else group
        rows.append({
            "signer": signer,
            "region_code": group["region_code"].iloc[0] if "region_code" in group else "",
            "n_files": int(group["source_file"].nunique()),
            "n_annotations": int(len(group)),
            "n_parsed": int(len(parsed)),
            "signing_span_ms": round(span_ms, 1),
            "annotated_ms": round(float(durations.sum()), 1),
            "annotations_per_minute": round(len(group) / (span_ms / 60000.0), 2)
                                      if span_ms > 0 else float("nan"),
            "median_duration_ms": round(float(durations.median()), 1)
                                  if not durations.empty else float("nan"),
            "n_unique_lexical_items": int(
                parsed.loc[parsed["has_lexical"], LEXICAL_COLUMN].nunique())
                if "has_lexical" in parsed.columns else 0,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# SIGNER-DISJOINT SPLITS
# ===========================================================================

def signer_balance(annotations: pd.DataFrame) -> pd.DataFrame:
    """Token share per signer: how far the corpus is from evenly sampled."""
    parsed = annotations[annotations["is_parsed"]] if "is_parsed" in annotations.columns \
        else annotations
    counts = parsed.groupby(SIGNER_COLUMN).size().sort_values(ascending=False)
    total = int(counts.sum()) or 1
    frame = counts.reset_index(name="n_tokens")
    frame["percent_of_tokens"] = (100 * frame["n_tokens"] / total).round(3)
    frame["cumulative_percent"] = frame["percent_of_tokens"].cumsum().round(3)
    frame["rank"] = range(1, len(frame) + 1)
    return frame


def split_feasibility(annotations: pd.DataFrame,
                      test_fraction: float = DEFAULT_TEST_FRACTION,
                      min_examples: Sequence[int] = (1, 5, 10),
                      label: str = GLOBAL_TAG) -> pd.DataFrame:
    """What a signer-disjoint held-out split would cost.

    Signers are moved into the test set smallest-first until the target token
    fraction is reached. Smallest-first is deliberate and deterministic: it
    holds out as many *different people* as the budget allows, which is what
    makes the test set a test of generalisation across signers rather than a
    test on one atypical person.

    The reported ``oov_token_percent`` is the fraction of test tokens whose
    gloss never appears in training. That number is the honest ceiling on
    accuracy for a closed-vocabulary recogniser, and it is invisible in any
    randomly-split evaluation.
    """
    parsed = annotations[annotations["is_parsed"] & annotations["has_lexical"]]
    if parsed.empty or parsed[SIGNER_COLUMN].nunique() < 2:
        return pd.DataFrame(columns=["tag", "test_fraction", "min_examples",
                                     "n_test_signers", "n_train_signers",
                                     "test_token_percent", "n_train_glosses",
                                     "n_test_glosses", "n_shared_glosses",
                                     "oov_gloss_percent", "oov_token_percent"])

    counts = parsed.groupby(SIGNER_COLUMN).size().sort_values()
    total = int(counts.sum())
    target = total * float(test_fraction)

    test_signers: List[str] = []
    accumulated = 0
    for signer, count in counts.items():
        if accumulated >= target or len(test_signers) >= len(counts) - 1:
            break
        test_signers.append(signer)
        accumulated += int(count)

    test = parsed[parsed[SIGNER_COLUMN].isin(test_signers)]
    train = parsed[~parsed[SIGNER_COLUMN].isin(test_signers)]

    rows = []
    for threshold in min_examples:
        train_counts = train[LEXICAL_COLUMN].value_counts()
        train_vocabulary = set(train_counts[train_counts >= int(threshold)].index)
        test_vocabulary = set(test[LEXICAL_COLUMN].unique())
        shared = test_vocabulary & train_vocabulary
        in_vocabulary = test[LEXICAL_COLUMN].isin(train_vocabulary)
        rows.append({
            "tag": label,
            "test_fraction": round(float(test_fraction), 3),
            "min_examples": int(threshold),
            "n_test_signers": len(test_signers),
            "n_train_signers": int(train[SIGNER_COLUMN].nunique()),
            "test_token_percent": round(100 * accumulated / (total or 1), 2),
            "n_train_glosses": len(train_vocabulary),
            "n_test_glosses": len(test_vocabulary),
            "n_shared_glosses": len(shared),
            "oov_gloss_percent": round(
                100 * (len(test_vocabulary) - len(shared)) / (len(test_vocabulary) or 1), 2),
            "oov_token_percent": round(100 * (~in_vocabulary).sum() / (len(test) or 1), 2),
        })
    return pd.DataFrame(rows)


def examples_per_signer(annotations: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    """For the most frequent glosses, how the examples spread across signers.

    ``max_signer_share`` is the flag to watch: a gloss at 0.8 has most of its
    examples from one person, and a model will learn that person.
    """
    parsed = annotations[annotations["is_parsed"] & annotations["has_lexical"]]
    if parsed.empty:
        return pd.DataFrame(columns=["gloss", "occurrences", "n_signers",
                                     "min_per_signer", "median_per_signer",
                                     "max_per_signer", "max_signer_share"])

    stats = gloss_statistics(parsed).head(int(top_n))
    wanted = set(stats["gloss"])
    per_signer = (parsed[parsed[LEXICAL_COLUMN].isin(wanted)]
                  .groupby([LEXICAL_COLUMN, SIGNER_COLUMN]).size()
                  .reset_index(name="n"))

    grouped = per_signer.groupby(LEXICAL_COLUMN)["n"]
    shape = pd.DataFrame({
        "min_per_signer": grouped.min(),
        "median_per_signer": grouped.median(),
        "max_per_signer": grouped.max(),
        "sum_per_signer": grouped.sum(),
    }).reset_index().rename(columns={LEXICAL_COLUMN: "gloss"})
    shape["max_signer_share"] = (shape["max_per_signer"]
                                 / shape["sum_per_signer"]).round(3)

    merged = stats.merge(shape, on="gloss", how="left")
    return merged[["gloss", "occurrences", "n_signers", "n_regions",
                   "min_per_signer", "median_per_signer", "max_per_signer",
                   "max_signer_share"]]


# ===========================================================================
# MARKER STRUCTURE
# ===========================================================================

def marker_cooccurrence(annotations: pd.DataFrame,
                        label: str = GLOBAL_TAG) -> pd.DataFrame:
    """Pairwise co-occurrence of markers on the same annotation.

    Reported as a count and as a conditional rate in both directions, because
    the two are rarely alike: a marker that almost always accompanies another is
    a candidate for being folded into it, whereas a symmetric pair is two
    independent phenomena that happen to coincide.
    """
    parsed = annotations[annotations["is_parsed"]] if "is_parsed" in annotations.columns \
        else annotations
    present = pd.DataFrame({key: nonempty(parsed[key])
                            for key in KEY_COLUMNS if key in parsed.columns})
    if present.empty:
        return pd.DataFrame(columns=["tag", "key_a", "key_b", "n_both",
                                     "n_a", "n_b", "percent_of_a", "percent_of_b"])

    keys = list(present.columns)
    columns = ["tag", "key_a", "key_b", "n_both", "n_a", "n_b",
               "percent_of_a", "percent_of_b"]
    rows = []
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1:]:
            both = int((present[key_a] & present[key_b]).sum())
            if both == 0:
                continue
            count_a = int(present[key_a].sum())
            count_b = int(present[key_b].sum())
            rows.append({
                "tag": label, "key_a": key_a, "key_b": key_b, "n_both": both,
                "n_a": count_a, "n_b": count_b,
                "percent_of_a": round(100 * both / (count_a or 1), 2),
                "percent_of_b": round(100 * both / (count_b or 1), 2),
            })
    if not rows:
        return pd.DataFrame(columns=columns)
    return (pd.DataFrame(rows).sort_values("n_both", ascending=False)
            .reset_index(drop=True))
