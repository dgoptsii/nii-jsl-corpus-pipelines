"""Exporting the result tables as LaTeX, ready to \\input into the report.

Each file written here is a *complete* ``tabular`` environment, not a fragment.
A fragment ending in ``\\\\`` breaks the ``\\bottomrule`` of whatever includes
it, which is a genuinely difficult error to read once it is three files deep.

Japanese glosses pass through unchanged, so the including document has to be
compiled with XeLaTeX or LuaLaTeX and a CJK-capable main font.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}"}


def escape(value: object) -> str:
    text = "" if value is None else str(value)
    if text.lower() in {"nan", "none", "<na>"}:
        return "--"
    return "".join(ESCAPES.get(character, character) for character in text)


class Literal(str):
    """A cell whose content is already LaTeX and must not be escaped."""


def _format(value: object) -> str:
    if isinstance(value, Literal):
        return str(value)
    if isinstance(value, float):
        if pd.isna(value):
            return "--"
        return f"{value:,.2f}".rstrip("0").rstrip(".") if abs(value) < 1000 \
            else f"{value:,.0f}"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    return escape(value)


#: Cell contents that mean "draw this rule here" rather than "print this".
RULE_COMMANDS = {r"\midrule", r"\addlinespace"}


def write_latex_table(frame: pd.DataFrame,
                      path: Path,
                      columns: Optional[Sequence[str]] = None,
                      headers: Optional[Dict[str, str]] = None,
                      alignment: Optional[str] = None,
                      max_rows: Optional[int] = None,
                      note: str = "") -> Path:
    """Write one complete ``tabular`` block.

    ``headers`` renames columns for display only, so the CSV keeps machine-
    readable names while the table reads as prose.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    subset = frame if columns is None else frame[[c for c in columns if c in frame.columns]]
    if max_rows is not None:
        subset = subset.head(int(max_rows))
    headers = headers or {}

    if alignment is None:
        alignment = "".join("r" if pd.api.types.is_numeric_dtype(subset[c]) else "l"
                            for c in subset.columns)

    # A supplied header is literal LaTeX -- it may contain \% or \textsc{} on
    # purpose -- so only the fallback, which is a raw column name, is escaped.
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
    lines.append(" & ".join(headers[c] if c in headers
                            else escape(c.replace("_", " "))
                            for c in subset.columns) + " \\\\")
    lines.append("\\midrule")
    first_column = subset.columns[0] if len(subset.columns) else None
    for _, row in subset.iterrows():
        # A row whose first cell is a bare rule command is a separator, not
        # data: emit the command on its own line. LaTeX accepts a rule only
        # between rows, so a table needing one cannot express it as a cell.
        cell = row[first_column] if first_column is not None else ""
        if isinstance(cell, Literal) and str(cell).strip() in RULE_COMMANDS:
            lines.append(str(cell).strip())
            continue
        lines.append(" & ".join(_format(row[c]) for c in subset.columns) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if note:
        lines.append("")
        lines.append(f"% {note}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ===========================================================================
# THE TABLES THE REPORT EXPECTS
# ===========================================================================

SUMMARY_COLUMNS = ["tag", "n_files_parsed", "total_recording_hms", "n_signers",
                   "n_annotations", "n_parsed", "n_ambiguous", "n_compound",
                   "n_unique_lexical_items", "n_lexical_items_occurring_once"]

SUMMARY_HEADERS = {
    "tag": "Region", "n_files_parsed": "Files", "total_recording_hms": "Duration",
    "n_signers": "Signers", "n_annotations": "Annot.", "n_parsed": "Parsed",
    "n_ambiguous": "Ambig.", "n_compound": "Comp.",
    "n_unique_lexical_items": "Vocab.", "n_lexical_items_occurring_once": "Hapax",
}

COVERAGE_COLUMNS = ["group", "min_signers", "n_glosses", "occurrences_min",
                    "occurrences_max", "cap", "total_occurrences",
                    "total_occurrences_capped", "percent_of_corpus_tokens"]

COVERAGE_HEADERS = {
    "group": "Vocabulary", "min_signers": "Min.\\ signers", "n_glosses": "Glosses",
    "occurrences_min": "Min.\\ occ.", "occurrences_max": "Max.\\ occ.",
    "cap": "Cap", "total_occurrences": "Total occ.",
    "total_occurrences_capped": "Total (capped)",
    "percent_of_corpus_tokens": "\\% of tokens",
}

KEY_COLUMNS_TABLE = ["key", "kind", "n_annotations", "percent_of_parsed",
                     "n_signers", "n_unique_values"]

KEY_HEADERS = {"key": "Marker", "kind": "Type", "n_annotations": "Count",
               "percent_of_parsed": "\\% of parsed", "n_signers": "Signers",
               "n_unique_values": "Distinct values"}

MOUTH_COLUMNS = ["unit", "n_annotations", "n_with_mouth", "percent_with_mouth",
                 "percent_Mouthing", "percent_MouthGesture", "percent_Others",
                 "n_agree", "n_disagree", "percent_agreement"]

MOUTH_HEADERS = {"unit": "Unit", "n_annotations": "Annot.",
                 "n_with_mouth": "With mouth", "percent_with_mouth": "\\% w/ mouth",
                 "percent_Mouthing": "\\% Mouthing",
                 "percent_MouthGesture": "\\% Gesture",
                 "percent_Others": "\\% Other", "n_agree": "Agree",
                 "n_disagree": "Disagree", "percent_agreement": "\\% agree"}


def global_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The whole-corpus rows of a table that also holds per-region ones.

    Every result table carries a scope column so the regional numbers travel in
    the same CSV. A LaTeX table that dropped that column while keeping the rows
    would silently stack the corpus and each prefecture on top of each other,
    which is the sort of error nobody catches in a typeset PDF.
    """
    for column in ("scope", "tag"):
        if column in frame.columns and "GLOBAL" in set(frame[column]):
            return frame[frame[column] == "GLOBAL"]
    return frame


# ---------------------------------------------------------------------------
# Parse outcome: the one table the report's narrative used to be
# ---------------------------------------------------------------------------

PARSE_OUTCOME_HEADERS = {"item": "", "count": "Count", "percent": "\\%"}


def parse_outcome_table(summary: pd.DataFrame) -> pd.DataFrame:
    """A three-block summary of what the parser made of the corpus.

    Block 1 is a share of every annotation, block 2 of the parsed ones, block 3
    of the row above it. Keeping this as a generated table rather than prose is
    what stops the report and the pipeline drifting apart.
    """
    row = global_rows(summary).iloc[0]

    def share(count, base):
        """A percentage as it should read in print: one decimal, always shown."""
        if not base:
            return Literal("")
        return Literal(f"{100.0 * float(count) / float(base):.1f}")

    def count_of(value):
        return Literal(f"{int(value):,}")

    def heading(text):
        return Literal(r"\itshape " + text)

    total = row["n_annotations"]
    parsed = row["n_parsed"]
    strings = row.get("n_unique_annotation_strings", 0)
    vocabulary = row["n_unique_lexical_items"]

    records = [
        (heading("of all annotations"), Literal(""), Literal("")),
        (Literal("Parsed"), count_of(parsed), share(parsed, total)),
        (Literal("Compound, left unsegmented"), count_of(row["n_compound"]),
         share(row["n_compound"], total)),
        (Literal("Ambiguous, flagged for check"), count_of(row["n_ambiguous"]),
         share(row["n_ambiguous"], total)),
        (Literal(r"\midrule"), Literal(""), Literal("")),
        (heading("of the parsed annotations"), Literal(""), Literal("")),
        (Literal("Plain lexical gloss"), count_of(row["n_lexical_only"]),
         share(row["n_lexical_only"], parsed)),
        (Literal("Lexical item $+$ marker"), count_of(row["n_lexical_with_key"]),
         share(row["n_lexical_with_key"], parsed)),
        (Literal("Marker only, no lexical item"), count_of(row["n_key_only"]),
         share(row["n_key_only"], parsed)),
        (Literal(r"\midrule"), Literal(""), Literal("")),
        # Counts only. A percentage here would need a different denominator on
        # every line -- strings, then vocabulary -- which reads as though the
        # column meant one thing when it meant three.
        (heading("cleaned lexicon"), Literal(""), Literal("")),
        (Literal("Distinct annotation strings"), count_of(strings), Literal("")),
        (Literal("Unique lexical items"), count_of(vocabulary), Literal("")),
        (Literal("Occurring exactly once"), count_of(row["n_lexical_items_occurring_once"]),
         Literal("")),
    ]
    return pd.DataFrame(records, columns=["item", "count", "percent"])


def write_report_tables(tables: Dict[str, pd.DataFrame], folder: Path) -> Dict[str, Path]:
    """Write every table the report section expects, skipping missing ones.

    ``summary`` keeps its per-region rows -- comparing prefectures is the point
    of that table. Every other export is restricted to the whole corpus; the
    regional versions stay in the CSVs.
    """
    folder = Path(folder)
    written: Dict[str, Path] = {}
    plan = [
        ("summary", "tab_corpus_summary.tex", SUMMARY_COLUMNS, SUMMARY_HEADERS,
         None, False),
        ("coverage", "tab_vocabulary_coverage.tex", COVERAGE_COLUMNS,
         COVERAGE_HEADERS, None, True),
        ("keys", "tab_marker_frequency.tex", KEY_COLUMNS_TABLE, KEY_HEADERS,
         None, True),
        ("mouth", "tab_mouth_overlap.tex", MOUTH_COLUMNS, MOUTH_HEADERS, 24, True),
    ]

    summary = tables.get("summary")
    if summary is not None and not summary.empty:
        written["parse_outcome"] = write_latex_table(
            parse_outcome_table(summary), folder / "tab_parse_outcome.tex",
            columns=["item", "count", "percent"], headers=PARSE_OUTCOME_HEADERS,
            alignment="lrr")

    for name, filename, columns, headers, max_rows, only_global in plan:
        frame = tables.get(name)
        if frame is None or frame.empty:
            continue
        if only_global:
            frame = global_rows(frame)
        frame, note = _prepare(name, frame)
        if frame.empty:
            continue
        written[name] = write_latex_table(frame, folder / filename,
                                          columns=columns, headers=headers,
                                          max_rows=max_rows, note=note)
    return written


def _prepare(name: str, frame: pd.DataFrame):
    """Display tidying that belongs in the typeset table, not in the CSV."""
    frame = frame.copy()
    note = ""

    if name == "keys":
        # A column of zeros is not worth a printed row; the inventory of unused
        # markers is worth one sentence, so it goes in the note.
        unused = sorted(frame.loc[frame["n_annotations"] == 0, "key"].str.upper())
        frame = frame[frame["n_annotations"] > 0]
        frame["key"] = frame["key"].str.upper()
        if unused:
            note = f"markers never used: {', '.join(unused)}"

    if name == "mouth":
        frame["unit"] = [u if " " in u else u.upper() for u in frame["unit"]]

    return frame, note
