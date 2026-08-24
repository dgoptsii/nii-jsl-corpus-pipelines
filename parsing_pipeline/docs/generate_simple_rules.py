#!/usr/bin/env python3
"""Generate `parsing_rules_simple.tex`: a compact keyword reference.

One section per annotation keyword: what it means, which column it fills, and a
table of worked examples. Every example is produced by actually running the
parser, so the document cannot drift away from the code.

    python3 docs/generate_simple_rules.py
    cd docs && latexmk -xelatex parsing_rules_simple.tex

The output is a single self-contained .tex file - no \\input, no generated/
folder - so it can be dropped straight into Overleaf or edited by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import parsing  # noqa: E402
from config import METADATA_COLUMNS  # noqa: E402

OUTPUT = DOCS_DIR / "parsing_rules_simple.tex"


# ---------------------------------------------------------------------------
# The keyword reference: (heading, columns filled, one-line note, examples)
# ---------------------------------------------------------------------------

SECTIONS: List[Dict[str, object]] = [
    {
        "title": "Normalisation applied to every annotation",
        "columns": "",
        "note": "These textual clean-ups happen before any keyword is read. "
                "A leading slash, the first tilde and all exclamation marks are removed; "
                "\\anno{\\&}, \\anno{＋}, \\anno{／} and an internal \\anno{/} all become \\anno{+}; "
                "\\anno{DR} is normalized as \\kw{DW}; \\anno{REPEAT} is normalized as \\kw{REP}; "
                "\\anno{口形} and \\anno{口型} are normalized as \\kw{M}.",
        "examples": ["/猫", "//猫", "~猫", "猫！", "「猫」", "猫、犬", "猫,犬",
                     "猫＋犬", "猫&犬", "猫／犬"],
    },
    {
        "title": "PT: pointing",
        "columns": "pt",
        "note": "The lexical item of a pointing annotation is always \\anno{pt} + its number. "
                "Japanese material written next to PT names the \\emph{referent} being pointed at, "
                "not a signed word, so it is dropped.",
        "examples": ["pt", "pt2", "PT1＝歯", "pt:体", "pt2:ひよこ", "PT3みんな",
                     "PT(アニメ)", "PT3(二つ目)", "PT(妻", "狙う(pt)",
                     "両手:pt", "両手:pt3", "pt3(dw:5種類)"],
    },
    {
        "title": "DW: depicting word (DR is the same keyword)",
        "columns": "dw",
        "note": "\\kw{DR} is rewritten to \\kw{DW} everywhere, including inside compact forms. "
                "A depicting word is a sign, so its value is kept as lexical material.",
        "examples": ["dw:5種類", "同じdw", "同じdr", "cl:dw", "cl:dr",
                     "pt:dw(みんな)", "pt3dw", "pt3dw(4つの具材)",
                     "ptdw", "ptdw:これら", "ptdw(みんな)"],
    },
    {
        "title": "FS: fingerspelling",
        "columns": "fs",
        "note": "Fingerspelled material is lexical, so it is recorded in \\pcol{fs} "
                "\\emph{and} kept in \\pcol{lexical\\_item}.",
        "examples": ["fs:あ", "FSあ", "fs:お(M:ロースポーク)", "fs:あ+aw:い"],
    },
    {
        "title": "AW: air writing",
        "columns": "aw",
        "note": "Behaves exactly like \\kw{FS}: recorded in its own column and kept as lexical material.",
        "examples": ["aw:あれ", "AW あれ"],
    },
    {
        "title": "LH and RH: hand attribution",
        "columns": "lh, rh",
        "note": "Markers: \\anno{LH} \\anno{L} \\anno{左手} \\anno{左} and "
                "\\anno{RH} \\anno{R} \\anno{右手} \\anno{右}. "
                "Values inside a hand stream are parsed too, so nested markers are still recognised.",
        "examples": ["LH(家)+RH(裏)", "L:娘R:娘", "RH:猫", "右手:楽しい", "左:猫",
                     "cl:家の裏{家(LH)+裏(RH)}", "楽しい(右手だけ)", "美味しい(右手ver)",
                     "二つ目:LH:人差し指"],
    },
    {
        "title": "CL: classifier / depiction",
        "columns": "cl",
        "note": "The value ends at the next top-level separator. "
                "A gesture note inside a classifier is lifted into \\pcol{ges}.",
        "examples": ["cl:猫", "CL(猫)", "cl", "cl:人型:(ges:へえ)",
                     "cl:人型(説明):(ges:)", "cl:人型:(猫)(ges:あ)"],
    },
    {
        "title": "M: mouth action (口形 / 口型)",
        "columns": "m",
        "note": "Separator may be \\anno{:} \\anno{：} \\anno{=} \\anno{＝} or the particle \\anno{は}. "
                "Mouth notes are extracted before the blocking markers are checked.",
        "examples": ["M:おる", "口形:おる", "口型:おる", "(M:あ)",
                     "cl:包丁で切る(口型:きった)", "cl:包丁で切る(口型:un(きった)"],
    },
    {
        "title": "GES: gesture",
        "columns": "ges",
        "note": "Recognised as \\anno{ges:value}, \\anno{GES(value)}, or a bare marker (\\anno{TRUE}).",
        "examples": ["ges:へえ", "GES(あ)", "ges"],
    },
    {
        "title": "NMM: non-manual marker (and NOD)",
        "columns": "nmm",
        "note": "A value of \\anno{neg} also sets \\pcol{neg}. \\anno{nod} is shorthand for "
                "\\pcol{nmm}~=~\\anno{nod}. NMM values are \\emph{not} counted as lexical material.",
        "examples": ["nmm:眉上げ", "NMM(neg)", "nmm", "nod", "nmm:眉上げ+猫"],
    },
    {
        "title": "REP: repetition (REPEAT and $\\times n$ are the same)",
        "columns": "rep",
        "note": "Value format: \\anno{word(count;hand)}, where hand is \\anno{L}, \\anno{R} or "
                "\\anno{N} for unspecified. A missing count means 1.",
        "examples": ["そうだ(rep)", "食べる rep2", "rep:3", "猫(2rep)",
                     "RH:男＊２", "LH:男*3", "RH(男)*2", "広がる(repeat)", "pt(rep2)"],
    },
    {
        "title": "STOP, HOLD, KEEP and INDEX: articulation flags",
        "columns": "stop, hold, keep, index",
        "note": "All boolean (\\anno{TRUE}). Recognised bare, with a colon, or in parentheses "
                "attached to a sign. The sign itself survives into \\pcol{lexical\\_item}.",
        "examples": ["猫 stop", "別(hold)", "猫 keep", "見る(index)"],
    },
    {
        "title": "D, FAL and UN: blocking markers",
        "columns": "d, fal, un",
        "note": "\\kw{D} = delete, \\kw{FAL} = false start, \\kw{UN} = unclear. "
                "When any of these fires, \\textbf{every other column is cleared}: "
                "the marker says the content of the annotation cannot be trusted.",
        "examples": ["d", "D:fs:て(RH)", "fal", "FAL:fs:あ", "cl:猫の形(fal)",
                     "un", "UN:RH:猫", "UN(FAL)", "FAL(UN)"],
    },
    {
        "title": "QM: question marking",
        "columns": "qm",
        "note": "Set by an explicit \\kw{QM} or by \\anno{?} / \\anno{？} anywhere in the annotation. "
                "Not set when a blocking marker is present.",
        "examples": ["猫?", "猫？", "猫 qm"],
    },
    {
        "title": "PAST and NEG: grammatical flags",
        "columns": "past, neg",
        "note": "Matched as standalone Latin words, so they do not fire inside a longer word. "
                "The marker is removed and the sign is preserved.",
        "examples": ["食べる PAST", "食べる NEG", "食べる past neg"],
    },
    {
        "title": "Compound signs: \\anno{<} \\anno{>}",
        "columns": "compound",
        "note": "An opening bracket starts a group that runs to the closing bracket. "
                "Every row in the group is copied verbatim into \\pcol{compound} and is not parsed, "
                "because a compound is not the sum of its parts.",
        "examples": ["<家", "裏>"],
    },
    {
        "title": "Ambiguity: rows sent to manual review",
        "columns": "ambiguous",
        "note": "A row is ambiguous if it has more than one internal slash, normalises to nothing, "
                "consists only of \\anno{ー}, or still contains unparsed non-Japanese text. "
                "All other columns are then cleared.",
        "examples": ["猫/犬/鳥", "/猫/", "ー", "ーー", "pt//"],
    },
    {
        "title": "Combined annotations",
        "columns": "",
        "note": "Each \\anno{+} stream is parsed independently and the results merged. "
                "\\pcol{lexical\\_item} collects every sign found, separated by \\anno{;}.",
        "examples": ["LH:cl:人型+RH:pt2", "pt2:ひよこ+cl:鳥の形",
                     "cl:家の裏{家(LH)+裏(RH)}+pt", "食べる PAST+nmm:眉上げ",
                     "aw:あれ+cl:人型+ges:あ"],
    },
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def describe(row: Dict[str, str]) -> str:
    parts: List[str] = []
    for column, value in row.items():
        if column in METADATA_COLUMNS or column == "annotation" or not value:
            continue
        parts.append(rf"\pcol{{{escape_latex(column)}}}\,=\,{escape_latex(value)}")
    return r",\; ".join(parts) if parts else r"\emph{(all columns empty)}"



COLUMN_MEANINGS = {
    "lexical_item": "Every lexical item found in the annotation, separated by \\anno{;}",
    "pt": "Pointing: the pointing number, \\anno{0} when unnumbered, or \\anno{dw}",
    "dw": "Depicting word",
    "fs": "Fingerspelling",
    "aw": "Air writing",
    "lh": "What the left hand articulated",
    "rh": "What the right hand articulated",
    "d": "Deleted / disregarded (blocking)",
    "cl": "Classifier / depiction",
    "m": "Mouth action",
    "ges": "Gesture",
    "nmm": "Non-manual marker",
    "rep": "Repetition, as \\anno{word(count;hand)}",
    "stop": "Movement stopped",
    "hold": "Handshape held",
    "index": "Indexing",
    "keep": "Hand kept in place",
    "fal": "False start (blocking)",
    "un": "Unclear / unidentified (blocking)",
    "qm": "Question marking",
    "past": "Past marker",
    "neg": "Negation marker",
    "compound": "Verbatim annotation, for \\anno{<...>} compound groups",
    "ambiguous": "Verbatim annotation, for rows needing manual review",
}

METADATA_MEANINGS = {
    "speaker_id": "Participant, taken from the Word tier name",
    "time_start": "Annotation start, in milliseconds",
    "time_end": "Annotation end, in milliseconds",
    "annotation": "The original string, verbatim and unmodified",
}


def render_column_table() -> str:
    """Build the schema table from the actual column list in config.py."""
    from config import PARSED_COLUMNS

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{The columns produced for every annotation}",
        r"\label{tab:schema}",
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        r"\textbf{Column} & \textbf{Meaning} \\",
        r"\midrule",
        r"\multicolumn{2}{@{}l@{}}{\itshape Metadata} \\",
    ]
    for column, meaning in METADATA_MEANINGS.items():
        lines.append(rf"\pcol{{{escape_latex(column)}}} & {meaning} \\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{@{}l@{}}{\itshape Parsed output} \\")
    for column in PARSED_COLUMNS:
        if column == "annotation":
            continue
        meaning = COLUMN_MEANINGS.get(column, "")
        lines.append(rf"\pcol{{{escape_latex(column)}}} & {meaning} \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def render_table(examples: Sequence[str]) -> str:
    lines = [
        r"\begin{longtable}{@{}p{0.33\linewidth}p{0.63\linewidth}@{}}",
        r"\toprule",
        r"\textbf{Annotation} & \textbf{Parsed columns} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Annotation} & \textbf{Parsed columns} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for annotation in examples:
        row = parsing.parse_annotation(annotation)
        lines.append(rf"\anno{{{escape_latex(annotation)}}} & {describe(row)} \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Document front matter. Edit the four \newcommand values below (or the
# corresponding lines in the generated .tex) to put your own names on it.
# ---------------------------------------------------------------------------

PREAMBLE = r"""% !TEX program = xelatex
%
% Annotation Parsing Rules - reference report.
%
% GENERATED FILE - produced by docs/generate_simple_rules.py, which runs the
% parser itself to build every example table. After changing a parsing rule,
% re-run that script rather than editing the tables here by hand.
%
% Build with XeLaTeX (required for the Japanese):
%   latexmk -xelatex parsing_rules_simple.tex
% In Overleaf: Menu -> Compiler -> XeLaTeX.
%
\documentclass[11pt,a4paper]{article}

%% ---------------------------------------------------------------------------
%% EDIT THESE
%% ---------------------------------------------------------------------------
\newcommand{\ReportAuthor}{Daria Goptsii}
\newcommand{\ReportGroup}{Research group / laboratory}
\newcommand{\ReportInstitution}{National Institute of Informatics}
\newcommand{\ReportVersion}{1.0}

\usepackage{fontspec}

%% Japanese-capable fonts: the first family that is actually installed wins.
%% 1. Noto CJK JP (Linux, Overleaf)  2. Hiragino (macOS)  3. Yu Mincho (Windows)
%% The MONOSPACE font must also have Japanese glyphs - the annotation strings
%% are Japanese, and a font like Menlo makes them silently disappear.
\IfFontExistsTF{Noto Serif CJK JP}{
  \setmainfont{Noto Serif CJK JP}
  \setsansfont{Noto Sans CJK JP}
  \setmonofont{Noto Sans Mono CJK JP}[Scale=0.92]
}{\IfFontExistsTF{Hiragino Mincho ProN}{
  \setmainfont{Hiragino Mincho ProN}
  \setsansfont{Hiragino Sans}
  \setmonofont{Hiragino Sans}[Scale=0.92]
}{\IfFontExistsTF{Yu Mincho}{
  \setmainfont{Yu Mincho}
  \setsansfont{Yu Gothic}
  \setmonofont{Yu Gothic}[Scale=0.92]
}{
  \GenericError{}{No Japanese font found. Install Noto CJK, or edit the font
    block at the top of this file to name a Japanese font you do have}{}{}
}}}

\usepackage[margin=2.5cm,headheight=14pt]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{float}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[hidelinks]{hyperref}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\itshape Annotation Parsing Rules}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\titleformat{\section}{\large\bfseries}{\thesection}{0.6em}{}
\titlespacing*{\section}{0pt}{1.5em}{0.5em}

\newcommand{\anno}[1]{\texttt{#1}}          % an annotation string
\newcommand{\pcol}[1]{\textsf{\small #1}}   % a parsed column
\newcommand{\kw}[1]{\textbf{\textsf{#1}}}   % a keyword
\newcommand{\fills}[1]{\noindent\textbf{Column:} \pcol{#1}\par\smallskip}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.4em}

%% Compact table of contents: article.cls puts 1em of vertical space before
%% every section entry, which spreads 19 short entries over two pages.
\makeatletter
\renewcommand*\l@section[2]{%
  \ifnum \c@tocdepth >\z@
    \begingroup
      \setlength\@tempdima{2.2em}%
      \parindent\z@ \rightskip\@pnumwidth \parfillskip-\@pnumwidth
      \leavevmode
      \advance\leftskip\@tempdima \hskip -\leftskip
      #1\nobreak\leaders\hbox{$\m@th\mkern 4.5mu\cdot\mkern 4.5mu$}\hfill
      \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
    \endgroup
  \fi}
\makeatother

\begin{document}

%% ---------------------------------------------------------------------------
%% Title page
%% ---------------------------------------------------------------------------
\begin{titlepage}
\thispagestyle{empty}
\centering
\vspace*{3cm}

{\Huge\bfseries Annotation Parsing Rules\par}
\vspace{0.8cm}
{\Large Specification of the rule-based parser for\\
Japanese Sign Language Word-tier annotations\par}

\vspace{2.5cm}
{\large \ReportAuthor\par}
\vspace{0.3cm}
{\ReportGroup\par}
{\ReportInstitution\par}

\vspace{2.5cm}
\begin{tabular}{@{}ll@{}}
\toprule
Document version & \ReportVersion \\
Pipeline & \texttt{elan-parsing-pipeline} \\
Status & Reference specification \\
\bottomrule
\end{tabular}

\end{titlepage}

%% ---------------------------------------------------------------------------
%% Front matter
%% ---------------------------------------------------------------------------
\setcounter{page}{1}

\section*{Abstract}
\addcontentsline{toc}{section}{Abstract}

Annotators record Japanese Sign Language on the ELAN Word tier as free text
that mixes Japanese lexical material with a set of Latin keywords. A single
annotation may state what was signed, which hand signed it, whether the mouth
was involved, whether the sign was repeated, and whether the annotator was
uncertain, which makes it difficult to query the data or perform statistical
analysis. This report specifies the rule-based parser that converts each such
string into a fixed set of structured columns. Each keyword is documented in
its own section: what it means, which column it fills, and a table of worked
examples showing exactly what the parser produces. The examples are generated
by running the parser over the listed inputs, so they are guaranteed to match
the implementation at the time this document was built.

\vspace{1em}
{\setlength{\parskip}{0pt}\tableofcontents}

\clearpage

%% ---------------------------------------------------------------------------
%% Introduction
%% ---------------------------------------------------------------------------
\section{Scope and conventions}

The parser reads every Japanese Word tier (a tier whose identifier contains
\anno{Word-jp}) of an ELAN file, converts each annotation into the columns
listed in Table~\ref{tab:schema}, and writes the result back into a copy of the
ELAN document as child tiers.

__COLUMN_TABLE__

\begin{table}[H]
\centering
\caption{Value conventions used in every table in this report}
\label{tab:conventions}
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Convention} & \textbf{Meaning} \\
\midrule
empty column & the rule was not triggered \\
\anno{TRUE} & the marker was present but carried no lexical item \\
\anno{;} & separates several values within one column \\
\pcol{lexical\_item} & every lexical item found in the annotation \\
\pcol{annotation} & the original string, always kept verbatim \\
\bottomrule
\end{tabular}
\end{table}

Sections \ref{sec:first}--\ref{sec:last} document the keywords in the order the
parser applies them.
"""


def main() -> int:
    parsing.set_exceptions([])
    parsing.configure_keywords(unknown_keywords=[], known_unknown_keywords=[])

    chunks = [PREAMBLE.replace("__COLUMN_TABLE__", render_column_table())]

    for index, section in enumerate(SECTIONS):
        chunks.append("")
        chunks.append(rf"\section{{{section['title']}}}")

        if index == 0:
            chunks.append(r"\label{sec:first}")
        if index == len(SECTIONS) - 1:
            chunks.append(r"\label{sec:last}")

        if section["columns"]:
            chunks.append(rf"\fills{{{escape_latex(str(section['columns']))}}}")

        if section["note"]:
            chunks.append(str(section["note"]))

        chunks.append("")
        chunks.append(render_table(list(section["examples"])))

    chunks.append("")
    chunks.append(r"\end{document}")

    OUTPUT.write_text("\n".join(chunks) + "\n", encoding="utf-8")

    total = sum(len(list(s["examples"])) for s in SECTIONS)
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  {len(SECTIONS)} sections, {total} generated examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
