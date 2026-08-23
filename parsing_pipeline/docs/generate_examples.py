#!/usr/bin/env python3
"""Generate the worked-example tables used by ``parsing_rules.tex``.

The examples are produced by actually running the parser, so the documentation
can never drift away from the implementation. Re-run this script whenever a
parsing rule changes:

    python docs/generate_examples.py

One file is written per example group into ``docs/generated/`` and pulled into
the report by ``\\ExampleTable{<group>}``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import parsing  # noqa: E402
from config import METADATA_COLUMNS  # noqa: E402

GENERATED_DIR = DOCS_DIR / "generated"

#: group id -> (caption, [annotation, ...])
GROUPS: Dict[str, Tuple[str, Sequence[str]]] = {
    "normalisation": (
        "Global normalisation applied before any keyword is read",
        [
            "/猫",
            "//猫",
            "~猫",
            "猫！",
            "「猫」",
            "猫、犬",
            "猫＋犬",
            "猫&犬",
            "猫／犬",
        ],
    ),
    "pt": (
        "PT --- pointing",
        [
            "pt",
            "pt2",
            "PT1＝歯",
            "pt:体",
            "pt2:ひよこ",
            "PT3みんな",
            "PT(アニメ)",
            "PT3(二つ目)",
            "PT(妻",
            "狙う(pt)",
            "両手:pt",
            "両手:pt3",
        ],
    ),
    "dw": (
        "DW / DR --- depicting words",
        [
            "dw:5種類",
            "同じdw",
            "同じdr",
            "cl:dw",
            "cl:dr",
            "pt3(dw:5種類)",
            "pt:dw(みんな)",
            "pt3dw",
            "pt3dw(4つの具材)",
            "ptdw",
            "ptdw:これら",
            "ptdw(みんな)",
        ],
    ),
    "fs": (
        "FS --- fingerspelling",
        [
            "fs:あ",
            "FSあ",
            "fs:お(M:ロースポーク)",
            "fs:あ+aw:い",
        ],
    ),
    "aw": (
        "AW --- air writing",
        [
            "aw:あれ",
            "AW あれ",
        ],
    ),
    "hands": (
        "LH / RH --- one-handed and two-handed forms",
        [
            "LH(家)+RH(裏)",
            "L:娘R:娘",
            "RH:猫",
            "右手:楽しい",
            "左:猫",
            "cl:家の裏{家(LH)+裏(RH)}",
            "楽しい(右手だけ)",
            "美味しい(右手ver)",
            "二つ目:LH:人差し指",
        ],
    ),
    "cl": (
        "CL --- classifiers",
        [
            "cl:猫",
            "CL(猫)",
            "cl",
            "cl:人型:(ges:へえ)",
            "cl:人型(説明):(ges:)",
            "cl:人型:(猫)(ges:あ)",
        ],
    ),
    "m": (
        "M --- mouth actions (口形 / 口型)",
        [
            "M:おる",
            "口形:おる",
            "口型:おる",
            "(M:あ)",
            "cl:包丁で切る(口型:きった)",
            "cl:包丁で切る(口型:un(きった)",
        ],
    ),
    "ges": (
        "GES --- gestures",
        [
            "ges:へえ",
            "GES(あ)",
            "ges",
        ],
    ),
    "nmm": (
        "NMM --- non-manual markers",
        [
            "nmm:眉上げ",
            "NMM(neg)",
            "nmm",
            "nod",
        ],
    ),
    "rep": (
        "REP --- repetition",
        [
            "そうだ(rep)",
            "食べる rep2",
            "rep:3",
            "猫(2rep)",
            "RH:男＊２",
            "LH:男*3",
            "RH(男)*2",
            "広がる(repeat)",
            "pt(rep2)",
        ],
    ),
    "flags": (
        "STOP, HOLD, KEEP, INDEX --- boolean articulation flags",
        [
            "猫 stop",
            "別(hold)",
            "猫 keep",
            "見る(index)",
        ],
    ),
    "blocking": (
        "D, FAL, UN --- blocking markers",
        [
            "d",
            "D:fs:て(RH)",
            "fal",
            "FAL:fs:あ",
            "cl:猫の形(fal)",
            "un",
            "UN:RH:猫",
            "UN(FAL)",
            "FAL(UN)",
        ],
    ),
    "qm": (
        "QM --- question marking",
        [
            "猫?",
            "猫？",
            "猫 qm",
        ],
    ),
    "pastneg": (
        "PAST and NEG --- grammatical flags",
        [
            "食べる PAST",
            "食べる NEG",
            "食べる past neg",
        ],
    ),
    "ambiguity": (
        "Ambiguity --- rows routed to manual review",
        [
            "猫/犬/鳥",
            "/猫/",
            "ー",
            "ーー",
            "pt//",
        ],
    ),
    "combinations": (
        "Combined annotations",
        [
            "LH:cl:人型+RH:pt2",
            "nmm:眉上げ+猫",
            "pt2:ひよこ+cl:鳥の形",
            "cl:家の裏{家(LH)+裏(RH)}+pt",
            "食べる PAST+nmm:眉上げ",
        ],
    ),
}


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def describe(row: Dict[str, str]) -> str:
    """Render the non-empty parsed columns of a row as ``key=value`` pairs."""
    parts: List[str] = []

    for column, value in row.items():
        if column in METADATA_COLUMNS or column == "annotation":
            continue
        if not value:
            continue
        parts.append(rf"\pcol{{{escape_latex(column)}}}\,=\,{escape_latex(value)}")

    return r",\; ".join(parts) if parts else r"\emph{(all columns empty)}"


def render_group(group_id: str, caption: str, annotations: Sequence[str]) -> str:
    parsing.set_exceptions([])

    lines = [
        "% Generated by docs/generate_examples.py -- do not edit by hand.",
        r"\begin{longtable}{@{}p{0.34\linewidth}p{0.62\linewidth}@{}}",
        rf"\caption{{{escape_latex(caption)}}}\label{{tab:ex-{group_id}}}\\",
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

    for annotation in annotations:
        row = parsing.parse_annotation(annotation)
        lines.append(
            rf"\anno{{{escape_latex(annotation)}}} & {describe(row)} \\"
        )

    lines.append(r"\end{longtable}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    for group_id, (caption, annotations) in GROUPS.items():
        target = GENERATED_DIR / f"ex_{group_id}.tex"
        target.write_text(render_group(group_id, caption, annotations), encoding="utf-8")
        print(f"wrote {target.relative_to(REPO_ROOT)} ({len(annotations)} examples)")

    print(f"\n{len(GROUPS)} example tables generated in {GENERATED_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
