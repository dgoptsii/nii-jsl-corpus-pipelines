#!/usr/bin/env python3
"""Rebuild ``signers.csv`` from the annotation-information spreadsheet.

The spreadsheet is the authority for who each signer is: the tier labels in the
corpus contradict it for 23 signers (report 1), so the pipelines read age,
gender and handedness only from here. Re-run this when the spreadsheet changes
rather than editing the CSV by hand.

    python3 signers_from_xlsx.py "CORPUS/進捗状況-Annotation information.xlsx" \
        -output_folder ../signing-space-analysis-pipeline/input_lists

Writes ``signers.csv``, and ``signers.txt`` recording every judgement the sheet
forced, since three of its handedness spellings are not simply left or right.

Handedness, the one column that is interpreted rather than copied:

    R         right
    L         left, and the signer's signing space is mirrored
    R(L)      mixed dominance. Leads with the right, so NOT mirrored: mirroring
    R (L)     would move their dominant hand to the wrong side of the space
    R L
    ?         unknown, assumed right, which is also the default for a signer
              the file does not list at all

``hand_in_sheet`` keeps the original cell beside the interpretation, so a
reader can always see what the spreadsheet actually said.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_corpus import PREFECTURE_CODES, read_signer_sheet   # noqa: E402

#: Prefix -> the prefecture's English name, for the ``region`` column.
REGION_NAMES = {
    "GM": "Gunma", "NR": "Nara", "NS": "Nagasaki", "FO": "Fukuoka",
    "IS": "Ishikawa", "TY": "Toyama", "IK": "Ibaraki",
}

COLUMNS = ["signer_id", "handedness", "age", "gender", "region", "region_code",
           "pair", "hand_in_sheet", "note"]


def interpret_hand(cell: str) -> tuple:
    """``(handedness, note)`` from one dominant-hand cell."""
    text = str(cell or "").strip().upper()
    if text == "L":
        return "left", "mirrored"
    if text in {"", "?", "-"}:
        return "right", "unknown in sheet - assumed right"
    if text != "R":
        # R(L), R (L), R L: leads right, so it is not mirrored.
        return "right", f"mixed ({cell.strip()}) - leads right, NOT mirrored"
    return "right", ""


def build_rows(signers: dict) -> list:
    """One row per signer, in prefecture then number order.

    The sheet writes the pair label (``1-2``) only on the first row of a pair,
    the way it writes a shared age only once, so it is carried down to the
    partner. Both signers of a pair are in the same recordings, so a blank
    there would lose the link.
    """
    rows = []
    pair_label = ""
    prefecture = ""
    for signer_id in sorted(signers, key=lambda s: (s[:2], int(s.split("_")[1]))):
        entry = signers[signer_id]
        if entry["prefecture"] != prefecture:
            prefecture, pair_label = entry["prefecture"], ""
        if entry["pair"].strip():
            pair_label = entry["pair"].strip()
        handedness, note = interpret_hand(entry["dominant_hand"])
        rows.append({
            "signer_id": signer_id,
            "handedness": handedness,
            "age": entry["age"],
            "gender": entry["gender"],
            "region": REGION_NAMES.get(entry["prefecture"], entry["prefecture"]),
            "region_code": entry["prefecture"],
            "pair": pair_label,
            "hand_in_sheet": entry["dominant_hand"],
            "note": note,
        })
    return rows


def write_notes(rows: list, path: Path, source: Path) -> None:
    judged = [r for r in rows if r["note"]]
    missing = [r for r in rows if not str(r["age"]).strip()
               or not str(r["gender"]).strip()]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"signers.csv rebuilt from {source.name}\n")
        handle.write(f"{len(rows)} signers\n\n")
        handle.write("Handedness cells that needed a judgement:\n")
        for r in judged:
            handle.write(f"  {r['signer_id']:8} sheet says {r['hand_in_sheet']!r:10}"
                         f" -> {r['handedness']:5}  ({r['note']})\n")
        if not judged:
            handle.write("  none\n")
        handle.write("\nSigners with no age or no gender in the sheet:\n")
        for r in missing:
            handle.write(f"  {r['signer_id']:8} age={r['age']!r} gender={r['gender']!r}\n")
        if not missing:
            handle.write("  none\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild signers.csv from the annotation-information spreadsheet.")
    parser.add_argument("sheet_file", type=Path,
                        help="The .xlsx. Its first sheet holds the signer blocks.")
    parser.add_argument("--sheet-name", dest="sheet_name", default=None,
                        help="Sheet to read. Default: the first one.")
    parser.add_argument("-output_folder", "--output-folder", dest="output_folder",
                        type=Path, default=Path("input_lists"),
                        help="Where to write. Default: input_lists")
    args = parser.parse_args(argv)

    if not args.sheet_file.exists():
        print(f"ERROR: {args.sheet_file} does not exist", file=sys.stderr)
        return 1

    signers = read_signer_sheet(args.sheet_file, args.sheet_name)
    if not signers:
        print("ERROR: no signers found. Is this the right sheet?", file=sys.stderr)
        print(f"       Prefecture headings recognised: "
              f"{', '.join(sorted(set(PREFECTURE_CODES)))}", file=sys.stderr)
        return 1

    rows = build_rows(signers)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_folder / "signers.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    notes_path = args.output_folder / "signers.txt"
    write_notes(rows, notes_path, args.sheet_file)

    left = sum(1 for r in rows if r["handedness"] == "left")
    print(f"{len(rows)} signers -> {csv_path}")
    print(f"  {left} left-handed (mirrored), "
          f"{sum(1 for r in rows if r['note'] and r['handedness'] == 'right')} "
          f"right-handed by judgement")
    print(f"  judgements listed in {notes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
