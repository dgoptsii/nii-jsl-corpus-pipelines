#!/usr/bin/env python3
"""Turn the corpus audit into a short list of tier names to check or correct.

``audit_corpus.py`` reports every finding, which runs to a few hundred rows
because the same signer recurs across their recordings and across duplicate
copies. This collapses that into one row per decision a person actually has to
make, ordered by how much the problem costs.

    python3 build_todo.py corpus_audit signers.csv -output_folder corpus_audit

Identity, empty-tier and duplicate-tier problems stay per tier, since each is a
separate edit in ELAN. Age and gender disagreements are per signer, since the
decision ("is FO_01 60 or 70?") is made once and then applied to every tier
that names them.

Writes ``tier_names_to_check.csv``: category, signer, recording, tier, problem,
what the spreadsheet says, whether the recording is in the current analysis
selection, the suggested action, and an empty ``done`` column.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

#: Problems settled tier by tier, with the order they should be worked in.
PER_TIER = {
    "signer from another recording":      (1, "Identity"),
    "several spellings in one file":      (3, "One person, two tiers"),
    "tier name and PARTICIPANT disagree": (4, "Name vs PARTICIPANT"),
}

#: Problems settled once per signer, however many tiers name them.
PER_SIGNER = {
    "gender contradicts the spreadsheet": (5, "Gender vs sheet"),
    "age contradicts the spreadsheet":    (6, "Age vs sheet"),
}

ACTION = {
    "Identity":              "Confirm whose signing this is, then rename or delete the tier",
    "Unreadable gloss":      "Move the content into the Word-jp tier, or rename the variant tier",
    "One person, two tiers": "Decide which tier is correct, then merge or delete the other",
    "Name vs PARTICIPANT":   "Make the tier name and the PARTICIPANT attribute agree",
    "Gender vs sheet":       "Check the person, then correct either the tier names or the spreadsheet",
    "Age vs sheet":          "Check the person, then correct either the tier names or the spreadsheet",
}

COLUMNS = ["category", "signer", "recording", "tier", "problem", "spreadsheet",
           "in_selection", "action", "done"]


def read_csv(path: Path) -> list:
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build_rows(audit_folder: Path, signers_file: Path) -> list:
    sheet = {row["signer_id"]: row for row in read_csv(signers_file)}
    inconsistencies = read_csv(audit_folder / "signer_id_inconsistencies.csv")
    variants = read_csv(audit_folder / "word_variants_by_file.csv")
    selected = {row["recording"] for row
                in read_csv(audit_folder / "finished_and_ongoing.csv")}

    def spreadsheet_says(signer_id: str) -> str:
        row = sheet.get(signer_id)
        return f"age {row['age']}, {row['gender']}" if row else "not in spreadsheet"

    rows = []
    per_tier: "collections.OrderedDict" = collections.OrderedDict()
    per_signer = collections.defaultdict(
        lambda: {"recordings": set(), "tiers": set(), "detail": ""})

    # A Word-jp tier that exists but is empty is not in the inconsistencies
    # file, because nothing about the name is wrong; the gloss is just on a
    # variant tier the parser does not read.
    for row in variants:
        if row.get("plain_word_jp_annotations", "").strip() != "0":
            continue
        recording = row["recording"]
        prefix, numbers = recording.split("_")[0], recording.split("_")[1]
        rows.append(dict(
            rank=2, category="Unreadable gloss", recording=recording,
            tier="all Word-jp tiers are empty",
            signer=" and ".join(f"{prefix}_{n}" for n in numbers.split("-")),
            problem=(f"gloss is on {row['variants']} instead; "
                     f"{row['variant_annotations']} annotations unread"),
            spreadsheet="",
            in_selection="yes" if recording in selected else "no"))

    for row in inconsistencies:
        problem = row["problem"]
        if problem in PER_TIER:
            rank, category = PER_TIER[problem]
            for tier in row["tiers"].split():
                key = (category, row["recording"], tier)
                per_tier.setdefault(key, dict(
                    rank=rank, category=category, recording=row["recording"],
                    tier=tier, signer=row["signer"], problem=row["detail"],
                    spreadsheet=spreadsheet_says(row["signer"]),
                    in_selection="yes" if row["recording"] in selected else "no"))
        elif problem in PER_SIGNER:
            rank, category = PER_SIGNER[problem]
            entry = per_signer[(category, row["signer"], rank)]
            entry["recordings"].add(row["recording"])
            entry["tiers"].update(row["tiers"].split())
            entry["detail"] = row["detail"]

    rows.extend(per_tier.values())

    for (category, signer_id, rank), entry in per_signer.items():
        tiers = sorted(entry["tiers"])
        shown = " ".join(tiers[:3]) + (" ..." if len(tiers) > 3 else "")
        rows.append(dict(
            rank=rank, category=category,
            recording=", ".join(sorted(entry["recordings"])),
            tier=f"{len(tiers)} tier(s): {shown}", signer=signer_id,
            problem=entry["detail"], spreadsheet=spreadsheet_says(signer_id),
            in_selection="yes" if entry["recordings"] & selected else "no"))

    rows.sort(key=lambda row: (row["rank"], row["signer"], row["recording"]))
    for row in rows:
        row["action"] = ACTION[row["category"]]
        row["done"] = ""
        del row["rank"]
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Collapse the corpus audit into a list of tiers to check.")
    parser.add_argument("audit_folder", type=Path,
                        help="Folder audit_corpus.py wrote its CSVs into.")
    parser.add_argument("signers_file", type=Path,
                        help="signers.csv, the spreadsheet transcription.")
    parser.add_argument("-output_folder", "--output-folder", dest="output_folder",
                        type=Path, default=None,
                        help="Where to write. Default: the audit folder.")
    args = parser.parse_args(argv)

    for path in (args.audit_folder, args.signers_file):
        if not path.exists():
            print(f"ERROR: {path} does not exist", file=sys.stderr)
            return 1

    rows = build_rows(args.audit_folder, args.signers_file)
    if not rows:
        print("Nothing to check: the audit reported no inconsistencies.")
        return 0

    destination = (args.output_folder or args.audit_folder) / "tier_names_to_check.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} rows -> {destination}")
    for category, count in collections.Counter(r["category"] for r in rows).items():
        print(f"  {category}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
