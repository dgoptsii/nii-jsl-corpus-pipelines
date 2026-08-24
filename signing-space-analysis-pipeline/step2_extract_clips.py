#!/usr/bin/env python3
"""STEP 2: cut a cropped video clip for every selected annotation.

    python3 step2_extract_clips.py OUTPUT_FOLDER --video-root /path/to/videos

Source recordings show two signers in a 2x2 layout; only the upper panel of the
relevant half is kept, so each clip holds one signer. The clip index written
here carries signer identity, handedness and age group into every later stage.
Requires ffmpeg and ffprobe on PATH.

Output: ``OUT/clips/<KEYWORD>/<REGION>/<SOURCE_FILE>/<SIDE>/<KEYWORD>_<ROW>.mp4``
and ``OUT/clips/clip_index.csv``
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from config import CLIPS_SUBFOLDER, CLIP_INDEX_FILE, KEY_ROWS_SUBFOLDER, VIDEO_EXTENSIONS
from io_utils import read_csv_safely, write_csv
from signers import load_signer_metadata


#: Columns of clip_index.csv, in order. Declared so the file always has a
#: header, even when nothing was cut.
CLIP_INDEX_COLUMNS = [
    "keyword", "clip_id", "clip_path", "region_code", "source_file",
    "signer_id", "handedness", "age", "age_group", "gender", "side",
    "row_index", "time_start", "time_end", "annotation", "lexical_item",
    "video_path",
]


def to_seconds(value: str) -> float:
    number = float(str(value).strip())
    return number / 1000.0 if number > 1000 else number   # ELAN uses milliseconds


def video_stems(source_file: str) -> List[str]:
    """Filename variants a source_file may appear under."""
    source_file = str(source_file).strip()
    stems = {source_file, source_file.replace("-", "_")}

    match = re.fullmatch(r"([A-Za-z]{2})_(\d{1,2})[-_](\d{1,2})_([A-Za-z0-9]+)", source_file)
    if match:
        region, first, second, task = match.groups()
        stems.add(f"{region}_{int(first):02d}-{int(second):02d}_{task}")
        stems.add(f"{region}_{int(first):02d}_{int(second):02d}_{task}")

    return sorted(stems)


def find_video(source_file: str, roots: Sequence[Path]) -> Optional[Path]:
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        for stem in video_stems(source_file):
            for extension in VIDEO_EXTENSIONS:
                matches = sorted(root.rglob(stem + extension))
                if matches:
                    return matches[0]
    return None


def video_size(video_path: Path) -> tuple:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(video_path)],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    width, height = result.stdout.strip().split("x")
    return int(width), int(height)


def infer_side(signer_id: str, forced: Optional[str], overrides: Dict[str, str]) -> str:
    """Which half of the frame this signer occupies."""
    if forced in {"left", "right"}:
        return forced

    for key, side in overrides.items():
        if key and key in signer_id:
            return side

    match = re.search(r"(\d+)", str(signer_id))
    if match:
        return "right" if int(match.group(1)) % 2 == 1 else "left"
    return "left"


def crop_filter(video_path: Path, side: str, panel_fraction: float,
                margin: int, y_offset: int) -> str:
    width, height = video_size(video_path)
    half_width = width // 2
    panel_height = max(2, min(int(round(height * panel_fraction)), height))

    crop_width = max(2, half_width - 2 * margin)
    x = margin if side == "left" else half_width + margin
    y = max(0, min(margin + y_offset, panel_height - 2))
    crop_height = max(2, panel_height - y - margin)

    return f"crop={crop_width}:{crop_height}:{x}:{y}"


def cut_clip(video_path: Path, start: float, end: float, side: str, out_path: Path,
             panel_fraction: float, margin: int, y_offset: int) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video_path),
         "-t", f"{max(0.001, end - start):.3f}",
         "-vf", crop_filter(video_path, side, panel_fraction, margin, y_offset),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an",
         "-movflags", "+faststart", str(out_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="step2_extract_clips.py",
        description="Cut one cropped clip per selected annotation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python3 step2_extract_clips.py ./out --video-root ~/corpus\n",
    )
    parser.add_argument("output_folder", type=Path,
                        help="The folder step 1 wrote to; clips go under it.")
    parser.add_argument("--video-root", "--video_root", dest="video_root",
                        type=Path, action="append", required=True,
                        help="Folder holding the source videos. Repeatable.")
    parser.add_argument("--keywords", nargs="*", default=[],
                        help="Only these keywords. Empty = every keyword found.")
    parser.add_argument("--regions", nargs="*", default=[],
                        help="Only these geographical regions. Empty = all.")
    parser.add_argument("-signers_file", "--signers_file", "--signers-file",
                        dest="signers_file", type=Path, default=None,
                        metavar="SIGNERS.csv",
                        help="CSV of signer_id,handedness,age. Left-handed signers "
                             "are mirrored; ages drive the age-group table. "
                             "Anyone absent is right-handed with an unknown age.")
    parser.add_argument("--padding-seconds", type=float, default=0.0,
                        help="Seconds added before and after each annotation.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after this many clips per keyword (0 = no limit).")
    parser.add_argument("--force-crop-side", choices=["left", "right"], default=None)
    parser.add_argument("--side-override", nargs="*", default=[],
                        help="Manual overrides, e.g. FO_01=right FO_02=left.")
    parser.add_argument("--upper-panel-height-fraction", type=float, default=0.465)
    parser.add_argument("--crop-margin-px", type=int, default=4)
    parser.add_argument("--crop-y-offset-px", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    output_folder = args.output_folder.expanduser().resolve()
    key_rows_root = output_folder / KEY_ROWS_SUBFOLDER
    clips_root = output_folder / CLIPS_SUBFOLDER

    if not key_rows_root.is_dir():
        print(f"ERROR: no key_rows folder in {output_folder}. Run step 1 first.", file=sys.stderr)
        return 1

    metadata = load_signer_metadata(args.signers_file)
    overrides = dict(
        item.split("=", 1) for item in args.side_override if "=" in item
    )
    overrides = {k.strip(): v.strip().lower() for k, v in overrides.items()}

    wanted_keywords = {k.upper() for k in args.keywords} if args.keywords else None
    wanted_regions = {r.upper() for r in args.regions} if args.regions else None

    csv_paths = sorted(key_rows_root.glob("*/ALL_*_rows.csv"))
    if wanted_keywords:
        csv_paths = [p for p in csv_paths if p.parent.name.upper() in wanted_keywords]

    if not csv_paths:
        print(f"No ALL_*_rows.csv files found under {key_rows_root}")
        return 1

    print(f"Key rows:      {key_rows_root}")
    print(f"Clips:         {clips_root}")
    print(f"Video roots:   {', '.join(str(r) for r in args.video_root)}")
    print(metadata.describe())
    print()

    index_rows: List[Dict[str, object]] = []
    video_cache: Dict[str, Optional[Path]] = {}
    failures = 0

    for csv_path in csv_paths:
        keyword = csv_path.parent.name.upper()
        frame = read_csv_safely(csv_path)
        if frame.empty:
            print(f"{keyword}: no rows")
            continue

        if wanted_regions and "region_code" in frame.columns:
            frame = frame[frame["region_code"].str.upper().isin(wanted_regions)]

        made = 0
        for row_index, row in frame.iterrows():
            if args.limit and made >= args.limit:
                break

            source_file = str(row.get("source_file", "")).strip()
            if source_file not in video_cache:
                video_cache[source_file] = find_video(source_file, args.video_root)
                found = video_cache[source_file]
                print(f"  {source_file} -> {found if found else 'NOT FOUND'}")

            video_path = video_cache[source_file]
            if video_path is None:
                failures += 1
                continue

            try:
                start = max(0.0, to_seconds(row["time_start"]) - args.padding_seconds)
                end = to_seconds(row["time_end"]) + args.padding_seconds
            except Exception:
                failures += 1
                continue
            if end <= start:
                failures += 1
                continue

            signer_id = str(row.get("signer_id", "")).strip()
            side = infer_side(signer_id, args.force_crop_side, overrides)
            region_code = str(row.get("region_code", "")).strip().upper()
            clip_name = f"{keyword}_{row_index:06d}.mp4"
            clip_path = clips_root / keyword / region_code / source_file / side / clip_name

            if not cut_clip(video_path, start, end, side, clip_path,
                            args.upper_panel_height_fraction,
                            args.crop_margin_px, args.crop_y_offset_px):
                failures += 1
                continue

            index_rows.append({
                "keyword": keyword,
                "clip_id": str(clip_path.relative_to(clips_root).with_suffix("")),
                "clip_path": str(clip_path),
                "region_code": region_code,
                "source_file": source_file,
                "signer_id": signer_id,
                "handedness": metadata.handedness(signer_id),
                "age": metadata.age(signer_id) if metadata.age(signer_id) is not None else "",
                "age_group": metadata.age_group(signer_id),
                "gender": metadata.gender(signer_id),
                "side": side,
                "row_index": row_index,
                "time_start": row.get("time_start", ""),
                "time_end": row.get("time_end", ""),
                "annotation": row.get("annotation", ""),
                "lexical_item": row.get("lexical_item", ""),
                "video_path": str(video_path),
            })
            made += 1

        print(f"{keyword:14s} {made:5d} clips")

    # Explicit columns so a run that cut nothing still writes a readable header.
    # Without them the file is zero bytes and every later stage dies on
    # "No columns to parse" instead of reporting an empty selection.
    index = pd.DataFrame(index_rows, columns=CLIP_INDEX_COLUMNS)
    index_path = write_csv(clips_root / CLIP_INDEX_FILE, index)

    print(f"\nDone. {len(index)} clips, {failures} skipped.")
    if len(index):
        print(f"Signers: {index['signer_id'].nunique()}  "
              f"left-handed: {(index['handedness'] == 'left').sum()} clips")

        # A signers file whose IDs do not match the corpus fails silently -
        # everyone just comes out right-handed with an unknown age. Say so.
        if args.signers_file is not None:
            signers = sorted({str(s) for s in index["signer_id"] if str(s).strip()})
            unmatched = [s for s in signers if metadata.age(s) is None]
            matched = len(signers) - len(unmatched)
            print(f"Matched to the signers file: {matched}/{len(signers)}")
            if unmatched:
                print(f"  WARNING: no row for {len(unmatched)} signers, so they "
                      f"default to right-handed with an unknown age.")
                print(f"  Check the ID prefixes in {args.signers_file}: "
                      + ", ".join(unmatched[:8])
                      + (" ..." if len(unmatched) > 8 else ""))
    print(f"Index: {index_path}")
    print(f"Next:  python3 step3_extract_landmarks.py {args.output_folder}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
