"""CSV helpers and small shared utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import re

import pandas as pd

from config import CSV_READ_ENCODINGS, CSV_WRITE_ENCODING, REGION_NAMES


def read_csv_safely(path: Path, dtype=str) -> pd.DataFrame:
    """Read a CSV, trying the Japanese-friendly encodings in turn."""
    path = Path(path)
    last_error: Optional[Exception] = None
    for encoding in CSV_READ_ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding, dtype=dtype,
                                keep_default_na=False)
            frame.columns = (frame.columns.astype(str)
                             .str.replace("﻿", "", regex=False).str.strip())
            return frame
        except UnicodeDecodeError as error:
            last_error = error
        except FileNotFoundError:
            raise
    raise RuntimeError(f"Could not read CSV {path}: {last_error}")


def write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding=CSV_WRITE_ENCODING)
    return path


def region_of(name: str) -> str:
    """Prefecture code from a filename: FO_01-02_AniN_parsed.csv -> FO."""
    stem = Path(str(name)).stem
    code = stem.split("_", 1)[0].upper()
    return code if code in REGION_NAMES else code


def region_label(code: str) -> str:
    """Human name for a prefecture code, falling back to the code itself."""
    return REGION_NAMES.get(str(code).upper(), str(code))


#: Suffixes the parser appends to a document name. They are stripped before
#: matching a parsed CSV to its .eaf, so FO_01-02_AniN_parsed.csv and
#: FO_01-02_AniN.eaf are recognised as the same recording.
PARSED_SUFFIXES = ["parsed", "parse", "annotations", "annotation", "output", "out"]


def file_key(name: object) -> str:
    """A comparable identity for one recording, from a CSV or .eaf filename.

    Lower-cased, alphanumerics only, with any parser suffix removed. Matching on
    this rather than on the raw name is what lets the ELAN index and the parsed
    annotations be produced by two unrelated tools.
    """
    stem = Path(str(name or "")).stem
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", stem) if p]
    while parts and parts[-1].lower() in PARSED_SUFFIXES:
        parts.pop()
    return "".join(parts).lower()


def nonempty(series: pd.Series) -> pd.Series:
    """True where a string column holds something other than blank/NaN text."""
    text = series.fillna("").astype(str).str.strip()
    return text.ne("") & ~text.str.lower().isin({"nan", "none", "null", "<na>"})


def format_hms(milliseconds: float) -> str:
    seconds = int(round(float(milliseconds) / 1000.0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def find_files(root: Path, suffix: str) -> List[Path]:
    """Every file with this suffix under root, searched recursively."""
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    suffix = suffix.lower()
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() == suffix)
