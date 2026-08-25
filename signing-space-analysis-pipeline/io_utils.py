"""CSV, text and argument helpers shared by every stage."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import CSV_READ_ENCODINGS, CSV_WRITE_ENCODING


def count_or_all(text: str) -> int:
    """An argparse type for "how many, or all of them".

    Accepts a non-negative integer, or the word ``all`` (and ``every`` / ``0``),
    which is returned as ``0`` - the internal sentinel for "no cap". Spelling it
    out matters here because ``--debug-limit 0`` reads like "none" and means the
    opposite.
    """
    value = str(text).strip().lower()
    if value in {"all", "every", "everything"}:
        return 0
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number of clips or the word 'all', not {text!r}"
        ) from None
    if number < 0:
        raise argparse.ArgumentTypeError("cannot be negative; use 'all' for no cap")
    return number


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read a CSV as strings, trying the Japanese-friendly encodings in turn."""
    path = Path(path)
    last_error: Optional[Exception] = None

    for encoding in CSV_READ_ENCODINGS:
        try:
            return pd.read_csv(
                path, encoding=encoding, dtype=str, keep_default_na=False
            )
        except UnicodeDecodeError as error:
            last_error = error
        except FileNotFoundError:
            raise
        except Exception as error:  # pragma: no cover - defensive
            last_error = error

    raise RuntimeError(f"Could not read CSV {path}: {last_error}")


def write_csv(path: Path, frame: pd.DataFrame) -> Path:
    """Write a DataFrame, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding=CSV_WRITE_ENCODING)
    return path


def read_text_list(path: Optional[Path]) -> List[str]:
    """Read a newline-separated list; blank lines and ``#`` comments ignored."""
    if path is None:
        return []

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"List file does not exist: {path}")

    items: List[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            value = line.split("#", 1)[0].strip()
            if value:
                items.append(value)

    return items
