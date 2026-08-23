"""Small IO helpers shared by every pipeline stage."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from config import CSV_READ_ENCODINGS, CSV_WRITE_ENCODING


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV as a list of dicts, trying several Japanese-friendly encodings."""
    last_error: Optional[Exception] = None

    for encoding in CSV_READ_ENCODINGS:
        try:
            with Path(path).open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as error:
            last_error = error
        except FileNotFoundError:
            raise
        except Exception as error:  # pragma: no cover - defensive
            last_error = error

    raise RuntimeError(f"Could not read CSV {path}: {last_error}")


def write_csv_rows(
    path: Path,
    rows: Iterable[Dict[str, str]],
    fieldnames: Sequence[str],
) -> Path:
    """Write ``rows`` to ``path``, keeping only ``fieldnames`` and in that order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding=CSV_WRITE_ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return path


def read_text_list(path: Path) -> List[str]:
    """Read a newline-separated list file.

    Blank lines and lines starting with ``#`` are ignored, so the file can be
    commented. Used for the *files of interest* list and for ``exceptions.txt``.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"List file does not exist: {path}")

    items: List[str] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                items.append(value)

    return items
