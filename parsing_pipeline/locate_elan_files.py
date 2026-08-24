"""Locate the ELAN files a run should process.

``recursive=True`` walks the whole tree; ``recursive=False`` reads only the
files sitting directly in the folder. An optional *files of interest* text file
narrows the selection: each line may be a bare stem, a file name, a name
produced by an earlier stage or a full path, since only the stem is used and
matching ignores case and punctuation.

**One file per name.** The corpus tree holds the same recording in several
places (region folder, gesture-annotation pass, old copy, file-sync conflicted
copy) and those copies are different *versions*, not byte-identical duplicates.
Every stage names its output after the input stem, so parsing two copies of one
name means the later silently overwrites the earlier, and the result depends on
directory-walk order. The corpus is therefore expected to hold one file per
recording.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from io_utils import read_text_list

#: A file in its curated place: NN_Region/<Task>/name.eaf.
CANONICAL_PATH = re.compile(
    r"(^|/)\d{2}_[^/]+/(AniN|Cur|Pro|ReS|Int)/[^/]+\.eaf$", re.IGNORECASE)

#: Suffixes appended by earlier pipeline stages, stripped before matching.
STAGE_SUFFIXES = (
    "_word_annotations",
    "_parsed",
    "-parsed",
    ".parsed",
    "_ambiguous_rows",
)


@dataclass(frozen=True)
class DiscoveryResult:
    """The outcome of an ELAN file search.

    ``files`` holds one path per stem. ``duplicate_stems`` lists every copy of
    any name that was found more than once: which the corpus is not supposed
    to contain, and which the caller must resolve before trusting a run.
    """

    files: List[Path]
    requested_stems: List[str]
    missing_stems: List[str]
    duplicate_stems: Dict[str, List[Path]] = field(default_factory=dict)

    @property
    def n_duplicate_copies(self) -> int:
        """Copies beyond the first, over every duplicated name."""
        return sum(len(paths) - 1 for paths in self.duplicate_stems.values())

    def __len__(self) -> int:  # pragma: no cover - convenience only
        return len(self.files)


def strip_stage_suffix(stem: str) -> str:
    """Remove a known pipeline suffix from a file stem."""
    stem = str(stem or "").strip()

    changed = True
    while changed:
        changed = False
        for suffix in STAGE_SUFFIXES:
            if stem.lower().endswith(suffix.lower()) and len(stem) > len(suffix):
                stem = stem[: -len(suffix)]
                changed = True

    return stem


def normalize_stem(value: str) -> str:
    """Normalise a stem for robust comparison (drops punctuation and case)."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(value or "")).upper()


def to_stem(item: str) -> str:
    """Turn a line of a file list into a comparable file stem."""
    text = str(item or "").strip().strip('"').strip("'")
    if not text:
        return ""

    path = Path(text)
    stem = path.name

    # Remove every extension: FO_01.eaf, FO_01_word_annotations.csv, ...
    while Path(stem).suffix:
        stem = Path(stem).stem

    return strip_stage_suffix(stem)


def read_file_list(path: Path) -> List[str]:
    """Read a *files of interest* list and return normalised stems."""
    stems: List[str] = []
    seen: Set[str] = set()

    for item in read_text_list(path):
        stem = to_stem(item)
        if not stem:
            continue
        key = normalize_stem(stem)
        if key and key not in seen:
            seen.add(key)
            stems.append(stem)

    return stems


def iter_eaf_files(folder: Path, recursive: bool = True) -> List[Path]:
    """Return the .eaf files in ``folder`` (recursively or not), sorted."""
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"ELAN folder does not exist: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"ELAN folder is not a directory: {folder}")

    pattern = "**/*.eaf" if recursive else "*.eaf"
    return sorted(path for path in folder.glob(pattern) if path.is_file())


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def is_canonical(path: Path) -> bool:
    """True when the file sits in a task folder inside a numbered region folder."""
    return bool(CANONICAL_PATH.search(str(path).replace("\\", "/")))


def group_by_stem(paths: Sequence[Path]) -> Dict[str, List[Path]]:
    """Group paths by normalised stem, each group in sorted path order."""
    grouped: Dict[str, List[Path]] = {}
    for path in sorted(paths):
        grouped.setdefault(normalize_stem(path.stem), []).append(path)
    return grouped


#: Columns of the manifest the parser publishes for the other pipelines.
MANIFEST_COLUMNS = ["stem", "region_code", "path", "in_task_folder"]


def write_manifest(discovery: "DiscoveryResult", path: Path) -> Path:
    """Write the list of .eaf files this run treats as the corpus.

    The corpus-statistics and signing-space pipelines read this instead of
    walking the corpus themselves, so all three agree on which copy of each
    recording is the real one. Without it, each pipeline reimplements the
    selection rule and they drift apart silently: which is precisely the
    failure this file exists to prevent.

    Written even for ``--list-only``, so the manifest can be refreshed without
    reparsing anything.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        for eaf in discovery.files:
            writer.writerow([
                eaf.stem,
                eaf.stem.split("_", 1)[0].upper(),
                str(eaf),
                "yes" if is_canonical(eaf) else "no",
            ])
    return path


def matches_region(path: Path, regions: Sequence[str]) -> bool:
    """Return True when a file belongs to one of the requested region prefixes."""
    if not regions:
        return True

    stem = path.stem.upper()
    return any(stem.startswith(region.strip().upper() + "_") for region in regions if region.strip())


def find_elan_files(
    folder: Path,
    recursive: bool = True,
    file_list: Optional[Path] = None,
    regions: Optional[Iterable[str]] = None,
) -> DiscoveryResult:
    """Find the ELAN files to process.

    Parameters
    ----------
    folder:
        Folder that contains the ``.eaf`` files.
    recursive:
        ``True`` walks the whole tree, ``False`` reads only the top level.
    file_list:
        Optional text file listing the files of interest, one per line.
    regions:
        Optional region prefixes (``FO``, ``NS``, ...). Ignored when
        ``file_list`` is given, because an explicit list is more specific.
    """
    candidates = iter_eaf_files(folder, recursive=recursive)

    if file_list is not None:
        requested = read_file_list(file_list)
        by_stem = group_by_stem(candidates)
        selected: List[Path] = []
        missing: List[str] = []
        duplicates: Dict[str, List[Path]] = {}

        for stem in requested:
            key = normalize_stem(stem)
            matches = by_stem.get(key, [])
            if not matches:
                missing.append(stem)
                continue
            selected.append(matches[0])
            if len(matches) > 1:
                duplicates[key] = matches

        return DiscoveryResult(
            files=sorted(selected),
            requested_stems=requested,
            missing_stems=missing,
            duplicate_stems=duplicates,
        )

    region_list = [str(region) for region in (regions or [])]
    in_region = [path for path in candidates if matches_region(path, region_list)]

    # Group after the region filter, so restricting to one prefecture cannot
    # change what that prefecture's run reports.
    by_stem = group_by_stem(in_region)
    selected = [paths[0] for paths in by_stem.values()]
    duplicates = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}

    return DiscoveryResult(
        files=sorted(selected),
        requested_stems=[path.stem for path in sorted(selected)],
        missing_stems=[],
        duplicate_stems=duplicates,
    )
