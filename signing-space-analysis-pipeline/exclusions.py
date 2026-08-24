"""An optional list of clips to drop after visual inspection.

Some crops are unusable (the signer steps out of frame, the panel is wrong) and
no automatic check catches them. This is the manual override: a text file of
clip names, applied from landmark extraction onward.

    # input_lists/excluded_clips.txt
    FS_000123                    # one clip
    FS/NS/NS_07-08_AniN          # whole recording

Matching is forgiving, since names get copied out of a file browser: bare name,
with or without ``.mp4``, full ``clip_id``, absolute path, or any leading folder
of a ``clip_id``. Case and slash direction are ignored. The one thing not
forgiven is an entry matching nothing: a typo here is otherwise invisible, so
every unused entry is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from config import VIDEO_EXTENSIONS


def normalise_clip_key(value: str) -> str:
    """Fold a clip name, path or id onto one comparable form."""
    text = str(value or "").strip().strip('"').strip("'").replace("\\", "/")
    lowered = text.lower()
    for extension in VIDEO_EXTENSIONS:
        if lowered.endswith(extension):
            text = text[: -len(extension)]
            break
    return text.strip("/").casefold()


@dataclass
class ClipExclusions:
    """Clip names to skip, and a record of which ones actually matched."""

    entries: List[str] = field(default_factory=list)       # normalised
    raw: List[str] = field(default_factory=list)           # as written
    sources: List[Path] = field(default_factory=list)
    used: Set[str] = field(default_factory=set)

    @property
    def source(self) -> str:
        """The file(s) the entries came from, for messages."""
        return ", ".join(str(p) for p in self.sources) or "(none)"

    def __bool__(self) -> bool:
        return bool(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def excludes(self, clip_id: str, clip_path: str = "") -> bool:
        """True when this clip is named by the list, directly or by folder."""
        if not self.entries:
            return False

        candidates = [normalise_clip_key(clip_id), normalise_clip_key(clip_path)]
        candidates = [candidate for candidate in candidates if candidate]

        for entry in self.entries:
            for candidate in candidates:
                if (
                    candidate == entry
                    or candidate.endswith("/" + entry)          # a tail: name, or side/name
                    or candidate.startswith(entry + "/")        # a leading folder
                    or ("/" + entry + "/") in candidate         # a folder anywhere
                ):
                    self.used.add(entry)
                    return True
        return False

    @property
    def unused(self) -> List[str]:
        """Entries that matched no clip - almost always a typo."""
        return [raw for raw, entry in zip(self.raw, self.entries)
                if entry not in self.used]

    def describe(self) -> str:
        if not self.entries:
            return "Excluded clips:   (none)"
        if len(self.sources) > 1:
            files = "\n                  ".join(str(p) for p in self.sources)
            return (f"Excluded clips:   {len(self.entries)} entries from "
                    f"{len(self.sources)} files\n                  {files}")
        return (f"Excluded clips:   {len(self.entries)} entries from {self.source}")

    def report_unused(self) -> None:
        """Say so when an entry matched nothing; a silent typo keeps bad data in."""
        missing = self.unused
        if not missing:
            return
        print(f"  WARNING: {len(missing)} exclusion entr"
              f"{'y' if len(missing) == 1 else 'ies'} matched no clip, so "
              f"nothing was dropped for them:")
        for raw in missing[:10]:
            print(f"    {raw}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")


def load_exclusions(paths) -> ClipExclusions:
    """Read one list, or several, into a single set of exclusions.

    Several because the reasons for dropping a clip are independent and arrive
    at different times: a batch rejected during one inspection pass, another
    during the next. Keeping them in separate files means each can be
    regenerated or cleared on its own; merging them by hand would make one of
    them go stale. Clips cut from the wrong source file are not handled here:
    those are deleted and re-extracted, because an exclusion list would leave
    the bad landmarks on disk for the next run to pick up.

    No file means an empty list, not an error.
    """
    if paths is None:
        return ClipExclusions()
    if isinstance(paths, (str, Path)):
        paths = [paths]

    raw: List[str] = []
    entries: List[str] = []
    seen: Set[str] = set()
    sources: List[Path] = []

    for item in paths:
        if item is None:
            continue
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"Exclusion file does not exist: {path}")
        sources.append(path)
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                text = line.split("#", 1)[0].strip()
                if not text:
                    continue
                key = normalise_clip_key(text)
                if key and key not in seen:
                    seen.add(key)
                    raw.append(text)
                    entries.append(key)

    return ClipExclusions(entries=entries, raw=raw, sources=sources)


def filter_index(index, exclusions: ClipExclusions, label: str = "") -> tuple:
    """Drop excluded rows from a clip index. Returns ``(index, n_dropped)``."""
    if not exclusions or index.empty:
        return index, 0

    keep = [
        not exclusions.excludes(str(row.get("clip_id", "")),
                                str(row.get("clip_path", "")))
        for _, row in index.iterrows()
    ]
    dropped = len(keep) - sum(keep)
    if dropped and label:
        print(f"{label}{dropped} clips excluded by {exclusions.source}")
    return index[keep], dropped


def apply_to_column(frame, exclusions: ClipExclusions,
                    clip_id_column: str = "clip_id") -> tuple:
    """Same, for a table that carries clip ids but no paths."""
    if not exclusions or frame.empty:
        return frame, 0
    keep = ~frame[clip_id_column].astype(str).map(exclusions.excludes)
    return frame[keep], int((~keep).sum())


__all__ = [
    "ClipExclusions",
    "apply_to_column",
    "filter_index",
    "load_exclusions",
    "normalise_clip_key",
]
