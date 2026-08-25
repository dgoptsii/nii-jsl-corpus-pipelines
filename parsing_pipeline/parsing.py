"""Rule-based parser for JSL Word-tier annotations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from config import (
    ATTR_COLUMNS,
    FIELDNAMES,
    HAND_COLUMNS,
    METADATA_COLUMNS,
)
from io_utils import read_text_list

FULLWIDTH_DIGIT_TRANS = str.maketrans("０１２３４５６７８９", "0123456789")

#: Keywords that make an annotation unparseable on sight. Empty by default;
#: configure with :func:`configure_keywords`.
UNKNOWN_KEYWORDS: List[str] = []

#: Keywords that are known but deliberately routed to manual review. Empty by
#: default; configure with :func:`configure_keywords`.
KNOWN_UNKNOWN_KEYWORDS: List[str] = []

#: Normalised annotations that must never be flagged ambiguous.
_NOT_AMBIGUOUS_EXCEPTIONS: Set[str] = set()

KEYWORD_SUBSTITUTIONS = {
    "ｆａｌ": "FAL",
    "ＦＡＬ": "FAL",
    "口形": "M",
    "口型": "M",
    "ｍ": "M",
    "Ｍ": "M",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def configure_keywords(
    unknown_keywords: Optional[Sequence[str]] = None,
    known_unknown_keywords: Optional[Sequence[str]] = None,
) -> None:
    """Replace the keyword lists that force manual review."""
    global UNKNOWN_KEYWORDS, KNOWN_UNKNOWN_KEYWORDS

    if unknown_keywords is not None:
        UNKNOWN_KEYWORDS = [str(keyword) for keyword in unknown_keywords]

    if known_unknown_keywords is not None:
        KNOWN_UNKNOWN_KEYWORDS = [str(keyword) for keyword in known_unknown_keywords]


def set_exceptions(annotations: Iterable[str]) -> Set[str]:
    """Set the manual "not ambiguous" exception list from raw annotation strings."""
    global _NOT_AMBIGUOUS_EXCEPTIONS
    _NOT_AMBIGUOUS_EXCEPTIONS = {
        normalize_exception_key(annotation) for annotation in annotations
    }
    return _NOT_AMBIGUOUS_EXCEPTIONS


def load_exceptions(path: Optional[Path]) -> Set[str]:
    """Load ``exceptions.txt``.

    One annotation per line; blank lines and ``#`` comments are ignored. A
    missing path clears the exception list, which keeps parser behaviour
    explicit and prevents hidden hard-coded exceptions from affecting results.
    """
    if path is None or not Path(path).exists():
        return set_exceptions([])

    return set_exceptions(read_text_list(Path(path)))


def normalize_exception_key(text: str) -> str:
    return normalize_annotation(text).lstrip("/").strip().lower()


def is_not_ambiguous_exception(annotation: str) -> bool:
    return normalize_exception_key(annotation) in _NOT_AMBIGUOUS_EXCEPTIONS


# ---------------------------------------------------------------------------
# Row scaffolding
# ---------------------------------------------------------------------------

def empty_attrs() -> Dict[str, str]:
    return {key: "" for key in ATTR_COLUMNS}


def make_empty_output_row(annotation: str, ambiguous: str = "") -> Dict[str, str]:
    row = {
        "speaker_id": "",
        "tier_id": "",
        "time_start": "",
        "time_end": "",
        "annotation": annotation,
        "lexical_item": "",
        "compound": "",
        "ambiguous": ambiguous,
    }
    row.update(empty_attrs())
    return row


def get_first_existing_value(row: Dict[str, str], possible_names: List[str]) -> str:
    """Return a value from an input row even if files use different column names."""
    if not row:
        return ""

    exact_lookup = {str(k): v for k, v in row.items()}
    lower_lookup = {str(k).strip().lower(): v for k, v in row.items()}

    for name in possible_names:
        if name in exact_lookup and exact_lookup[name] not in (None, ""):
            return str(exact_lookup[name]).strip()

        lower_name = name.strip().lower()
        if lower_name in lower_lookup and lower_lookup[lower_name] not in (None, ""):
            return str(lower_lookup[lower_name]).strip()

    return ""


def extract_metadata(input_row: Dict[str, str]) -> Dict[str, str]:
    """Keep the information needed to rebuild ELAN annotations later.

    The extractor writes the signer to ``file_id`` (``FO_03``) and the tier it
    read to ``tier_id`` (``FO_03_NG_40F-Word-jp``). Older intermediate files may
    use ``speaker_id``, ``participant_id`` and friends, and have no tier column
    at all; those still parse, and ``tier_id`` is simply left empty, which the
    rebuild treats as "find the tier by name" exactly as it used to.

    Downstream steps join parsed rows back onto the original ELAN annotations
    using these values together with the time interval, so ``speaker_id`` must
    not be left empty when ``file_id`` is present.
    """
    speaker_id = get_first_existing_value(input_row, [
        "speaker_id", "file_id", "speaker", "participant_id", "participant",
        "signer_id", "signer", "person_id", "tier_id", "tier", "tier_name",
    ])

    tier_id = get_first_existing_value(input_row, ["tier_id", "tier", "tier_name"])

    time_start = get_first_existing_value(input_row, [
        "time_start", "start_time", "start", "start_ms", "start_time_ms",
        "begin", "begin_time", "begin_ms", "onset", "onset_ms",
    ])

    time_end = get_first_existing_value(input_row, [
        "time_end", "end_time", "time_finish", "finish_time", "finish", "end",
        "end_ms", "end_time_ms", "offset", "offset_ms", "stop_time",
    ])

    return {
        "speaker_id": speaker_id,
        "tier_id": tier_id,
        "time_start": time_start,
        "time_end": time_end,
    }


def attach_metadata(output_row: Dict[str, str], metadata: Dict[str, str]) -> Dict[str, str]:
    for column in METADATA_COLUMNS:
        output_row[column] = metadata.get(column, "")
    return output_row


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_digits(text: str) -> str:
    return str(text or "").translate(FULLWIDTH_DIGIT_TRANS)


def add_unique(attrs: Dict[str, str], key: str, value: str) -> None:
    value = str(value or "").strip()
    if not value or key not in attrs:
        return

    old = attrs.get(key, "")
    if not old:
        attrs[key] = value
        return

    values = old.split(";")
    if value not in values:
        attrs[key] = old + ";" + value


def add_lexical_component(row: Dict[str, str], value: str) -> None:
    value = cleanup_lexical_item(value)
    if not value:
        return

    old = row.get("lexical_item", "")
    if not old:
        row["lexical_item"] = value
        return

    parts = old.split(";")
    if value not in parts:
        row["lexical_item"] = old + ";" + value


def merge_attrs(target: Dict[str, str], source: Dict[str, str]) -> None:
    for key in ATTR_COLUMNS:
        for value in str(source.get(key, "") or "").split(";"):
            add_unique(target, key, value)


def merge_rows(target: Dict[str, str], source: Dict[str, str]) -> None:
    if source.get("lexical_item"):
        for part in str(source["lexical_item"]).split(";"):
            add_lexical_component(target, part)

    merge_attrs(target, source)

    if source.get("compound"):
        target["compound"] = source["compound"]
        target["lexical_item"] = ""

    if source.get("ambiguous"):
        target["ambiguous"] = source["ambiguous"]
        target["lexical_item"] = ""

    if source.get("_skip_ambiguous_file"):
        target["_skip_ambiguous_file"] = True


def remove_first_tilde(text: str) -> str:
    return re.sub(r"[˜~〜]", "", str(text or ""), count=1)


def remove_exclamation_marks(text: str) -> str:
    return re.sub(r"[!！]+", "", str(text or ""))


def normalize_keywords(text: str) -> str:
    text = str(text or "")
    for src, dst in KEYWORD_SUBSTITUTIONS.items():
        text = text.replace(src, dst)

    # DR and DW are treated as the same annotation keyword.
    # This also normalizes compact forms such as ptdr and cl:dr.
    text = re.sub(r"(?i)dr", "dw", text)

    # REPEAT is treated as the same annotation marker as REP.
    # Example: 広がるの抽象的表現(repeat) -> rep=広がるの抽象的表現(1;N)
    text = re.sub(r"(?i)\bREPEAT\b", "REP", text)
    return text


def strip_leading_annotation_slashes(text: str) -> str:
    """Strip one or more annotation-prefix slashes only at the beginning.

    Examples:
      /猫   -> 猫
      //LH(...) +RH(...) -> LH(...) +RH(...)
      A/B   -> A/B  # internal slash is not touched here
    """
    return re.sub(r"^\s*/+", "", str(text or "")).strip()


def normalize_slash_separators(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("／", "+")
    leading_slash = text.startswith("/")
    prefix = "/" if leading_slash else ""
    body = text[1:] if leading_slash else text
    body = re.sub(r"/+$", "", body)
    body = re.sub(r"\s*/\s*", "+", body)
    body = re.sub(r"^\s*\++\s*", "", body)
    body = re.sub(r"\s*\++\s*$", "", body)
    return prefix + body.strip()


def normalize_annotation(text: str) -> str:
    text = remove_first_tilde(text)
    text = remove_exclamation_marks(text)
    text = normalize_keywords(text)
    text = strip_leading_annotation_slashes(text)
    text = text.replace("&", "+").replace("＋", "+")
    text = normalize_slash_separators(text)
    return text.strip()


def split_outside_brackets(text: str, separator: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0

    for ch in str(text or ""):
        if ch in "（({｛":
            depth += 1
            current.append(ch)
        elif ch in "）)}｝":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == separator and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(ch)

    part = "".join(current).strip()
    if part:
        parts.append(part)

    return parts


def split_plus_streams(text: str) -> List[str]:
    text = str(text or "").strip()
    parts = split_outside_brackets(text, "+")

    final_parts: List[str] = []
    for part in parts:
        subparts = re.split(
            r"\s+(?=(?i:RH|R|LH|L)\s*[:：]|右手\s*[:：]|左手\s*[:：])",
            part,
        )
        final_parts.extend([p.strip() for p in subparts if p.strip()])

    return final_parts


def split_lexical_components(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []

    # Comma-like separators become semicolon-separated lexical_item components.
    parts = re.split(r"[、,，]\s*", text)
    return [cleanup_lexical_item(p) for p in parts if cleanup_lexical_item(p)]


def strip_compound_marks(text: str) -> str:
    return str(text or "").replace("<", "").replace(">", "").replace("＜", "").replace("＞", "")


def has_compound_marker(text: str) -> bool:
    return any(ch in str(text or "") for ch in ["<", ">", "＜", "＞"])


def has_compound_open(text: str) -> bool:
    return "<" in str(text or "") or "＜" in str(text or "")


def has_compound_close(text: str) -> bool:
    return ">" in str(text or "") or "＞" in str(text or "")


def has_more_than_one_slash(annotation: str) -> bool:
    return str(annotation or "").count("/") > 1


def is_simple_slash_wrapped_annotation(annotation: str) -> bool:
    text = str(annotation or "").strip()
    if not (text.startswith("/") and text.endswith("/")):
        return False

    inner = text[1:-1].strip()
    if not inner:
        return False

    return "/" not in inner


def contains_unknown_keyword(text: str) -> bool:
    text = str(text or "")
    for keyword in UNKNOWN_KEYWORDS:
        keyword = str(keyword).strip()
        if keyword and re.search(re.escape(keyword), text, flags=re.I):
            return True
    return False


def annotation_has_known_unknown(text: str) -> bool:
    text = normalize_annotation(text)
    for keyword in KNOWN_UNKNOWN_KEYWORDS:
        if re.search(rf"(?i)(^|[^A-Za-z]){re.escape(keyword)}([^A-Za-z]|$)", text):
            return True
        if re.search(rf"(?i)^/?{re.escape(keyword)}", text):
            return True
    return False


def is_allowed_latin_lexical_item(text: str) -> bool:
    return bool(re.fullmatch(r"(?i)ok|pt[0-9]*", str(text or "").strip()))


def has_non_japanese_residue(text: str) -> bool:
    if not text:
        return False

    parts = [p.strip() for p in str(text).split(";") if p.strip()]
    if parts and all(is_allowed_latin_lexical_item(p) for p in parts):
        return False

    text_for_check = re.sub(r"(?i)(^|;)\s*pt[0-9]*\s*(?=;|$)", ";", str(text))
    text_for_check = re.sub(r"\[[A-Za-z]+\]", "[]", text_for_check)

    return bool(
        re.search(
            r"[^぀-ゟ"
            r"゠-ヿ"
            r"一-鿿"
            r"ー々〆〤"
            r"0-9０-９"
            r"、。，．〒"
            r"「」”“\""
            r":："
            r"\s"
            r"＋+\-−"
            r"\(\)（）"
            r"\[\]"
            r";"
            r"]",
            text_for_check,
        )
    )


def cleanup_lexical_item(text: str) -> str:
    text = str(text or "").strip()
    text = strip_compound_marks(text)

    # Remove Japanese corner quotation marks only; keep ” and " because they can be part of notes.
    text = re.sub(r"[「」]", "", text)

    text = re.sub(r"^/+", "", text)
    text = text.strip("【】")

    # Remove empty parentheses only.
    text = re.sub(r"[\(（]\s*[\)）]", "", text)

    text = re.sub(r"^[\s:,;＝=]+", "", text)
    text = re.sub(r"[\s:,;＝=]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Fix only genuinely unmatched edge brackets, but keep meaningful balanced parentheses.
    if text.startswith((")", "）")) and not re.search(r"[\(（]", text):
        text = re.sub(r"^[\s\)）]+", "", text).strip()
    if text.startswith(("(", "（")) and not re.search(r"[\)）]", text):
        text = re.sub(r"^[\s\(（]+", "", text).strip()

    open_count = len(re.findall(r"[\(（]", text))
    close_count = len(re.findall(r"[\)）]", text))
    if open_count > close_count:
        text += ")" * (open_count - close_count)
    elif close_count > open_count:
        # Surplus closers, which are left behind when a marker value is lifted
        # out of a wrapper: (ges:X(Y)) leaves "(Y))". Drop only the surplus, and
        # only from the end, so a balanced pair inside the text is untouched.
        surplus = close_count - open_count
        while surplus and re.search(r"[\)）]\s*$", text):
            text = re.sub(r"[\)）]\s*$", "", text).strip()
            surplus -= 1
        if surplus and not re.search(r"[\(（]", text):
            text = re.sub(r"[\s\)）]+$", "", text).strip()

    return text.strip()


# ---------------------------------------------------------------------------
# Hand detection
# ---------------------------------------------------------------------------

MARKER_TO_HAND = {
    "L": "lh", "LH": "lh", "左手": "lh", "左": "lh",
    "R": "rh", "RH": "rh", "右手": "rh", "右": "rh",
}


def detect_hand_from_japanese_explanation(original_text: str) -> Optional[str]:
    text = str(original_text or "")
    if "右手" in text or "右" in text:
        return "rh"
    if "左手" in text or "左" in text:
        return "lh"
    return None


def parse_parenthetical_hand(text: str) -> Tuple[str, Optional[str]]:
    match = re.search(r"[\(（]\s*(RH|R|LH|L|右手|左手)\s*[\)）]", text, flags=re.I)
    if not match:
        return text, None

    marker = match.group(1)

    if marker.upper() in {"RH", "R"} or marker == "右手":
        hand = "rh"
    else:
        hand = "lh"

    text = re.sub(
        r"[\(（]\s*(RH|R|LH|L|右手|左手)\s*[\)）]",
        "",
        text,
        flags=re.I,
    )

    return text.strip(), hand


# ---------------------------------------------------------------------------
# Marker parsers
# ---------------------------------------------------------------------------

def remove_pt_objects(text: str) -> str:
    def repl(match):
        inner = match.group(2).strip()
        if re.fullmatch(r"(?i)REP\s*[:：]?\s*[0-9０-９]*", inner):
            return match.group(0)
        if re.match(r"(?i)DW\s*[:：]", inner):
            return match.group(0)
        # Keep ordinary PT(object) for parse_pt(), so the object is not lost.
        return match.group(0)

    return re.sub(
        r"(?i)\b(PT[0-9]*)\s*[\(（]([^\)）]*)[\)）]",
        repl,
        text,
    )


def parse_prefix_chain(segment: str) -> Tuple[str, bool]:
    text = str(segment or "").strip()
    active_fs = False

    while True:
        match = re.match(r"(?i)^FS\s*[:：]\s*", text)
        if match:
            active_fs = True
            text = text[match.end():].strip()
            continue
        break

    return text, active_fs


def parse_nmm(text: str, attrs: Dict[str, str]) -> str:
    found = False

    pattern = r"(?i)(^|[^A-Za-z])NMM\s*[:：]\s*([^()\[\]+/;,]*?)(?=(?i:rep|stop|hold|keep|index)|$|[＋+,;/\[\]（）()])"
    for match in re.finditer(pattern, text):
        found = True
        value = cleanup_lexical_item(match.group(2).strip())
        if value:
            add_unique(attrs, "nmm", value)
            if value.lower() == "neg":
                add_unique(attrs, "neg", "TRUE")

    text = re.sub(pattern, " ", text)

    for match in re.finditer(r"(?i)(^|[^A-Za-z])NMM\s*[\(（]\s*([^\)）]*)\s*[\)）]", text):
        found = True
        value = cleanup_lexical_item(match.group(2).strip())
        if value:
            add_unique(attrs, "nmm", value)
            if value.lower() == "neg":
                add_unique(attrs, "neg", "TRUE")

    text = re.sub(r"(?i)(^|[^A-Za-z])NMM\s*[\(（][^\)）]*[\)）]", " ", text)

    if re.search(r"(?i)(^|[^A-Za-z])NMM([^A-Za-z]|$)", text):
        found = True
        text = re.sub(r"(?i)(^|[^A-Za-z])NMM([^A-Za-z]|$)", " ", text)

    if found and not attrs.get("nmm"):
        attrs["nmm"] = "TRUE"

    return text.strip()


def parse_past_neg(text: str, attrs: Dict[str, str]) -> str:
    """Parse removable boolean markers PAST, NEG and the NOD shorthand.

    Examples:
      食べる PAST -> lexical_item=食べる, past=TRUE
      NMM(neg) is handled by parse_nmm, but standalone neg is handled here.
    """
    if re.search(r"(?i)(^|[^A-Za-z])PAST([^A-Za-z]|$)", text):
        add_unique(attrs, "past", "TRUE")
        text = re.sub(r"(?i)(^|[^A-Za-z])PAST([^A-Za-z]|$)", lambda m: (m.group(1) or " ") + (m.group(2) or " "), text)

    if re.search(r"(?i)(^|[^A-Za-z])NEG([^A-Za-z]|$)", text):
        add_unique(attrs, "neg", "TRUE")
        text = re.sub(r"(?i)(^|[^A-Za-z])NEG([^A-Za-z]|$)", lambda m: (m.group(1) or " ") + (m.group(2) or " "), text)

    # Standalone NMM shorthand. Example: nod -> nmm=nod, lexical_item empty.
    if re.search(r"(?i)(^|[^A-Za-z])NOD([^A-Za-z]|$)", text):
        add_unique(attrs, "nmm", "nod")
        text = re.sub(r"(?i)(^|[^A-Za-z])NOD([^A-Za-z]|$)", lambda m: (m.group(1) or " ") + (m.group(2) or " "), text)

    return text.strip()


def get_rep_hand_code(active_hand: Optional[str]) -> str:
    if active_hand == "rh":
        return "R"
    if active_hand == "lh":
        return "L"
    return "N"


def is_rep_marker_text(text: str) -> bool:
    """Return True for repetition markers like rep, rep2, 2rep, rep:2."""
    text = normalize_digits(str(text or "").strip())
    return bool(re.fullmatch(r"(?i)(?:REP\s*[:：]?\s*[0-9]*|[0-9]+\s*REP)", text))


def infer_repeated_word(text: str) -> str:
    """Infer which sign is repeated, before the rep markers are stripped.

    Examples:
      そうだ(rep) -> そうだ
      食べる rep2 -> 食べる
      RH(男)*2    -> 男
    """
    text = str(text or "").strip()
    text = normalize_digits(text)

    # Remove hand labels but keep their value.
    match = re.search(r"(?i)(?:LH|RH|L|R|左手|右手|左|右)\s*[\(（]\s*([^\)）]+)\s*[\)）]", text)
    if match:
        text = match.group(1).strip()

    match = re.search(r"(?i)(?:LH|RH|L|R|左手|右手|左|右)\s*[:：]\s*([^＋+/;,]+)", text)
    if match:
        text = match.group(1).strip()

    # If a bare PT annotation contains only a repetition marker, the repeated sign is PT itself.
    if re.fullmatch(r"(?i)PT[0-9]*\s*[\(（]\s*(?:REP\s*[:：]?\s*[0-9０-９]*|[0-9０-９]+\s*REP)\s*[\)）]", text):
        return "pt"

    # Remove rep markers and repetition symbols.
    text = re.sub(r"(?i)\bREP\s*[:：]?\s*[0-9０-９]*", "", text)
    text = re.sub(r"(?i)\b[0-9０-９]+\s*REP\b", "", text)
    text = re.sub(r"(?i)[\(（]\s*REP\s*[:：]?\s*[0-9０-９]*\s*[\)）]", "", text)
    text = re.sub(r"(?i)[\(（]\s*[0-9０-９]+\s*REP\s*[\)）]", "", text)
    text = re.sub(r"[×xX＊*]\s*[0-9０-９]+", "", text)

    # Remove common keywords that are not the repeated value.
    text = re.sub(r"(?i)\b(CL|FS|AW|GES|NMM|M|PT[0-9]*)\s*[:：]?", "", text)

    text = cleanup_lexical_item(text)

    # If comma-separated, use the whole cleaned expression.
    return text


def parse_rep(
    text: str,
    attrs: Dict[str, str],
    active_hand: Optional[str] = None,
    repeated_word: Optional[str] = None,
) -> str:
    hand_code = get_rep_hand_code(active_hand)
    repeated_word = cleanup_lexical_item(repeated_word or infer_repeated_word(text))

    def save_rep(number: str) -> None:
        number = normalize_digits(number or "1")
        word = repeated_word if repeated_word else "TRUE"
        add_unique(attrs, "rep", f"{word}({number};{hand_code})")

    # rep / rep2 / rep:2
    for match in re.finditer(r"(?i)\bREP\s*[:：]?\s*([0-9０-９]*)", text):
        value = match.group(1)
        save_rep(value if value else "1")

    text = re.sub(r"(?i)\bREP\s*[:：]?\s*[0-9０-９]*", "", text)

    # (rep) / (rep2)
    for match in re.finditer(r"(?i)[\(（]\s*REP\s*[:：]?\s*([0-9０-９]*)\s*[\)）]", text):
        value = match.group(1)
        save_rep(value if value else "1")

    text = re.sub(r"(?i)[\(（]\s*REP\s*[:：]?\s*[0-9０-９]*\s*[\)）]", "", text)

    # 2rep / (2rep)
    for match in re.finditer(r"(?i)[\(（]\s*([0-9０-９]+)\s*REP\s*[\)）]", text):
        save_rep(match.group(1))

    text = re.sub(r"(?i)[\(（]\s*[0-9０-９]+\s*REP\s*[\)）]", "", text)

    for match in re.finditer(r"(?i)\b([0-9０-９]+)\s*REP\b", text):
        save_rep(match.group(1))

    text = re.sub(r"(?i)\b[0-9０-９]+\s*REP\b", "", text)

    # *2 / ＊２ / x2
    for match in re.finditer(r"[×xX＊*]\s*([0-9０-９]+)", text):
        save_rep(match.group(1))

    text = re.sub(r"[×xX＊*]\s*[0-9０-９]+", "", text)

    return text.strip()


def parse_stop_hold_keep_index(text: str, attrs: Dict[str, str]) -> str:
    for key in ["stop", "hold", "keep", "index"]:
        if re.search(rf"(?i)(^|[^A-Za-z]){key}([^A-Za-z]|$)", text):
            attrs[key] = "TRUE"

    # Remove parenthetical marker forms first: 別(hold) -> 別, 見る(index) -> 見る
    for key in ["stop", "hold", "keep", "index"]:
        text = re.sub(
            rf"(?i)[\(（]\s*{key}(?:\s*[:：]\s*[^\)）]*)?\s*[\)）]",
            " ",
            text,
        )

    text = re.sub(r"(?i)(^|[^A-Za-z])STOP\s*[:：]?", " ", text)
    text = re.sub(r"(?i)(^|[^A-Za-z])HOLD\s*[:：]?", " ", text)
    text = re.sub(r"(?i)(^|[^A-Za-z])KEEP\s*[:：]?", " ", text)
    text = re.sub(r"(?i)(^|[^A-Za-z])INDEX\s*[:：]?", " ", text)

    return text.strip()


def parse_qm(text: str, attrs: Dict[str, str]) -> str:
    if "?" in text or "？" in text:
        attrs["qm"] = "TRUE"

    if re.search(r"(?i)(^|[^A-Za-z])QM([^A-Za-z]|$)", text):
        attrs["qm"] = "TRUE"

    text = text.replace("?", "").replace("？", "")
    text = re.sub(r"(?i)(^|[^A-Za-z])QM\s*[:：]?", " ", text)

    return text.strip()


def parse_dw(text: str, attrs: Dict[str, str]) -> str:
    """Parse DW (depicting word) information separately from PT.

    Examples:
      pt3(dw:5種類) -> pt=3, dw=5種類, lexical_item keeps 5種類
      pt:dw(みんな) -> pt=0, dw=みんな, lexical_item keeps みんな
      dw:5種類      -> dw=5種類, lexical_item keeps 5種類
    """
    def save(value: str) -> str:
        value = cleanup_lexical_item(value)
        if value:
            add_unique(attrs, "dw", value)
            return value
        return ""

    # cl:dw / cl:dr (DR is normalized to DW) means the classifier is DW,
    # and the DW value itself is unknown/boolean.
    def repl_cl_dw(match):
        add_unique(attrs, "cl", "dw")
        add_unique(attrs, "dw", "TRUE")
        return (match.group(1) or "") + " "

    text = re.sub(
        r"(?i)(^|[^A-Za-z])CL\s*[:：]\s*DW(?![A-Za-z])",
        repl_cl_dw,
        text,
    )

    # Lexical suffix DW/DR: 同じdw -> lexical_item=同じ, dw=TRUE
    def repl_lexical_dw_suffix(match):
        add_unique(attrs, "dw", "TRUE")
        return (match.group(1) or "") + cleanup_lexical_item(match.group(2))

    text = re.sub(
        r"(?i)(^|[^A-Za-z])([぀-ヿ一-鿿々〆〤ー〒0-9０-９]+)DW(?![A-Za-z])",
        repl_lexical_dw_suffix,
        text,
    )

    # Compact PT-number + DW value: pt3dw(4つの具材)
    def repl_compact_pt_number_dw_paren(match):
        pt_number = normalize_digits(match.group(2) or "0")
        add_unique(attrs, "pt", pt_number if pt_number else "0")
        add_unique(attrs, "pt", "dw")
        value = save(match.group(3))
        attrs["_force_clean_pt_number"] = pt_number if pt_number else "0"
        return (match.group(1) or "") + (value if value else " ")

    text = re.sub(
        r"(?i)(^|[^A-Za-z])PT([0-9]+)DW\s*[\(（]\s*([^\)）]*)[\)）]?",
        repl_compact_pt_number_dw_paren,
        text,
    )

    # Compact PT-number + bare DW: pt3dw -> pt=3;dw, dw=TRUE, lexical_item=pt3
    def repl_compact_pt_number_dw_bare(match):
        pt_number = normalize_digits(match.group(2) or "0")
        add_unique(attrs, "pt", pt_number if pt_number else "0")
        add_unique(attrs, "pt", "dw")
        add_unique(attrs, "dw", "TRUE")
        attrs["_force_clean_pt_number"] = pt_number if pt_number else "0"
        return (match.group(1) or "") + " "

    text = re.sub(
        r"(?i)(^|[^A-Za-z])PT([0-9]+)DW(?!\s*[:：\(（])(?=$|[^A-Za-z])",
        repl_compact_pt_number_dw_bare,
        text,
    )

    # Legacy compact PTDW:value -> pt=dw, dw=value, lexical_item=pt;value
    def repl_legacy_ptdw(match):
        add_unique(attrs, "pt", "dw")
        value = save(match.group(2))
        if value:
            attrs["_force_clean_pt"] = "TRUE"
            return (match.group(1) or "") + value
        return " "

    text = re.sub(
        r"(?i)(^|[^A-Za-z])PTDW\s*[:：]\s*([^＋+/;,()\s]*)",
        repl_legacy_ptdw,
        text,
    )

    # Legacy compact PTDW(value) -> pt=dw, dw=value, lexical_item=pt;value
    def repl_legacy_ptdw_paren(match):
        add_unique(attrs, "pt", "dw")
        value = save(match.group(2))
        if value:
            attrs["_force_clean_pt"] = "TRUE"
            return (match.group(1) or "") + value
        add_unique(attrs, "dw", "TRUE")
        attrs["_force_clean_pt"] = "TRUE"
        return (match.group(1) or "") + " "

    text = re.sub(
        r"(?i)(^|[^A-Za-z])PTDW\s*[\(（]\s*([^\)）]*)[\)）]?",
        repl_legacy_ptdw_paren,
        text,
    )

    # Bare legacy compact PTDW -> pt=dw, dw=TRUE, lexical_item=pt
    def repl_bare_legacy_ptdw(match):
        add_unique(attrs, "pt", "dw")
        add_unique(attrs, "dw", "TRUE")
        attrs["_force_clean_pt"] = "TRUE"
        return (match.group(1) or "") + " "

    text = re.sub(
        r"(?i)(^|[^A-Za-z])PTDW(?!\s*[:：])(?=$|[^A-Za-z])",
        repl_bare_legacy_ptdw,
        text,
    )

    # pt:dw(みんな) / pt2:dw(みんな)
    def repl_pt_dw_paren(match):
        pt_number = normalize_digits(match.group(2) or "0")
        add_unique(attrs, "pt", pt_number if pt_number else "0")
        value = save(match.group(3))
        return (match.group(1) or "") + (value if value else " ")

    text = re.sub(
        r"(?i)(^|[^A-Za-z])PT([0-9]*)\s*[:：]\s*DW\s*[\(（]\s*([^\)）]*)[\)）]?",
        repl_pt_dw_paren,
        text,
    )

    # pt:dw:みんな / pt2:dw:みんな
    def repl_pt_dw_colon(match):
        pt_number = normalize_digits(match.group(2) or "0")
        add_unique(attrs, "pt", pt_number if pt_number else "0")
        value = save(match.group(3))
        return (match.group(1) or "") + (value if value else " ")

    text = re.sub(
        r"(?i)(^|[^A-Za-z])PT([0-9]*)\s*[:：]\s*DW\s*[:：]\s*([^＋+/;,()\s]*)",
        repl_pt_dw_colon,
        text,
    )

    # (dw:5種類) / (dw 5種類)
    def repl_paren(match):
        value = save(match.group(1))
        return ";" + value if value else " "

    text = re.sub(r"[\(（]\s*DW\s*[:：]?\s*([^\)）]*)[\)）]", repl_paren, text, flags=re.I)

    # standalone dw:5種類
    def repl_standalone(match):
        value = save(match.group(2))
        return " " + value if value else " "

    text = re.sub(r"(?i)(^|[^A-Za-z])DW\s*[:：]\s*([^＋+/;,()\s]*)", repl_standalone, text)
    return text.strip()


def parse_pt_value(value: str, attrs: Dict[str, str], active_hand: Optional[str] = None) -> None:
    value = normalize_digits(value)

    for match in re.finditer(r"(?i)(^|[^A-Za-z])PT([0-9]*)", value):
        number = match.group(2)
        add_unique(attrs, "pt", number if number else "0")


def pt_column_to_lexical_item(pt_value: str) -> str:
    """Render a ``pt`` column value as the lexical item of a pointing sign.

    The lexical item of any pointing annotation is the pointing sign itself:
    ``pt`` when unnumbered, ``pt`` + the number otherwise. The Japanese material
    an annotator writes alongside PT names the *referent* being pointed at, not
    a signed word, so it never becomes lexical material.

      ""      -> ""      (not a pointing annotation)
      "0"     -> "pt"
      "2"     -> "pt2"
      "dw"    -> "pt"
      "3;dw"  -> "pt3"   (first value wins)
    """
    values = [part.strip() for part in str(pt_value or "").split(";") if part.strip()]
    if not values:
        return ""

    for value in values:
        digits = normalize_digits(value)
        if digits.isdigit():
            return "pt" if digits == "0" else "pt" + digits

    # Non-numeric values such as "dw" carry no pointing number.
    return "pt"


def parse_pt(text: str, attrs: Dict[str, str], active_hand: Optional[str] = None) -> str:
    """Parse pointing forms and record the pointing number in ``pt``.

    Every recognised form sets ``pt``; the lexical item is then derived from
    that column by :func:`pt_column_to_lexical_item`, so all pointing
    annotations produce ``pt`` + number regardless of how they were written:

      pt          -> pt=0,  lexical_item=pt
      pt2         -> pt=2,  lexical_item=pt2
      PT1＝歯     -> pt=1,  lexical_item=pt1   (歯 is the referent, dropped)
      pt:体       -> pt=0,  lexical_item=pt
      pt2:ひよこ  -> pt=2,  lexical_item=pt2
      PT3みんな   -> pt=3,  lexical_item=pt3
      PT(アニメ)  -> pt=0,  lexical_item=pt
      狙う(pt)    -> pt=0,  lexical_item=pt
    """
    text = normalize_digits(text)

    # Lexical value with a parenthetical PT marker: 狙う(pt) -> lexical_item=狙う, pt=0.
    # This must run before the PT(object) rule below.
    for match in re.finditer(r"(?i)[\(（]\s*PT([0-9]*)\s*[\)）]", text):
        add_unique(attrs, "pt", match.group(1) if match.group(1) else "0")
    text = re.sub(r"(?i)[\(（]\s*PT[0-9]*\s*[\)）]", "", text)

    # PT(アニメ), PT(妻), PT3(二つ目), and malformed PT(アニメ without a closing
    # parenthesis. The object inside PT(...) is only the thing pointed at, not
    # the lexical sign itself, so lexical_item keeps pt/pt2/pt3, not the object.
    pattern_obj_closed = r"(?i)(^|[^A-Za-z])PT([0-9]*)\s*[\(（]\s*([^\)）]*)[\)）]"

    def repl_pt_object_closed(match):
        add_unique(attrs, "pt", match.group(2) if match.group(2) else "0")
        inner = match.group(3)
        if is_rep_marker_text(inner):
            return (match.group(1) or "") + "pt(" + inner.strip() + ")"
        digits = match.group(2) or ""
        return (match.group(1) or "") + "pt" + digits + ";"

    text = re.sub(pattern_obj_closed, repl_pt_object_closed, text)

    pattern_obj_unclosed = r"(?i)(^|[^A-Za-z])PT([0-9]*)\s*[\(（]\s*([^\)）]*)$"

    def repl_pt_object_unclosed(match):
        add_unique(attrs, "pt", match.group(2) if match.group(2) else "0")
        inner = match.group(3)
        if is_rep_marker_text(inner):
            return (match.group(1) or "") + "pt(" + inner.strip() + ")"
        digits = match.group(2) or ""
        return (match.group(1) or "") + "pt" + digits

    text = re.sub(pattern_obj_unclosed, repl_pt_object_unclosed, text)

    # PT1＝歯 / PT1=歯 / pt2:ひよこ / pt:体
    pattern_sep = r"(?i)(^|[^A-Za-z])PT([0-9]*)\s*[=:：＝]\s*([^＋+/;,()\s]+)"
    for match in re.finditer(pattern_sep, text):
        add_unique(attrs, "pt", match.group(2) if match.group(2) else "0")
    text = re.sub(pattern_sep, lambda m: (m.group(1) or "") + cleanup_lexical_item(m.group(3)), text)

    # PT3みんな / PT1歯, but avoid matching bare pt/pt2 only.
    pattern_attached = r"(?i)(^|[^A-Za-z])PT([0-9]+)([぀-ヿ一-鿿々〆〤ー][^＋+/;,()\s]*)"
    for match in re.finditer(pattern_attached, text):
        add_unique(attrs, "pt", match.group(2) if match.group(2) else "0")
    text = re.sub(pattern_attached, lambda m: (m.group(1) or "") + cleanup_lexical_item(m.group(3)), text)

    # Bare PT / PT2.
    parse_pt_value(text, attrs, active_hand)

    def repl_bare(match):
        digits = match.group(2) or ""
        return (match.group(1) or "") + "pt" + digits

    text = re.sub(r"(?i)(^|[^A-Za-z])PT([0-9]*)(?![぀-ヿ一-鿿々〆〤ー])", repl_bare, text)
    return text.strip()


def parse_fs(text: str, attrs: Dict[str, str]) -> str:
    match = re.search(r"(?i)(^|[^A-Za-z])FS\s*[:：]\s*([^()\[\]+/;,＋+\s]+)", text)
    if match:
        value = cleanup_lexical_item(match.group(2).strip())
        add_unique(attrs, "fs", value)
        text = text[:match.start()] + value + text[match.end():]
        return text.strip()

    match = re.search(r"(?i)^FS\s*([^\s()\[\]+/;,＋+]+)", text)
    if match:
        value = cleanup_lexical_item(match.group(1).strip())
        add_unique(attrs, "fs", value)
        text = value + text[match.end():]
        return text.strip()

    text = re.sub(r"(?i)(^|[^A-Za-z])FS\s*[:：]?", " ", text)
    return text.strip()


def parse_aw(text: str, attrs: Dict[str, str]) -> str:
    match = re.search(r"(?i)(^|[^A-Za-z])AW\s*[:：]\s*([^()\[\]+/;,＋+\s]+)", text)
    if match:
        value = cleanup_lexical_item(match.group(2).strip())
        add_unique(attrs, "aw", value)
        text = text[:match.start()] + value + text[match.end():]
        return text.strip()

    match = re.search(r"(?i)^AW\s+([^\s()\[\]+/;,＋+]+)", text)
    if match:
        value = cleanup_lexical_item(match.group(1).strip())
        add_unique(attrs, "aw", value)
        text = value + text[match.end():]
        return text.strip()

    text = re.sub(r"(?i)(^|[^A-Za-z])AW\s*[:：]?", " ", text)
    return text.strip()


def save_mouth_value(value: str, attrs: Dict[str, str]) -> None:
    """Save a mouth value and its related flags.

    Mouth notes may contain UN as a mouth descriptor, e.g. M:un(きった). In that
    case UN fills the separate ``un`` column, but must not block the whole
    annotation as an unknown lexical item.
    """
    value = cleanup_lexical_item(value)
    if not value:
        return
    add_unique(attrs, "m", value)
    if re.search(r"(?i)(^|[^A-Za-z])UN([^A-Za-z]|$)", value) or re.match(r"(?i)^UN\s*[(:：]", value):
        add_unique(attrs, "un", "TRUE")
    if re.search(r"(?i)(^|[^A-Za-z])NEG([^A-Za-z]|$)", value) or re.match(r"(?i)^NEG\s*[(:：]", value):
        add_unique(attrs, "neg", "TRUE")


def parse_mouth_notes_early(text: str, attrs: Dict[str, str]) -> str:
    """Extract parenthetical mouth notes before the FAL/UN/D blocking checks.

    This prevents mouth-only values like 口型:un(きった) from being treated as
    blocking UN annotations. Handles nested or unbalanced parentheses:
      cl:包丁で切る(口型:un(きった)   -> cl:包丁で切る, m=un(きった), un=TRUE
      cl:(野菜を)突く(口型:un(...))   -> cl:(野菜を)突く, m=un(...), un=TRUE
    """
    text = normalize_keywords(str(text or ""))
    result: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch not in "（(":
            result.append(ch)
            i += 1
            continue

        j = i + 1
        while j < n and text[j].isspace():
            j += 1

        if j < n and text[j].upper() == "M":
            k = j + 1
            while k < n and text[k].isspace():
                k += 1
            if k < n and text[k] in ":：=＝は":
                start_value = k + 1
                depth = 1
                pos = start_value
                while pos < n:
                    if text[pos] in "（(":
                        depth += 1
                    elif text[pos] in "）)":
                        depth -= 1
                        if depth == 0:
                            break
                    pos += 1

                # If no matching close exists, consume to the end.
                end_value = pos if pos < n and depth == 0 else n
                value = text[start_value:end_value].strip()
                save_mouth_value(value, attrs)
                i = pos + 1 if pos < n and depth == 0 else n
                result.append(" ")
                continue

        # Not a mouth note; keep the character.
        result.append(ch)
        i += 1

    return "".join(result).strip()


def parse_m(text: str, attrs: Dict[str, str]) -> str:
    # Extract M inside parentheses and keep the mouth value in lexical_item later.
    def repl_paren(match):
        value = cleanup_lexical_item(match.group(1))
        if value:
            save_mouth_value(value, attrs)
            return ";" + value
        return " "

    text = re.sub(r"[\(（]\s*M\s*(?:[:：=＝]|は)\s*([^\)）]*)[\)）]", repl_paren, text, flags=re.I)

    # Extract standalone M:value
    def repl_standalone(match):
        value = cleanup_lexical_item(match.group(2))
        if value:
            save_mouth_value(value, attrs)
            return " " + value
        return " "

    text = re.sub(r"(?i)(^|[^A-Za-z])M\s*(?:[:：=＝]|は)\s*([^＋+/;,()\s]*)", repl_standalone, text)

    return text.strip()


def _as_component(value: str) -> str:
    """Wrap a marker value being substituted back into the text as its own component."""
    value = str(value or "").strip()
    return ";" + value + ";" if value else " "


def marker_value_components(attrs: Dict[str, str], keys: Iterable[str]) -> Set[str]:
    """Every lexical value already owned by one of ``keys``."""
    owned: Set[str] = set()
    for key in keys:
        for value in str(attrs.get(key, "") or "").split(";"):
            value = cleanup_lexical_item(value)
            if value and not re.fullmatch(r"(?i)TRUE", value):
                owned.add(value)
    return owned


def parse_marker_with_value(text: str, attrs: Dict[str, str], key: str) -> str:
    """Save a KEY:value / KEY(value) marker and leave its value in the text.

    The value is put back as its own ``;`` component rather than joined with a
    space. A space is not a word boundary in Japanese, so ``fs:\u304a(ges:\u3042)`` used
    to leave the residue ``\u304a \u3042``: one blob that the fingerspelled-material
    snapshot in parse_segment could not tell apart, which is how the gesture
    ended up inside the fs column.
    """
    # key:value
    pattern = rf"(?i)(^|[^A-Za-z]){key}\s*[:：]\s*([^＋+/;,()\s]+)"
    for match in re.finditer(pattern, text):
        value = cleanup_lexical_item(match.group(2))
        if value:
            add_unique(attrs, key.lower(), value)

    text = re.sub(pattern, lambda m: _as_component(cleanup_lexical_item(m.group(2))), text)

    # key(value)
    pattern_paren = rf"(?i)(^|[^A-Za-z]){key}\s*[\(（]\s*([^\)）]*)[\)）]"
    for match in re.finditer(pattern_paren, text):
        value = cleanup_lexical_item(match.group(2))
        if value:
            add_unique(attrs, key.lower(), value)

    text = re.sub(pattern_paren,
                  lambda m: _as_component(cleanup_lexical_item(m.group(2))), text)

    # bare key
    if re.search(rf"(?i)(^|[^A-Za-z]){key}([^A-Za-z]|$)", text):
        add_unique(attrs, key.lower(), "TRUE")
        text = re.sub(rf"(?i)(^|[^A-Za-z]){key}\s*[:：]?", " ", text)

    return text.strip()


def parse_d_fal_un(text: str, attrs: Dict[str, str], key: str) -> Tuple[str, bool]:
    key_lower = key.lower()
    found = False

    # key(value)
    pattern_paren = rf"(?i)(^|[^A-Za-z]){key}\s*[\(（]\s*([^\)）]*)[\)）]"
    for _ in re.finditer(pattern_paren, text):
        found = True
        add_unique(attrs, key_lower, "TRUE")

    text = re.sub(pattern_paren, " ", text)

    # key:value
    pattern_value = rf"(?i)(^|[^A-Za-z]){key}\s*[:：]\s*([^＋+/;,()\s]+)"
    for _ in re.finditer(pattern_value, text):
        found = True
        add_unique(attrs, key_lower, "TRUE")

    text = re.sub(pattern_value, " ", text)

    # bare key
    if re.search(rf"(?i)(^|[^A-Za-z]){key}([^A-Za-z]|$)", text):
        found = True
        add_unique(attrs, key_lower, "TRUE")
        text = re.sub(rf"(?i)(^|[^A-Za-z]){key}\s*[:：]?", " ", text)

    return text.strip(), found


def split_classifier_value_and_annotations(value: str, attrs: Dict[str, str]) -> str:
    """Clean a CL value while extracting embedded GES notes and extra CL objects.

    Examples:
      人型:(ges:へえそうなんですか) -> CL 人型, GES value
      人型(説明):(ges:)             -> CL 人型(説明), GES TRUE
      人型:(猫)(ges:あ！)           -> CL 人型, lexical extra 猫, GES value
    """
    raw = normalize_keywords(str(value or "")).strip()
    raw = re.sub(r"/+$", "", raw).strip()

    # Extract gesture notes but do NOT leave the gesture wording as CL residue.
    def repl_ges_paren(match):
        inner = cleanup_lexical_item(match.group(1))
        add_unique(attrs, "ges", inner if inner else "TRUE")
        return " "

    raw = re.sub(r"(?i)[\(（]\s*GES\s*[:：]?\s*([^\)）]*)[\)）]", repl_ges_paren, raw)

    def repl_ges_colon(match):
        inner = cleanup_lexical_item(match.group(2))
        add_unique(attrs, "ges", inner if inner else "TRUE")
        return match.group(1) or " "

    raw = re.sub(r"(?i)(^|[^A-Za-z])GES\s*[:：]\s*([^＋+/;,()\s]*)", repl_ges_colon, raw)

    lexical_extras: List[str] = []
    raw = re.sub(r"[:：]\s*$", "", raw).strip()

    # 人型:(猫) -> CL 人型 + clean extra 猫.
    match = re.match(r"^(.+?)\s*[:：]\s*[\(（]([^\)）]*)[\)）]\s*$", raw)
    if match:
        base = cleanup_lexical_item(match.group(1))
        extra = cleanup_lexical_item(match.group(2))
        if extra:
            lexical_extras.append(extra)
        raw = base

    raw = cleanup_lexical_item(raw)
    if lexical_extras:
        return raw + ";" + ";".join(lexical_extras)
    return raw


def parse_cl(text: str, attrs: Dict[str, str]) -> str:
    # Special marker-only CL depiction forms: cl:dw / cl:dr.
    if re.fullmatch(r"(?i)\s*CL\s*[:：]\s*DW\s*/?\s*", str(text or "")):
        add_unique(attrs, "cl", "dw")
        add_unique(attrs, "dw", "TRUE")
        return ""

    # CL:value. Stop before top-level separators; then clean embedded notes.
    pattern = r"(?i)(^|[^A-Za-z])CL\s*[:：]\s*([^＋+/;,]+)"
    for match in re.finditer(pattern, text):
        value = split_classifier_value_and_annotations(match.group(2), attrs)
        cl_value = cleanup_lexical_item(str(value).split(";")[0])
        if cl_value and not re.fullmatch(r"(?i)GES\s*[:：]?", cl_value):
            add_unique(attrs, "cl", cl_value)

    def repl_cl_value(match):
        value = split_classifier_value_and_annotations(match.group(2), attrs)
        return " " + cleanup_lexical_item(value)

    text = re.sub(pattern, repl_cl_value, text)

    # CL(value)
    pattern_paren = r"(?i)(^|[^A-Za-z])CL\s*[\(（]\s*([^\)）]*)[\)）]"
    for match in re.finditer(pattern_paren, text):
        value = split_classifier_value_and_annotations(match.group(2), attrs)
        cl_value = cleanup_lexical_item(str(value).split(";")[0])
        if cl_value:
            add_unique(attrs, "cl", cl_value)

    text = re.sub(
        pattern_paren,
        lambda m: " " + cleanup_lexical_item(split_classifier_value_and_annotations(m.group(2), attrs)),
        text,
    )

    # bare CL
    if re.search(r"(?i)(^|[^A-Za-z])CL([^A-Za-z]|$)", text):
        add_unique(attrs, "cl", "TRUE")
        text = re.sub(r"(?i)(^|[^A-Za-z])CL\s*[:：]?", " ", text)

    return text.strip()


def clean_hand_value_and_parse_inner(value: str, attrs: Dict[str, str], hand: str) -> str:
    """Clean the value inside LH/RH while still parsing nested FS/M/PT/CL/GES markers."""
    raw_value = normalize_digits(str(value or "").strip())
    raw_value = normalize_keywords(raw_value)

    # Save hold/stop/keep/index before cleaning the hand value.
    value_no_markers = parse_stop_hold_keep_index(raw_value, attrs)

    # Repetition is saved with hand information; the hand value is cleaned after.
    value_no_rep = parse_rep(
        value_no_markers,
        attrs,
        active_hand=hand,
        repeated_word=infer_repeated_word(value_no_markers),
    )

    # DW can appear inside a PT hand stream: rh:pt3(dw:5種類)
    value_no_rep = parse_dw(value_no_rep, attrs)

    # FS inside a hand stream: save the fs value.
    fs_match = re.search(r"(?i)^FS\s*[:：]\s*([^()\[\]+/;,＋+\s]+)", value_no_rep)
    if fs_match:
        fs_value = cleanup_lexical_item(fs_match.group(1))
        if fs_value:
            add_unique(attrs, "fs", fs_value)

    value_for_clean = value_no_rep
    value_for_clean = parse_m(value_for_clean, attrs)
    value_for_clean = parse_cl(value_for_clean, attrs)
    value_for_clean = parse_marker_with_value(value_for_clean, attrs, "GES")
    value_for_clean = parse_fs(value_for_clean, attrs)
    value_for_clean = parse_aw(value_for_clean, attrs)
    value_for_clean = parse_pt(value_for_clean, attrs, active_hand=hand)

    parse_pt_value(value_for_clean, attrs, hand)

    hand_value = cleanup_lexical_item(value_for_clean)
    if not hand_value:
        for key in ("ges", "fs", "aw", "cl", "m"):
            if attrs.get(key):
                hand_value = cleanup_lexical_item(str(attrs[key]).split(";")[0])
                break

    return hand_value


def parse_embedded_parenthetical_hand_streams(text: str, attrs: Dict[str, str]) -> str:
    pattern = r"[\(（]([^\)）]*(?i:(?:^|[\s/＋+])(RH|R|LH|L)|(?:^|[\s/＋+])(右手|左手|左|右))\s*[:：][^\)）]*)[\)）]"

    def repl(match):
        inner = match.group(1).strip()
        parse_hand_markers(inner, attrs)
        return " "

    return re.sub(pattern, repl, text)


def parse_hand_markers(text: str, attrs: Dict[str, str]) -> Tuple[str, Optional[str]]:
    last_hand = None

    paren_pattern = r"(?i)(?:^|[\s/＋+])(?:\s*)(LH|RH|L|R|左手|右手|左|右)\s*[\(（]\s*([^\)）]+)\s*[\)）]"

    for match in re.finditer(paren_pattern, text):
        raw_marker = match.group(1)
        hand = MARKER_TO_HAND.get(raw_marker.upper(), MARKER_TO_HAND.get(raw_marker))
        if not hand:
            continue

        value = clean_hand_value_and_parse_inner(match.group(2), attrs, hand)
        last_hand = hand
        add_unique(attrs, hand, value)
        parse_pt_value(value, attrs, hand)

    text = re.sub(paren_pattern, " ", text)

    # Colon forms. The lookahead stops a value before the next adjacent hand
    # marker: L:娘R:娘 -> lh=娘, rh=娘
    pattern = r"(?i)(?<![A-Za-z])(LH|RH|L|R|左手|右手|左|右)\s*[:：]\s*(.*?)(?=(?<![A-Za-z])(?:LH|RH|L|R|左手|右手|左|右)\s*[:：]|[＋+/;,]|$)"

    for match in list(re.finditer(pattern, text)):
        raw_marker = match.group(1)
        hand = MARKER_TO_HAND.get(raw_marker.upper(), MARKER_TO_HAND.get(raw_marker))
        if not hand:
            continue

        value = clean_hand_value_and_parse_inner(match.group(2), attrs, hand)
        last_hand = hand
        add_unique(attrs, hand, value)
        parse_pt_value(value, attrs, hand)

    text = re.sub(pattern, " ", text)
    return text.strip(), last_hand


def add_hand_from_lexical_item_parentheses(lexical_item: str, attrs: Dict[str, str]) -> None:
    """Keep lexical_item as-is but also save the base word to RH/LH.

    楽しい(右手だけ) -> rh=楽しい
    楽しい(左手だけ) -> lh=楽しい
    """
    text = str(lexical_item or "").strip()
    match = re.match(r"^(.+?)[\(（]([^\)）]*(右手|左手|右|左)[^\)）]*)[\)）]$", text)
    if not match:
        return

    base_word = cleanup_lexical_item(match.group(1))
    hand_info = match.group(2)

    if not base_word:
        return

    if "右手" in hand_info or "右" in hand_info:
        add_unique(attrs, "rh", base_word)

    if "左手" in hand_info or "左" in hand_info:
        add_unique(attrs, "lh", base_word)


def strip_hand_note_from_lexical_item_and_fill_attrs(lexical_item: str, attrs: Dict[str, str]) -> str:
    """Remove hand-only notes from lexical_item and save the base word to LH/RH.

    美味しい(右手ver) -> lexical_item=美味しい, rh=美味しい
    楽しい(左手だけ)  -> lexical_item=楽しい, lh=楽しい

    Semantic notes that do not mention left/right hands are kept.
    """
    cleaned_parts: List[str] = []
    for part in str(lexical_item or "").split(";"):
        part = cleanup_lexical_item(part)
        if not part:
            continue

        match = re.fullmatch(r"(.+?)[\(（]([^\)）]*(右手|左手|右|左)[^\)）]*)[\)）]", part)
        if not match:
            cleaned_parts.append(part)
            continue

        base_word = cleanup_lexical_item(match.group(1))
        hand_info = match.group(2)
        if not base_word:
            cleaned_parts.append(part)
            continue

        if "右手" in hand_info or "右" in hand_info:
            add_unique(attrs, "rh", base_word)
        if "左手" in hand_info or "左" in hand_info:
            add_unique(attrs, "lh", base_word)

        cleaned_parts.append(base_word)

    result: List[str] = []
    for part in cleaned_parts:
        if part and part not in result:
            result.append(part)
    return ";".join(result)


def parse_braced_hand_components(text: str, attrs: Dict[str, str]) -> str:
    """Parse hand specifications inside braces and remove the braces.

    cl:家の裏{家(LH)+裏(RH)} -> remaining text cl:家の裏, lh=家, rh=裏
    """
    def repl(match):
        inner = match.group(1).strip()
        for component in split_outside_brackets(inner, "+"):
            component = component.strip()
            if not component:
                continue

            paren = re.fullmatch(
                r"(.+?)[\(（]\s*(LH|RH|L|R|左手|右手|左|右)\s*[\)）]",
                component,
                flags=re.I,
            )
            colon = re.fullmatch(
                r"(?i)(LH|RH|L|R|左手|右手|左|右)\s*[:：]\s*(.+)",
                component,
            )

            if paren:
                value = cleanup_lexical_item(paren.group(1))
                raw_marker = paren.group(2)
            elif colon:
                raw_marker = colon.group(1)
                value = cleanup_lexical_item(colon.group(2))
            else:
                continue

            hand = MARKER_TO_HAND.get(raw_marker.upper(), MARKER_TO_HAND.get(raw_marker))
            if hand and value:
                add_unique(attrs, hand, value)

        return " "

    return re.sub(r"[\{｛]([^\}｝]*)[\}｝]", repl, str(text or ""))


def parse_both_hands_pt_shortcut(
    text: str,
    whole_annotation: str,
    attrs: Dict[str, str],
) -> Optional[Dict[str, str]]:
    """Parse the shorthand where both hands perform pointing.

    両手:pt  -> lexical_item=pt,  pt=0, lh=pt,  rh=pt
    両手:pt/ -> lexical_item=pt,  pt=0, lh=pt,  rh=pt
    両手:pt3 -> lexical_item=pt3, pt=3, lh=pt3, rh=pt3
    """
    match = re.fullmatch(r"(?i)両手\s*[:：]\s*PT([0-9]*)\s*/?", str(text or "").strip())
    if not match:
        return None

    digits = normalize_digits(match.group(1) or "")
    pt_value = digits if digits else "0"
    clean_value = "pt" + digits

    attrs["pt"] = pt_value
    attrs["lh"] = clean_value
    attrs["rh"] = clean_value

    return {
        "annotation": whole_annotation,
        "lexical_item": clean_value,
        **attrs,
        "compound": "",
        "ambiguous": "",
    }


def extract_multiple_bare_blocking_markers(text: str) -> Dict[str, str]:
    """Detect compact marker-only combinations such as UN(FAL) or FAL(UN).

    These save marker names, not TRUE, and not the inner marker as a value.
    """
    raw = normalize_digits(str(text or "").strip())
    raw = re.sub(r"^/+", "", raw).strip()
    result: Dict[str, str] = {}

    match = re.fullmatch(r"(?i)(FAL|UN|D)\s*[\(（]\s*(FAL|UN|D)\s*[\)）]", raw)
    if match:
        for marker in match.groups():
            result[marker.lower()] = "TRUE"
        return result

    match = re.fullmatch(r"(?i)(FAL|UN|D)\s*[:：]?\s*$", raw)
    if match:
        result[match.group(1).lower()] = "TRUE"

    return result


def extract_blocking_marker(text: str) -> Tuple[Optional[str], str]:
    """Detect FAL / UN / D and return the target column.

    Detecting one of these prevents inner parsing into cl/fs/rh/lh and friends.

      D:fs:て(RH)    -> ("d", "TRUE")
      cl:猫の形(fal) -> ("fal", "TRUE")
      FAL:fs:あ      -> ("fal", "TRUE")
      UN:RH:猫       -> ("un", "TRUE")
    """
    raw = str(text or "").strip()
    raw = re.sub(r"^/+", "", raw).strip()

    # Prefix marker: D:xxx, FAL:xxx, UN:xxx
    #
    # The (?![A-Za-z]) guard is essential. Without it the optional colon lets
    # the leading "d" of dw:5種類 (and dr:, and any annotation starting with a
    # Latin d) match as the D marker, which then blanks every other column.
    # The marker must be a whole token, not the first letter of a longer one.
    match = re.match(r"(?i)^(FAL|UN|D)(?![A-Za-z])\s*[:：]?\s*(.*)$", raw)
    if match:
        return match.group(1).lower(), "TRUE"

    # Suffix parenthetical marker: xxx(fal), xxx(un), xxx(d)
    match = re.search(r"(?i)[\(（]\s*(FAL|UN|D)\s*[\)）]\s*$", raw)
    if match:
        return match.group(1).lower(), "TRUE"

    return None, ""


def hand_value_to_lexical_component(value: str) -> str:
    """Convert a stored LH/RH value into lexical lexical_item material."""
    value = cleanup_lexical_item(value)
    if not value:
        return ""

    # fs:お(M:ロースポーク) -> お;ロースポーク
    temp_attrs = empty_attrs()
    text = normalize_keywords(value)
    text = parse_dw(text, temp_attrs)
    text = parse_m(text, temp_attrs)
    text = parse_cl(text, temp_attrs)
    text = parse_pt(text, temp_attrs)
    text = parse_fs(text, temp_attrs)
    text = parse_aw(text, temp_attrs)
    text = cleanup_lexical_item(text)

    parts: List[str] = []
    candidates = [
        text,
        temp_attrs.get("fs", ""),
        temp_attrs.get("aw", ""),
        temp_attrs.get("cl", ""),
        temp_attrs.get("m", ""),
        temp_attrs.get("dw", ""),
    ]
    for candidate in candidates:
        for part in str(candidate or "").split(";"):
            part = cleanup_lexical_item(part)
            if part and not re.fullmatch(r"(?i)pt[0-9]*|cl|fs|aw|TRUE", part) and part not in parts:
                parts.append(part)
    return ";".join(parts)


# ---------------------------------------------------------------------------
# Segment / annotation level parsing
# ---------------------------------------------------------------------------

def _normalise_lexical_components(lexical_item: str) -> str:
    """Clean, de-duplicate and re-join the ``;`` components of a lexical_item."""
    parts: List[str] = []
    for part in str(lexical_item or "").split(";"):
        part = cleanup_lexical_item(part)
        if part and part not in parts:
            parts.append(part)
    return ";".join(parts)


def parse_segment(
    segment: str,
    whole_annotation: str,
    inherited_hand: Optional[str] = None,
) -> Tuple[Dict[str, str], Optional[str]]:
    attrs = empty_attrs()

    # A Japanese explanation can contain words like 左手 inside an RH
    # description; only infer from the explanation when there is no explicit
    # hand marker.
    if re.search(r"(?i)(^|[^A-Za-z])(LH|RH|L|R)\s*[:：(（]|左手\s*[:：]|右手\s*[:：]", str(segment or "")):
        hand_from_explanation = None
    else:
        hand_from_explanation = detect_hand_from_japanese_explanation(segment)

    text = normalize_annotation(segment)
    text = strip_compound_marks(text)
    text = parse_braced_hand_components(text, attrs)

    both_hands_pt_row = parse_both_hands_pt_shortcut(text, whole_annotation, attrs)
    if both_hands_pt_row is not None:
        return both_hands_pt_row, inherited_hand

    bare_blocking = extract_multiple_bare_blocking_markers(text)
    if bare_blocking:
        for marker_key, marker_value in bare_blocking.items():
            add_unique(attrs, marker_key, marker_value)
        return {
            "annotation": whole_annotation,
            "lexical_item": "",
            **attrs,
            "compound": "",
            "ambiguous": "",
        }, inherited_hand

    blocking_column, blocking_value = extract_blocking_marker(text)
    if blocking_column:
        add_unique(attrs, blocking_column, blocking_value)
        return {
            "annotation": whole_annotation,
            "lexical_item": "",
            **attrs,
            "compound": "",
            "ambiguous": "",
        }, inherited_hand

    text, hand_from_parentheses = parse_parenthetical_hand(text)
    if hand_from_parentheses:
        inherited_hand = hand_from_parentheses

    if contains_unknown_keyword(text):
        return make_empty_output_row(whole_annotation, ambiguous=whole_annotation), inherited_hand

    text, active_fs = parse_prefix_chain(text)
    active_hand = inherited_hand

    text = remove_pt_objects(text)

    text = parse_nmm(text, attrs)
    text = parse_stop_hold_keep_index(text, attrs)
    text = parse_qm(text, attrs)
    text = parse_past_neg(text, attrs)
    text = parse_mouth_notes_early(text, attrs)
    text, found_fal = parse_d_fal_un(text, attrs, "FAL")
    text, found_un = parse_d_fal_un(text, attrs, "UN")
    text, found_d = parse_d_fal_un(text, attrs, "D")

    if found_fal or found_un or found_d:
        # FAL / UN / D are blocking labels. Once any of them is found, no other
        # parsed column and no lexical_item may be filled.
        fal_value = attrs.get("fal", "")
        un_value = attrs.get("un", "")
        d_value = attrs.get("d", "")

        attrs = empty_attrs()
        attrs["fal"] = fal_value
        attrs["un"] = un_value
        attrs["d"] = d_value

        return {
            "annotation": whole_annotation,
            "lexical_item": "",
            **attrs,
            "compound": "",
            "ambiguous": "",
        }, inherited_hand

    blocked_lexical_item = False

    text = parse_embedded_parenthetical_hand_streams(text, attrs)

    text, hand_from_marker = parse_hand_markers(text, attrs)
    if hand_from_marker:
        active_hand = hand_from_marker

    text = parse_fs(text, attrs)
    text = parse_aw(text, attrs)
    text = parse_dw(text, attrs)

    text = parse_pt(text, attrs, active_hand=active_hand)
    if attrs.get("_last_hand_from_pt_object") in HAND_COLUMNS:
        active_hand = attrs.get("_last_hand_from_pt_object")
    rep_word_before_cleanup = infer_repeated_word(text)
    text = parse_rep(text, attrs, active_hand=active_hand, repeated_word=rep_word_before_cleanup)
    text = parse_cl(text, attrs)
    text = parse_m(text, attrs)
    text = parse_marker_with_value(text, attrs, "GES")

    lexical_item = "" if blocked_lexical_item else cleanup_lexical_item(text)

    if attrs.get("_force_clean_pt") and not blocked_lexical_item:
        current_parts = [p.strip() for p in str(lexical_item or "").split(";") if p.strip()]
        if "pt" not in current_parts:
            lexical_item = "pt" + (";" + lexical_item if lexical_item else "")

    if attrs.get("_force_clean_pt_number") and not blocked_lexical_item:
        pt_clean = "pt" + str(attrs.get("_force_clean_pt_number"))
        current_parts = [p.strip() for p in str(lexical_item or "").split(";") if p.strip()]
        if pt_clean not in current_parts:
            lexical_item = pt_clean + (";" + lexical_item if lexical_item else "")

    # If the whole annotation was explicit hand streams only, keep the hand
    # values in lexical_item. Do not do this for pure repetition rows such as
    # RH:男＊２.
    if not lexical_item and not attrs.get("rep") and not blocked_lexical_item:
        hand_values: List[str] = []
        for hand_key in ("rh", "lh"):
            for value in str(attrs.get(hand_key, "") or "").split(";"):
                for clean_value in hand_value_to_lexical_component(value).split(";"):
                    clean_value = cleanup_lexical_item(clean_value)
                    if clean_value and clean_value not in hand_values:
                        hand_values.append(clean_value)
        if hand_values:
            lexical_item = ";".join(hand_values)

    # A pointing annotation IS the pointing sign. Whatever Japanese material the
    # annotator wrote alongside it names the referent being pointed at, not a
    # signed word, so it must not become lexical material and inflate lexical
    # frequency counts. The leftover text of a PT construction is therefore
    # replaced by "pt" + number:
    #
    #   PT1＝歯     -> lexical_item = pt1   (歯 is the referent, dropped)
    #   pt2:ひよこ  -> lexical_item = pt2
    #   PT(アニメ)  -> lexical_item = pt
    #   狙う(pt)    -> lexical_item = pt
    #
    # This happens before the marker values are collected below, so a pointing
    # annotation that also carries a real sign keeps both:
    #   pt3(dw:5種類) -> lexical_item = pt3;5種類
    if not blocked_lexical_item:
        pt_lexical_item = pt_column_to_lexical_item(attrs.get("pt", ""))
        if pt_lexical_item:
            # Substituting "ptN" would otherwise hide anything the parser failed
            # to account for, because "ptN" is allowed Latin and passes the
            # residue check. Inspect the material being dropped first: if it
            # still contains unparsed non-Japanese text, the annotation was not
            # understood and must go to manual review regardless.
            discarded = ";".join(split_lexical_components(lexical_item))
            discarded = re.sub(r"(?i)(^|;)\s*pt[0-9]*\s*(?=;|$)", ";", discarded)
            if discarded and has_non_japanese_residue(discarded):
                attrs["_pt_discarded_residue"] = "TRUE"
            lexical_item = pt_lexical_item

    # What the FS: prefix chain actually fingerspelled: the leftover text on its
    # own, before the values of other markers are merged in below. Without this
    # snapshot, fs:お(M:ロースポーク) would copy the merged "お;ロースポーク" into
    # the fs column, putting the mouthing into fingerspelling.
    fingerspelled_material = lexical_item

    # A marker whose value stays in the text (GES, CL) would otherwise ride
    # along into fs: fs:\u304a(ges:\u3042) must give fs=\u304a, not fs=\u304a;\u3042. Values another
    # column already owns are therefore removed from the snapshot. This
    # generalises the M case above to every marker that leaves its value behind.
    owned_elsewhere = marker_value_components(attrs, ("ges", "cl", "m", "dw", "aw", "nmm"))
    if owned_elsewhere and fingerspelled_material:
        kept = [
            part for part in re.split(r"[;\s]+", fingerspelled_material)
            if cleanup_lexical_item(part) and cleanup_lexical_item(part) not in owned_elsewhere
        ]
        fingerspelled_material = ";".join(cleanup_lexical_item(p) for p in kept)

    # Collect the lexical material recorded in the marker columns, so that
    # lexical_item is the full list of signs in the annotation, separated by ";".
    #
    #   二つ目:LH:人差し指   -> lexical_item = 二つ目;人差し指   (lh = 人差し指)
    #   cl:人型:(猫)(ges:あ) -> lexical_item = 人型;猫;あ        (cl = 人型, ges = あ)
    #
    # NMM and REP are deliberately excluded: an eyebrow raise is not a signed
    # word, and rep values are formatted as word(count;hand) rather than plain
    # lexical material.
    for extra_key in ("fs", "m", "dw", "cl", "aw", "ges", "lh", "rh"):
        for extra_value in str(attrs.get(extra_key, "") or "").split(";"):
            extra_value = cleanup_lexical_item(extra_value)
            # Mouth values such as un(きった) are stored in m/un but must not
            # become lexical_item material, because the Latin UN marker would make
            # the row look ambiguous. Plain Japanese mouth values such as おる
            # are still allowed into lexical_item.
            if extra_key == "m" and (
                re.search(r"(?i)(^|[^A-Za-z])(UN|NEG|NOD)([^A-Za-z]|$)", extra_value)
                or has_non_japanese_residue(extra_value)
            ):
                continue
            if extra_value and not re.fullmatch(r"(?i)TRUE", extra_value):
                if lexical_item:
                    current_parts = [p.strip() for p in lexical_item.split(";") if p.strip()]
                    if extra_value not in current_parts:
                        lexical_item += ";" + extra_value
                else:
                    lexical_item = extra_value

    if lexical_item:
        lexical_item = strip_hand_note_from_lexical_item_and_fill_attrs(lexical_item, attrs)
        add_hand_from_lexical_item_parentheses(lexical_item, attrs)
        # Clean each component on its own. Splitting a marker value out of the
        # text can leave the material beside it holding one half of a bracket
        # pair (ges:\u3057\u3073\u308c\u308b(\u732b) leaves "(\u732b))") and cleanup_lexical_item only
        # balances brackets across the whole string, so it cannot see that.
        lexical_item = _normalise_lexical_components(lexical_item)

    if active_fs and lexical_item:
        # Only the fingerspelled material goes to fs / to the active hand;
        # values belonging to M, GES, CL and friends stay in their own columns.
        fs_material = cleanup_lexical_item(fingerspelled_material) or lexical_item
        add_unique(attrs, "fs", fs_material)
        if active_hand in HAND_COLUMNS:
            add_unique(attrs, active_hand, fs_material)

    elif active_hand in HAND_COLUMNS and lexical_item and not attrs.get(active_hand):
        add_unique(attrs, active_hand, lexical_item)

    if hand_from_explanation in HAND_COLUMNS and lexical_item:
        base_clean = cleanup_lexical_item(
            re.sub(r"[\(（][^\)）]*(右手|左手|右|左)[^\)）]*[\)）]", "", lexical_item)
        )
        add_unique(attrs, hand_from_explanation, base_clean if base_clean else lexical_item)

    if active_hand in HAND_COLUMNS and attrs.get("pt") and not lexical_item and not attrs[active_hand]:
        add_unique(attrs, active_hand, "pt")

    row = {
        "annotation": whole_annotation,
        "lexical_item": "",
        **attrs,
        "compound": "",
        "ambiguous": "",
    }

    for component in split_lexical_components(lexical_item):
        add_lexical_component(row, component)

    if annotation_has_known_unknown(whole_annotation):
        row["ambiguous"] = whole_annotation

    elif (
        row.get("lexical_item")
        and (
            has_non_japanese_residue(row["lexical_item"])
            or attrs.get("_pt_discarded_residue")
        )
        and not is_not_ambiguous_exception(whole_annotation)
    ):
        row["ambiguous"] = whole_annotation
        row["lexical_item"] = ""

    # cl:dw / cl:dr are valid marker-only rows even though the CL value is Latin.
    if row.get("cl") == "dw" and row.get("dw") == "TRUE" and not row.get("lexical_item"):
        row["ambiguous"] = ""

    return row, active_hand


def parse_annotation_text(
    annotation_text: str,
    whole_annotation: str,
    inherited_hand: Optional[str] = None,
) -> Dict[str, str]:
    output = make_empty_output_row(whole_annotation)
    current_hand = inherited_hand

    for segment in split_plus_streams(annotation_text):
        parsed, hand_after = parse_segment(segment, whole_annotation, current_hand)
        merge_rows(output, parsed)

        if hand_after in HAND_COLUMNS:
            current_hand = hand_after

    output["lexical_item"] = cleanup_lexical_item(output.get("lexical_item", ""))

    # A prolonged sound mark by itself is not a confidently parsed lexical word.
    lexical_item_without_separators = re.sub(r"[;\s]+", "", output["lexical_item"])
    if lexical_item_without_separators and re.fullmatch(r"ー+", lexical_item_without_separators):
        output["ambiguous"] = whole_annotation

    return output


def parse_plain_annotation(annotation: str) -> List[Dict[str, str]]:
    if is_not_ambiguous_exception(annotation):
        normalized_annotation = normalize_annotation(annotation)
        return [parse_annotation_text(normalized_annotation, annotation)]

    # Leading / or // are ELAN/annotation-prefix marks, not separators.
    # Only internal repeated slashes trigger the ambiguity rule.
    slash_check_annotation = strip_leading_annotation_slashes(annotation)
    # Trailing slash marks like pt// or cl:.../ are suffix marks, not separators.
    slash_check_annotation = re.sub(r"/+$", "", slash_check_annotation).strip()
    if has_more_than_one_slash(slash_check_annotation) and not is_simple_slash_wrapped_annotation(slash_check_annotation):
        row = make_empty_output_row(annotation, ambiguous=annotation)
        row["_skip_ambiguous_file"] = True
        return [row]

    if has_compound_marker(annotation):
        row = make_empty_output_row(annotation)
        row["compound"] = annotation
        return [row]

    normalized_annotation = normalize_annotation(annotation)

    if contains_unknown_keyword(normalized_annotation):
        return [make_empty_output_row(annotation, ambiguous=annotation)]

    if not normalized_annotation:
        return [make_empty_output_row(annotation, ambiguous=annotation)]

    return [parse_annotation_text(normalized_annotation, annotation)]


def mark_known_unknown_flags(row: Dict[str, str]) -> None:
    annotation = normalize_annotation(row.get("annotation", ""))

    if re.search(r"(?i)FAL", annotation) and not row.get("fal"):
        row["fal"] = "FAL"
    if re.search(r"(?i)UN", annotation) and not row.get("un"):
        row["un"] = "UN"
    if re.search(r"(?i)(^|[^A-Za-z])D([^A-Za-z]|$)", annotation) and not row.get("d"):
        row["d"] = "D"

    # FAL / UN / D are blocking markers. If one exists, do not add QM afterwards.
    if row.get("fal") or row.get("un") or row.get("d"):
        return

    if (
        "?" in annotation
        or "？" in annotation
        or re.search(r"(?i)(^|[^A-Za-z])QM([^A-Za-z]|$)", annotation)
    ) and not row.get("qm"):
        row["qm"] = "TRUE"


def clear_parsed_values_for_ambiguous(row: Dict[str, str]) -> None:
    """If a row is ambiguous, keep only the annotation and the ambiguous value."""
    if not row.get("ambiguous"):
        return
    row["lexical_item"] = ""
    row["compound"] = ""
    for key in ATTR_COLUMNS:
        row[key] = ""


def apply_blocking_marker_cleanup(row: Dict[str, str]) -> None:
    """FAL / UN / D block every other parsed column."""
    if not (row.get("fal") or row.get("un") or row.get("d")):
        return

    row["lexical_item"] = ""
    for column in [
        "pt", "dw", "fs", "aw", "lh", "rh", "cl", "m",
        "ges", "nmm", "rep", "stop", "hold", "index",
        "keep", "qm", "past", "neg",
    ]:
        row[column] = ""


def collect_compound_group(rows: Sequence[Dict[str, str]], start_index: int) -> Tuple[List[Dict[str, str]], int]:
    group = [rows[start_index]]
    i = start_index + 1

    while i < len(rows):
        group.append(rows[i])
        annotation = str(rows[i].get("annotation", "") or "")
        i += 1
        if has_compound_close(annotation):
            break

    return group, i


def strip_internal_fields(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop helper keys beginning with ``_`` so debug flags never reach the CSV."""
    return [{key: row.get(key, "") for key in FIELDNAMES} for row in rows]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_annotation(annotation: str) -> Dict[str, str]:
    """Parse a single annotation string into a fully finished parsed row.

    This is the function used by the tests and by the worked examples in the
    LaTeX documentation.
    """
    rows = parse_rows([{"annotation": annotation}])
    return rows[0] if rows else make_empty_output_row(annotation)


def parse_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Parse one file's worth of extracted annotation rows.

    ``rows`` are dictionaries as produced by :mod:`elan_pipeline.extract` (or
    read from a ``*_word_annotations.csv``). The returned rows follow
    :data:`elan_pipeline.config.FIELDNAMES`.
    """
    parsed_rows: List[Dict[str, str]] = []

    index = 0
    while index < len(rows):
        input_row = rows[index]
        annotation = str(input_row.get("annotation", "") or "").strip()
        metadata = extract_metadata(input_row)

        if not annotation:
            index += 1
            continue

        if has_compound_open(annotation):
            group, next_index = collect_compound_group(rows, index)

            for group_row in group:
                original = str(group_row.get("annotation", "") or "").strip()
                row = make_empty_output_row(original)
                row["compound"] = original
                attach_metadata(row, extract_metadata(group_row))
                parsed_rows.append(row)

            index = next_index
            continue

        for parsed_row in parse_plain_annotation(annotation):
            attach_metadata(parsed_row, metadata)
            parsed_rows.append(parsed_row)

        index += 1

    for row in parsed_rows:
        row["lexical_item"] = cleanup_lexical_item(row.get("lexical_item", ""))
        mark_known_unknown_flags(row)
        clear_parsed_values_for_ambiguous(row)
        apply_blocking_marker_cleanup(row)

    return strip_internal_fields(parsed_rows)


def select_ambiguous_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return only the rows that need manual review."""
    return [row for row in rows if row.get("ambiguous")]


def parse_statistics(rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    """Summarise a parsed file: totals, resolved rows and percentage."""
    total = len(rows)
    ambiguous = len(select_ambiguous_rows(rows))
    parsed = total - ambiguous

    return {
        "total_rows": total,
        "parsed_rows": parsed,
        "ambiguous_rows": ambiguous,
        "parsed_percentage": (parsed / total * 100) if total else 0.0,
    }
