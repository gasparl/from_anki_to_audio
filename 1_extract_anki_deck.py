#!/usr/bin/env python3
"""
ANKI CONTENT EXTRACTOR V3 - FULL OR DIFFICULT-CARD MODE

Reads an Anki .apkg deck and creates one compact JSON file containing only
the material needed for the later AI sentence-generation step.

Set ``SELECTION`` in the user settings below:

* False preserves the original workflow and extracts every Anki note once.
* True ranks individual cards/directions from their review histories, selects
  the hardest subset, and repeats the hardest cards at the end of the output.

Output fields per note:
  id                  = simple sequential number
  source_note_id      = original Anki note ID for traceability
  japanese            = Japanese phrase / grammar point
  english             = English meaning
  explanation         = explanation/grammar note used as private AI context
  example_japanese    = existing Japanese example, if present
  example_english     = existing English example, if present

The explanation is retained because it often identifies the exact grammar or
nuance the learner intended to study. It is passed to the AI as reference
context only and is never spoken in the audiobook.

In full mode, the script extracts Anki NOTES rather than generated CARDS. In
selected mode, JP-to-EN and EN-to-JP cards are ranked independently. Their
source-note content is still written in the same downstream-compatible shape.

For modern Anki packages, install the only non-standard dependency with:
    pip install zstandard
"""

import html
import json
import math
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


SCRIPT_VERSION = "3.1-FULL-OR-SELECTED"


# ===================== SIMPLE USER SETTINGS =====================

# False = original full-note extraction.
# True  = select difficult individual cards/directions from review history.
SELECTION = True

# Put the APKG in the same folder as this script, or enter an absolute path.
INPUT_FILE = "Base.apkg"
OUTPUT_ROOT_DIR_NAME = "anki_audio_output_v3"
OUTPUT_FILE = "anki_content_v3.json"

# ---------------- Settings used only when SELECTION = True ----------------

# Total rows written, INCLUDING repeat copies.
OUTPUT_COUNT = 300

# The hardest N individual cards are included twice. With the defaults, the
# first 60 rows are their first instances and the last 60 rows are copies;
# between them are ranks 61-240. Total: 240 unique cards + 60 copies = 300.
REPEAT_TOP_CARDS = 60

# Reviews lose half their recency weight after this many days. A smaller value
# focuses more sharply on recent study; a larger value remembers longer.
RECENCY_HALF_LIFE_DAYS = 90.0

# Old reviews retain this fraction of their weight in mature-lapse and Hard
# signals. The explicitly recent-Again signal always decays toward zero.
OLD_REVIEW_WEIGHT_FLOOR = 0.05

# Equivalent number of average deck reviews mixed into rate estimates.
# Increase this to be more conservative about cards with little history.
SMOOTHING_PRIOR_REVIEWS = 12.0

# Cards with fewer recorded answer-button reviews are not ranked.
MIN_REVIEWS_PER_CARD = 3

# Anki defines a mature card as one with an interval of at least 21 days.
MATURE_INTERVAL_DAYS = 21

# Relative score weights. They need not add to 1; the script normalizes them.
WEIGHT_RECENT_AGAIN = 0.50
WEIGHT_LIFETIME_AGAIN_RATE = 0.25
WEIGHT_MATURE_LAPSES = 0.15
WEIGHT_HARD_PRESSES = 0.10

# Include suspended cards. True can recover difficult leeches; use False if
# suspended cards were intentionally removed from study.
INCLUDE_SUSPENDED_CARDS = True

# True: resolve relative paths beside this script.
# False: resolve relative paths from Spyder's current working directory.
USE_SCRIPT_FOLDER = True

# Field names used by the attached deck. Change these if another deck uses
# different names. Matching is case-insensitive.
JAPANESE_FIELD = "phrase"
ENGLISH_FIELD = "translation"
EXPLANATION_FIELD = "explanation"
EXAMPLE_JAPANESE_FIELD = "example use"
EXAMPLE_ENGLISH_FIELD = "example translation"

# note_id follows the deck's original note/creation order.
# card_due follows the earliest generated card's Anki due position.
SORT_ORDER = "note_id"  # "note_id" or "card_due"

# Text cleaning
REMOVE_FURIGANA = True
COMPACT_JAPANESE_SPACES = True

# Console output
PREVIEW_NOTES = 3
PROGRESS_EVERY = 250

# ===================== ADVANCED SETTINGS =====================

# These convert raw event counts to bounded 0..1 score components.
RECENT_AGAIN_SATURATION = 2.0
MATURE_LAPSE_SATURATION = 2.0

# Selected-mode console preview.
PREVIEW_SELECTED_CARDS = 8

# =============================================================


FIELD_SEPARATOR = "\x1f"
REVIEW_TYPE = 1
SUSPENDED_QUEUE = -1


def get_base_dir() -> Path:
    """Return the folder used for relative input and output paths."""
    if USE_SCRIPT_FOLDER and "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


def resolve_input_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return get_base_dir() / path


def get_output_root() -> Path:
    """Return the single folder used by all three V3 pipeline scripts."""
    return get_base_dir() / OUTPUT_ROOT_DIR_NAME


def resolve_output_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return get_output_root() / path


class SimpleHTMLExtractor(HTMLParser):
    """Convert Anki field HTML into readable plain text."""

    BLOCK_TAGS = {
        "blockquote", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "ol", "p", "table", "tr", "ul",
    }
    SKIP_TAGS = {"script", "style", "rt", "rp"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()

        if self.skip_depth:
            self.skip_depth += 1
            return

        if tag in self.SKIP_TAGS:
            self.skip_depth = 1
            return

        if tag == "br":
            self.parts.append("\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n")

        # Keep image filenames visible if a future deck contains useful images.
        if tag == "img":
            attributes = dict(attrs)
            source = (attributes.get("src") or "").strip()
            if source:
                self.parts.append(f"[image: {source}]")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if self.skip_depth:
            self.skip_depth -= 1
            return

        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def get_text(self) -> str:
        return "".join(self.parts)


# Examples handled:
#   気[き]に入る     -> 気に入る
#   できる[できる]   -> できる
FURIGANA_RE = re.compile(
    r"(?<=[\u3040-\u30ff\u3400-\u9fff々〆ヶヵ])"
    r"\[[ぁ-ゖァ-ヺー・\s]+\]"
)

JAPANESE_CHARACTERS = (
    r"\u3000-\u303f\u3040-\u30ff\u31f0-\u31ff"
    r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff々〆ヶヵ"
)

JAPANESE_SPACE_RE = re.compile(
    rf"(?<=[{JAPANESE_CHARACTERS}])\s+(?=[{JAPANESE_CHARACTERS}])"
)

SOUND_RE = re.compile(r"\[sound:[^\]]+\]", flags=re.IGNORECASE)


def clean_text(raw_text: str) -> str:
    """Remove HTML and normalize whitespace without using AI."""
    if not raw_text:
        return ""

    raw_text = SOUND_RE.sub("", str(raw_text))

    parser = SimpleHTMLExtractor()
    try:
        parser.feed(raw_text)
        parser.close()
        text = parser.get_text()
    except Exception:
        # Anki fields can contain imperfect HTML. Preserve readable content
        # instead of discarding the field if parsing fails.
        text = raw_text

    text = html.unescape(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines).strip()

    if REMOVE_FURIGANA:
        text = FURIGANA_RE.sub("", text)

    if COMPACT_JAPANESE_SPACES:
        previous = None
        while previous != text:
            previous = text
            text = JAPANESE_SPACE_RE.sub("", text)

    return text.strip()


def safe_field_name(name: str) -> str:
    """Normalize field names for case-insensitive matching."""
    return re.sub(r"\s+", " ", str(name).strip().casefold())


def register_anki_collation(connection: sqlite3.Connection) -> None:
    """Register the collation used by newer Anki database indexes."""

    def unicase(left: str, right: str) -> int:
        left_value = (left or "").casefold()
        right_value = (right or "").casefold()
        return (left_value > right_value) - (left_value < right_value)

    connection.create_collation("unicase", unicase)


def get_table_names(connection: sqlite3.Connection) -> set:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )
    return {row[0] for row in rows}


def load_note_type_fields(
    connection: sqlite3.Connection,
) -> Tuple[Dict[int, str], Dict[int, List[str]], str]:
    """Read note-type and field names from modern or older Anki schemas."""
    tables = get_table_names(connection)
    note_type_names: Dict[int, str] = {}
    fields_by_note_type: Dict[int, List[str]] = {}

    if {"notetypes", "fields"}.issubset(tables):
        schema_style = "modern normalized schema"

        for row in connection.execute("SELECT id, name FROM notetypes"):
            note_type_names[int(row["id"])] = row["name"]

        rows = connection.execute(
            "SELECT ntid, ord, name FROM fields ORDER BY ntid, ord"
        )
        for row in rows:
            note_type_id = int(row["ntid"])
            field_number = int(row["ord"])
            field_names = fields_by_note_type.setdefault(note_type_id, [])

            while len(field_names) <= field_number:
                field_names.append("")

            field_names[field_number] = row["name"]

    else:
        schema_style = "older JSON schema"

        row = connection.execute("SELECT models FROM col LIMIT 1").fetchone()
        if not row:
            raise ValueError("No Anki note-type information was found")

        models = json.loads(row["models"] or "{}")

        for model_key, model in models.items():
            note_type_id = int(model.get("id", model_key))
            note_type_names[note_type_id] = model.get(
                "name", f"Note type {note_type_id}"
            )

            fields = sorted(
                model.get("flds", []),
                key=lambda item: int(item.get("ord", 0)),
            )
            fields_by_note_type[note_type_id] = [
                field.get("name", "") for field in fields
            ]

    return note_type_names, fields_by_note_type, schema_style


def load_template_names(
    connection: sqlite3.Connection,
) -> Dict[Tuple[int, int], str]:
    """Map each (note type, card ordinal) to its direction/template name."""
    tables = get_table_names(connection)
    template_names: Dict[Tuple[int, int], str] = {}

    if "templates" in tables:
        rows = connection.execute(
            "SELECT ntid, ord, name FROM templates ORDER BY ntid, ord"
        )
        for row in rows:
            key = (int(row["ntid"]), int(row["ord"]))
            template_names[key] = row["name"]
        return template_names

    row = connection.execute("SELECT models FROM col LIMIT 1").fetchone()
    if not row:
        return template_names

    models = json.loads(row["models"] or "{}")
    for model_key, model in models.items():
        note_type_id = int(model.get("id", model_key))
        for template in model.get("tmpls", []):
            ordinal = int(template.get("ord", 0))
            template_names[(note_type_id, ordinal)] = template.get(
                "name", f"Card {ordinal + 1}"
            )
    return template_names


def extract_collection_database(
    archive: zipfile.ZipFile,
    database_path: Path,
) -> str:
    """Extract the best available collection database from the APKG."""
    archive_members = set(archive.namelist())

    for member_name in (
        "collection.anki21b",
        "collection.anki21",
        "collection.anki2",
    ):
        if member_name not in archive_members:
            continue

        with archive.open(member_name) as source:
            with database_path.open("wb") as destination:
                if member_name.endswith(".anki21b"):
                    try:
                        import zstandard
                    except ImportError as error:
                        raise ImportError(
                            "This deck uses Anki's compressed database format. "
                            "Install it once with: pip install zstandard"
                        ) from error

                    decompressor = zstandard.ZstdDecompressor()
                    with decompressor.stream_reader(source) as reader:
                        shutil.copyfileobj(reader, destination)
                else:
                    shutil.copyfileobj(source, destination)

        return member_name

    raise ValueError("No supported Anki collection database was found")


def get_note_rows(connection: sqlite3.Connection):
    """Return notes in the configured deterministic order."""
    if SORT_ORDER == "note_id":
        query = """
            SELECT id, mid, flds
            FROM notes
            ORDER BY id
        """
    elif SORT_ORDER == "card_due":
        query = """
            SELECT n.id, n.mid, n.flds, MIN(c.due) AS first_due
            FROM notes AS n
            LEFT JOIN cards AS c ON c.nid = n.id
            GROUP BY n.id, n.mid, n.flds
            ORDER BY first_due, n.id
        """
    else:
        raise ValueError(
            f"Unknown SORT_ORDER: {SORT_ORDER}. Use 'note_id' or 'card_due'."
        )

    return list(connection.execute(query))


def get_field(
    fields: Dict[str, str],
    configured_name: str,
) -> str:
    """Retrieve one configured field using case-insensitive matching."""
    wanted = safe_field_name(configured_name)

    for field_name, field_value in fields.items():
        if safe_field_name(field_name) == wanted:
            return clean_text(field_value)

    return ""


def extract_note_content(
    row: sqlite3.Row,
    fields_by_note_type: Dict[int, List[str]],
) -> Dict[str, str]:
    """Extract and clean the shared content fields from one note row."""
    note_type_id = int(row["mid"])
    field_names = fields_by_note_type.get(note_type_id, [])
    field_values = (row["flds"] or "").split(FIELD_SEPARATOR)

    # Fill missing stored fields with empty strings. Unexpected extras are
    # deliberately ignored because this is a content-only export.
    if len(field_values) < len(field_names):
        field_values.extend(
            "" for _ in range(len(field_names) - len(field_values))
        )

    fields = {
        field_name: field_value
        for field_name, field_value in zip(field_names, field_values)
    }
    return {
        "japanese": get_field(fields, JAPANESE_FIELD),
        "english": get_field(fields, ENGLISH_FIELD),
        "explanation": get_field(fields, EXPLANATION_FIELD),
        "example_japanese": get_field(fields, EXAMPLE_JAPANESE_FIELD),
        "example_english": get_field(fields, EXAMPLE_ENGLISH_FIELD),
    }


def show_note_type_info(
    note_type_names: Dict[int, str],
    fields_by_note_type: Dict[int, List[str]],
    note_type_counts: Counter,
) -> None:
    """Print the relevant Anki schema in a Spyder-friendly format."""
    print("\nNOTE TYPES AND FIELDS")
    print("-" * 50)

    for note_type_id, count in note_type_counts.items():
        note_type_name = note_type_names.get(
            note_type_id, f"Unknown note type {note_type_id}"
        )
        field_names = fields_by_note_type.get(note_type_id, [])

        print(f"{note_type_name}: {count} notes")
        print(f"  Fields: {', '.join(field_names)}")


def warn_missing_configured_fields(
    note_type_names: Dict[int, str],
    fields_by_note_type: Dict[int, List[str]],
    note_type_counts: Counter,
) -> None:
    """Warn in either mode when configured content fields are unavailable."""
    configured_fields = [
        JAPANESE_FIELD,
        ENGLISH_FIELD,
        EXPLANATION_FIELD,
        EXAMPLE_JAPANESE_FIELD,
        EXAMPLE_ENGLISH_FIELD,
    ]
    for note_type_id in note_type_counts:
        available = {
            safe_field_name(name)
            for name in fields_by_note_type.get(note_type_id, [])
        }
        missing = [
            name
            for name in configured_fields
            if safe_field_name(name) not in available
        ]
        if missing:
            note_type_name = note_type_names.get(
                note_type_id, str(note_type_id)
            )
            print(
                f"WARNING: {note_type_name} is missing configured fields: "
                f"{', '.join(missing)}"
            )


def extract_notes(
    connection: sqlite3.Connection,
    note_type_names: Dict[int, str],
    fields_by_note_type: Dict[int, List[str]],
) -> List[Dict[str, object]]:
    """Create the minimal content records used by the future AI script."""
    rows = get_note_rows(connection)
    note_type_counts = Counter(int(row["mid"]) for row in rows)

    show_note_type_info(
        note_type_names,
        fields_by_note_type,
        note_type_counts,
    )
    warn_missing_configured_fields(
        note_type_names,
        fields_by_note_type,
        note_type_counts,
    )

    print("\nEXTRACTING CONTENT")
    print("-" * 50)

    extracted: List[Dict[str, object]] = []

    for index, row in enumerate(rows, 1):
        note_id = int(row["id"])
        extracted.append(
            {
                "id": index,
                "source_note_id": note_id,
                **extract_note_content(row, fields_by_note_type),
            }
        )

        if PROGRESS_EVERY and index % PROGRESS_EVERY == 0:
            print(f"  Extracted {index:,}/{len(rows):,} notes...")

    return extracted


def recency_weight(
    review_ms: int,
    reference_ms: int,
    floor: float = 0.0,
) -> float:
    """Return exponential half-life weight for a review timestamp."""
    age_days = max(0.0, (reference_ms - review_ms) / 86_400_000.0)
    decayed = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    return floor + (1.0 - floor) * decayed


def saturating_count(value: float, scale: float) -> float:
    """Map a non-negative event count smoothly onto 0..1."""
    return 1.0 - math.exp(-max(0.0, value) / scale)


def smoothed_rate(
    successes: float,
    observations: float,
    global_rate: float,
) -> float:
    """Estimate a rate using the deck average as an empirical beta prior."""
    numerator = successes + SMOOTHING_PRIOR_REVIEWS * global_rate
    denominator = observations + SMOOTHING_PRIOR_REVIEWS
    if denominator <= 0:
        return global_rate
    return numerator / denominator


def score_cards(
    connection: sqlite3.Connection,
    fields_by_note_type: Dict[int, List[str]],
    template_names: Dict[Tuple[int, int], str],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Score every eligible card/direction from its own review history."""
    histories: Dict[int, List[sqlite3.Row]] = defaultdict(list)
    all_reviews = list(
        connection.execute(
            """
            SELECT cid, id, ease, lastIvl, type
            FROM revlog
            WHERE ease BETWEEN 1 AND 4
            ORDER BY cid, id
            """
        )
    )
    for review in all_reviews:
        histories[int(review["cid"])].append(review)

    total_reviews = len(all_reviews)
    total_again = sum(int(row["ease"]) == 1 for row in all_reviews)
    total_hard = sum(int(row["ease"]) == 2 for row in all_reviews)
    global_again_rate = total_again / total_reviews if total_reviews else 0.0
    global_hard_rate = total_hard / total_reviews if total_reviews else 0.0

    now_ms = int(time.time() * 1000)
    latest_review_ms = max(
        (int(row["id"]) for row in all_reviews),
        default=now_ms,
    )
    # Protect against a device clock that placed the newest review in the
    # future, while still allowing an old export to become less recent.
    reference_ms = max(now_ms, latest_review_ms)

    raw_weights = {
        "recent_again": float(WEIGHT_RECENT_AGAIN),
        "lifetime_again_rate": float(WEIGHT_LIFETIME_AGAIN_RATE),
        "mature_lapses": float(WEIGHT_MATURE_LAPSES),
        "hard_presses": float(WEIGHT_HARD_PRESSES),
    }
    weight_total = sum(raw_weights.values())
    normalized_weights = {
        name: value / weight_total for name, value in raw_weights.items()
    }

    query = """
        SELECT
            c.id AS card_id,
            c.nid AS note_id,
            c.ord AS card_ordinal,
            c.queue AS card_queue,
            n.mid AS mid,
            n.flds AS flds
        FROM cards AS c
        JOIN notes AS n ON n.id = c.nid
        ORDER BY c.id
    """
    scored: List[Dict[str, object]] = []
    excluded = Counter()

    for row in connection.execute(query):
        card_id = int(row["card_id"])
        reviews = histories.get(card_id, [])
        if len(reviews) < MIN_REVIEWS_PER_CARD:
            excluded["too_few_reviews"] += 1
            continue
        if (
            not INCLUDE_SUSPENDED_CARDS
            and int(row["card_queue"]) == SUSPENDED_QUEUE
        ):
            excluded["suspended"] += 1
            continue

        content = extract_note_content(row, fields_by_note_type)
        if not content["japanese"] or not content["english"]:
            excluded["missing_required_content"] += 1
            continue

        again_count = 0
        hard_count = 0
        mature_lapse_count = 0
        recent_again_count = 0.0
        weighted_hard_count = 0.0
        weighted_review_count = 0.0
        weighted_mature_lapses = 0.0

        for review in reviews:
            review_ms = int(review["id"])
            ease = int(review["ease"])
            pure_recency = recency_weight(review_ms, reference_ms)
            retained_recency = recency_weight(
                review_ms,
                reference_ms,
                OLD_REVIEW_WEIGHT_FLOOR,
            )
            weighted_review_count += retained_recency

            if ease == 1:
                again_count += 1
                recent_again_count += pure_recency
                if (
                    int(review["type"]) == REVIEW_TYPE
                    and int(review["lastIvl"]) >= MATURE_INTERVAL_DAYS
                ):
                    mature_lapse_count += 1
                    weighted_mature_lapses += retained_recency
            elif ease == 2:
                hard_count += 1
                weighted_hard_count += retained_recency

        components = {
            "recent_again": saturating_count(
                recent_again_count,
                RECENT_AGAIN_SATURATION,
            ),
            "lifetime_again_rate": smoothed_rate(
                again_count,
                len(reviews),
                global_again_rate,
            ),
            "mature_lapses": saturating_count(
                weighted_mature_lapses,
                MATURE_LAPSE_SATURATION,
            ),
            "hard_presses": smoothed_rate(
                weighted_hard_count,
                weighted_review_count,
                global_hard_rate,
            ),
        }
        difficulty_score = sum(
            normalized_weights[name] * components[name]
            for name in normalized_weights
        )

        note_type_id = int(row["mid"])
        card_ordinal = int(row["card_ordinal"])
        template_name = template_names.get(
            (note_type_id, card_ordinal),
            f"Card {card_ordinal + 1}",
        )
        scored.append(
            {
                "source_card_id": card_id,
                "source_note_id": int(row["note_id"]),
                "card_ordinal": card_ordinal,
                "card_template": template_name,
                "difficulty_score": difficulty_score,
                "components": components,
                "review_count": len(reviews),
                "again_count": again_count,
                "hard_count": hard_count,
                "mature_lapse_count": mature_lapse_count,
                **content,
            }
        )

    scored.sort(
        key=lambda item: (
            -float(item["difficulty_score"]),
            -float(item["components"]["recent_again"]),
            -float(item["components"]["lifetime_again_rate"]),
            -float(item["components"]["mature_lapses"]),
            -float(item["components"]["hard_presses"]),
            int(item["source_card_id"]),
        )
    )
    for rank, card in enumerate(scored, 1):
        card["difficulty_rank"] = rank

    diagnostics = {
        "review_rows_used": total_reviews,
        "cards_scored": len(scored),
        "cards_excluded": dict(excluded),
        "global_again_rate": global_again_rate,
        "global_hard_rate": global_hard_rate,
        "reference_time_utc": datetime.fromtimestamp(
            reference_ms / 1000.0,
            tz=timezone.utc,
        ).isoformat(),
        "normalized_weights": normalized_weights,
    }
    return scored, diagnostics


def select_output_cards(
    ranked_cards: Sequence[Dict[str, object]],
) -> Tuple[List[Tuple[Dict[str, object], int]], int]:
    """Put first copies first and second copies of the hardest cards last."""
    unique_target = OUTPUT_COUNT - REPEAT_TOP_CARDS
    selected_unique = list(ranked_cards[:unique_target])
    repeated_count = min(REPEAT_TOP_CARDS, len(selected_unique))

    # This ordering intentionally places ranks 1..REPEAT_TOP_CARDS at the
    # beginning, and their second instances at the very end.
    selections = [(card, 1) for card in selected_unique]
    selections.extend((card, 2) for card in selected_unique[:repeated_count])
    return selections, repeated_count


def make_selected_records(
    selections: Sequence[Tuple[Dict[str, object], int]],
) -> List[Dict[str, object]]:
    """Create normal pipeline rows plus selection audit fields."""
    records: List[Dict[str, object]] = []
    for output_id, (card, copy_number) in enumerate(selections, 1):
        records.append(
            {
                # These seven fields are the unchanged V3 contract.
                "id": output_id,
                "source_note_id": int(card["source_note_id"]),
                "japanese": card["japanese"],
                "english": card["english"],
                "explanation": card["explanation"],
                "example_japanese": card["example_japanese"],
                "example_english": card["example_english"],
                # Stage 2 ignores these useful audit fields.
                "source_card_id": int(card["source_card_id"]),
                "card_ordinal": int(card["card_ordinal"]),
                "card_template": card["card_template"],
                "difficulty_rank": int(card["difficulty_rank"]),
                "difficulty_score": round(
                    float(card["difficulty_score"]), 8
                ),
                "output_copy": copy_number,
                "review_count": int(card["review_count"]),
                "again_count": int(card["again_count"]),
                "hard_count": int(card["hard_count"]),
                "mature_lapse_count": int(card["mature_lapse_count"]),
                "difficulty_components": {
                    name: round(float(value), 8)
                    for name, value in card["components"].items()
                },
            }
        )
    return records


def save_results(
    notes: List[Dict[str, object]],
    output_path: Path,
    input_path: Path,
    collection_member: str,
) -> None:
    """Save one readable JSON file for the next pipeline stage."""
    output = {
        "metadata": {
            "schema_version": 3,
            "source_file": input_path.name,
            "collection_member": collection_member,
            "total_notes": len(notes),
            "sort_order": SORT_ORDER,
            "content_fields": [
                "japanese",
                "english",
                "explanation",
                "example_japanese",
                "example_english",
            ],
        },
        "notes": notes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)


def save_selected_results(
    records: List[Dict[str, object]],
    output_path: Path,
    input_path: Path,
    collection_member: str,
    diagnostics: Dict[str, object],
    repeated_count: int,
) -> None:
    """Save selected records with transparent ranking metadata."""
    unique_cards = {int(row["source_card_id"]) for row in records}
    unique_notes = {int(row["source_note_id"]) for row in records}
    output = {
        "metadata": {
            "schema_version": 3,
            "extractor_version": SCRIPT_VERSION,
            "source_file": input_path.name,
            "collection_member": collection_member,
            "total_notes": len(records),
            "unique_cards": len(unique_cards),
            "unique_source_notes": len(unique_notes),
            "repeated_cards": repeated_count,
            "ranked_unit": "individual Anki card/direction",
            "output_order": (
                "unique cards in difficulty order, followed by second "
                "copies of the hardest cards"
            ),
            "selection_settings": {
                "selection": SELECTION,
                "output_count_including_repeats": OUTPUT_COUNT,
                "repeat_top_cards": REPEAT_TOP_CARDS,
                "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
                "old_review_weight_floor": OLD_REVIEW_WEIGHT_FLOOR,
                "smoothing_prior_reviews": SMOOTHING_PRIOR_REVIEWS,
                "minimum_reviews_per_card": MIN_REVIEWS_PER_CARD,
                "mature_interval_days": MATURE_INTERVAL_DAYS,
                "include_suspended_cards": INCLUDE_SUSPENDED_CARDS,
                "score_weights": diagnostics["normalized_weights"],
            },
            "selection_diagnostics": diagnostics,
            "content_fields": [
                "japanese",
                "english",
                "explanation",
                "example_japanese",
                "example_english",
            ],
        },
        "notes": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)


def validate_configuration() -> None:
    """Validate common settings and selected-mode settings when enabled."""
    if not isinstance(SELECTION, bool):
        raise ValueError("SELECTION must be True or False")
    if SORT_ORDER not in {"note_id", "card_due"}:
        raise ValueError(
            f"Unknown SORT_ORDER: {SORT_ORDER}. Use 'note_id' or 'card_due'."
        )
    if not SELECTION:
        return

    if isinstance(OUTPUT_COUNT, bool) or not isinstance(OUTPUT_COUNT, int):
        raise ValueError("OUTPUT_COUNT must be an integer")
    if OUTPUT_COUNT < 1:
        raise ValueError("OUTPUT_COUNT must be at least 1")
    if (
        isinstance(REPEAT_TOP_CARDS, bool)
        or not isinstance(REPEAT_TOP_CARDS, int)
    ):
        raise ValueError("REPEAT_TOP_CARDS must be an integer")
    if not 0 <= REPEAT_TOP_CARDS < OUTPUT_COUNT:
        raise ValueError(
            "REPEAT_TOP_CARDS must be between 0 and OUTPUT_COUNT - 1"
        )
    if RECENCY_HALF_LIFE_DAYS <= 0:
        raise ValueError("RECENCY_HALF_LIFE_DAYS must be greater than 0")
    if not 0 <= OLD_REVIEW_WEIGHT_FLOOR <= 1:
        raise ValueError("OLD_REVIEW_WEIGHT_FLOOR must be between 0 and 1")
    if SMOOTHING_PRIOR_REVIEWS < 0:
        raise ValueError("SMOOTHING_PRIOR_REVIEWS cannot be negative")
    if MIN_REVIEWS_PER_CARD < 1:
        raise ValueError("MIN_REVIEWS_PER_CARD must be at least 1")
    if MATURE_INTERVAL_DAYS < 1:
        raise ValueError("MATURE_INTERVAL_DAYS must be at least 1")
    if RECENT_AGAIN_SATURATION <= 0 or MATURE_LAPSE_SATURATION <= 0:
        raise ValueError("Saturation values must be greater than 0")
    weights = (
        WEIGHT_RECENT_AGAIN,
        WEIGHT_LIFETIME_AGAIN_RATE,
        WEIGHT_MATURE_LAPSES,
        WEIGHT_HARD_PRESSES,
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("Difficulty weights cannot be negative")
    if sum(weights) <= 0:
        raise ValueError("At least one difficulty weight must be positive")
    if not isinstance(INCLUDE_SUSPENDED_CARDS, bool):
        raise ValueError("INCLUDE_SUSPENDED_CARDS must be True or False")


def print_summary(notes: List[Dict[str, object]], output_path: Path) -> None:
    """Print extraction statistics and a few samples."""
    field_names = [
        "japanese",
        "english",
        "explanation",
        "example_japanese",
        "example_english",
    ]

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Notes extracted: {len(notes):,}")
    print(f"Output file:    {output_path}")

    print("\nFIELD COVERAGE")
    for field_name in field_names:
        populated = sum(
            1 for note in notes if str(note.get(field_name, "")).strip()
        )
        missing = len(notes) - populated
        print(
            f"  {field_name:18s}: "
            f"{populated:>5,} populated | {missing:>5,} empty"
        )

    if notes and PREVIEW_NOTES > 0:
        print("\nSAMPLE OUTPUT")
        print("-" * 60)

        for note in notes[:PREVIEW_NOTES]:
            print(f"\nNote {note['id']} (Anki ID {note['source_note_id']})")
            print(f"  Japanese:    {note['japanese']}")
            print(f"  English:     {note['english']}")

            explanation = str(note["explanation"]).replace("\n", " ")
            if len(explanation) > 180:
                explanation = explanation[:177] + "..."
            print(f"  Explanation: {explanation}")

            if note["example_japanese"]:
                print(f"  JP example:  {note['example_japanese']}")
            if note["example_english"]:
                print(f"  EN example:  {note['example_english']}")

    print("\nThe JSON is ready for the V3 AI-generation script.")
    print("=" * 60)


def print_selected_summary(
    records: Sequence[Dict[str, object]],
    output_path: Path,
    diagnostics: Dict[str, object],
) -> None:
    """Print selected-mode counts, directions, and highest-ranked cards."""
    unique_cards = {int(row["source_card_id"]) for row in records}
    unique_notes = {int(row["source_note_id"]) for row in records}
    second_copies = sum(int(row["output_copy"]) == 2 for row in records)
    directions = Counter(str(row["card_template"]) for row in records)

    print("\n" + "=" * 64)
    print("DIFFICULT-CARD EXTRACTION COMPLETE")
    print("=" * 64)
    print(f"Output rows:          {len(records):,}")
    print(f"Unique cards:        {len(unique_cards):,}")
    print(f"Unique source notes: {len(unique_notes):,}")
    print(f"Second copies:       {second_copies:,}")
    print(f"Cards ranked:        {diagnostics['cards_scored']:,}")
    print(f"Output file:         {output_path}")

    print("\nSELECTED DIRECTIONS (including repeats)")
    for direction, count in directions.most_common():
        print(f"  {direction}: {count:,}")

    if len(records) < OUTPUT_COUNT:
        print(
            f"\nWARNING: Requested {OUTPUT_COUNT:,} rows, but only "
            f"{len(records):,} could be produced from eligible cards."
        )

    if records and PREVIEW_SELECTED_CARDS > 0:
        print("\nHIGHEST-RANKED UNIQUE CARDS")
        print("-" * 64)
        shown = 0
        seen_cards = set()
        for record in records:
            card_id = int(record["source_card_id"])
            if card_id in seen_cards:
                continue
            seen_cards.add(card_id)
            shown += 1
            print(
                f"#{record['difficulty_rank']:>3}  "
                f"score={record['difficulty_score']:.4f}  "
                f"{record['card_template']}  "
                f"reviews={record['review_count']}  "
                f"Again={record['again_count']}  "
                f"Hard={record['hard_count']}  "
                f"mature lapses={record['mature_lapse_count']}"
            )
            print(f"     {record['japanese']} / {record['english']}")
            if shown >= PREVIEW_SELECTED_CARDS:
                break

    print("\nORDER CHECK")
    print(
        f"  First {second_copies:,} rows: first instances of the hardest cards"
    )
    print(
        f"  Last {second_copies:,} rows: second instances of those same cards"
    )
    print("\nThe JSON is ready for the V3 AI-generation script.")
    print("=" * 64)


def main() -> bool:
    """Main function, designed for direct execution in Spyder."""
    print("=" * 60)
    print("ANKI CONTENT EXTRACTOR V3")
    print(f"Version: {SCRIPT_VERSION} (no command-line arguments required)")
    print("=" * 60)

    input_path = resolve_input_path(INPUT_FILE)
    output_path = resolve_output_path(OUTPUT_FILE)

    print(
        "Mode:           "
        + ("SELECTED DIFFICULT CARDS" if SELECTION else "FULL DECK")
    )
    print(f"Input:          {input_path}")
    print(f"Output folder:  {get_output_root()}")
    print(f"Output:         {output_path}")
    if SELECTION:
        print(f"Output rows:    {OUTPUT_COUNT} (including repeats)")
        print(f"Repeat top:     {REPEAT_TOP_CARDS}")
        print(f"Recency:        {RECENCY_HALF_LIFE_DAYS:g}-day half-life")
    else:
        print(f"Sort order:     {SORT_ORDER}")
    print(f"Remove furigana: {REMOVE_FURIGANA}")
    print("=" * 60)

    try:
        validate_configuration()

        if not input_path.exists():
            print(f"\nERROR: Input file not found: {input_path}")
            print("Put Base.apkg beside the script or change INPUT_FILE.")
            return False

        if not zipfile.is_zipfile(input_path):
            print(f"\nERROR: This is not a valid APKG/ZIP file: {input_path}")
            return False

        print("\nOpening Anki package...")

        with tempfile.TemporaryDirectory(prefix="anki_extract_") as temp_dir:
            database_path = Path(temp_dir) / "collection.sqlite3"

            with zipfile.ZipFile(input_path, "r") as archive:
                collection_member = extract_collection_database(
                    archive,
                    database_path,
                )

            print(f"Using database: {collection_member}")

            connection = sqlite3.connect(
                f"file:{database_path.as_posix()}?mode=ro",
                uri=True,
            )
            connection.row_factory = sqlite3.Row
            register_anki_collation(connection)
            connection.execute("PRAGMA query_only = ON")

            try:
                (
                    note_type_names,
                    fields_by_note_type,
                    schema_style,
                ) = load_note_type_fields(connection)

                print(f"Database style: {schema_style}")

                if SELECTION:
                    # Reuse the full-mode schema display and field checks by
                    # inspecting note types before ranking their cards.
                    note_rows = get_note_rows(connection)
                    note_type_counts = Counter(
                        int(row["mid"]) for row in note_rows
                    )
                    show_note_type_info(
                        note_type_names,
                        fields_by_note_type,
                        note_type_counts,
                    )
                    warn_missing_configured_fields(
                        note_type_names,
                        fields_by_note_type,
                        note_type_counts,
                    )
                    print("\nSCORING INDIVIDUAL CARDS")
                    print("-" * 60)
                    template_names = load_template_names(connection)
                    ranked_cards, diagnostics = score_cards(
                        connection,
                        fields_by_note_type,
                        template_names,
                    )
                    selections, repeated_count = select_output_cards(
                        ranked_cards
                    )
                    notes = make_selected_records(selections)
                else:
                    notes = extract_notes(
                        connection,
                        note_type_names,
                        fields_by_note_type,
                    )
            finally:
                connection.close()

        if not notes:
            if SELECTION:
                print("\nERROR: No eligible reviewed cards were found.")
            else:
                print("\nERROR: No notes were found in the deck.")
            return False

        print("\nSaving compact JSON output...")
        if SELECTION:
            save_selected_results(
                notes,
                output_path,
                input_path,
                collection_member,
                diagnostics,
                repeated_count,
            )
            print_selected_summary(notes, output_path, diagnostics)
        else:
            save_results(
                notes,
                output_path,
                input_path,
                collection_member,
            )
            print_summary(notes, output_path)
        return True

    except ImportError as error:
        print(f"\nERROR: {error}")
        return False
    except Exception as error:
        print(f"\nERROR: {error}")
        import traceback
        traceback.print_exc()
        return False


# ===================== RUN =====================
if __name__ == "__main__":
    print()
    success = main()

    if success:
        print("\nDone!")
    else:
        print("\nFailed!")
        sys.exit(1)
