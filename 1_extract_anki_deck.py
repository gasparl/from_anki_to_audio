#!/usr/bin/env python3
"""
ANKI CONTENT EXTRACTOR V3 - FULL OR DIFFICULT-NOTE MODE

Reads an Anki .apkg deck and creates one compact JSON file containing only
the material needed for the later AI sentence-generation step.

Set ``SELECTION`` in the user settings below:

* False extracts every Anki note once. Active notes keep the original two
  generated units; fully suspended notes request only one.
* True scores each card direction from its own review history, combines sibling
  directions into one weighted score per note, selects the hardest notes, and
  repeats the hardest notes at the end of the output.

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

In full mode, the script extracts every Anki note once. In selected mode,
JP-to-EN and EN-to-JP cards are scored independently, then combined 40:60 into
one note score. Selection therefore never duplicates a note merely because
both sibling directions were difficult.

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


SCRIPT_VERSION = "3.4-ADAPTIVE-UNITS"


# ===================== SIMPLE USER SETTINGS =====================

# False = full-note extraction, with fewer units for fully suspended notes.
# True  = select difficult notes using combined card-direction histories.
SELECTION = False

# Put the APKG in the same folder as this script, or enter an absolute path.
INPUT_FILE = "Base.apkg"
OUTPUT_ROOT_DIR_NAME = "anki_audio_output_v3"
OUTPUT_FILE = "anki_content_v3.json"

# Full mode still outputs every note. These settings only vary how many units
# Stage 2 generates: a note is "fully suspended" only when all its cards are
# suspended, so one active direction is enough to retain the normal target.
FULL_MODE_UNITS_PER_NOTE = 2
FULL_MODE_UNITS_IF_FULLY_SUSPENDED = 1

# ---------------- Settings used only when SELECTION = True ----------------

# Total rows written, INCLUDING repeat copies.
OUTPUT_COUNT = 300

# The hardest N notes are included twice. With the defaults, the
# first 60 rows are their first instances and the last 60 rows are copies;
# between them are ranks 61-240. Total: 240 unique notes + 60 copies = 300.
REPEAT_TOP_NOTES = 60

# Stage 2 generates this many learning units for each selected output row.
# At 1, ranks 1-60 get two units in total because those rows are repeated,
# while ranks 61-240 get one.
UNITS_PER_SELECTED_ROW = 1

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

# Each direction first receives its own complete review-history score. The two
# sibling scores are then averaged into one NOTE score using these weights.
# They need not add to 1. A missing, unreviewed, or excluded direction
# contributes zero instead of letting the other direction inherit its weight.
# EN -> JP receives slightly more weight because it measures Japanese recall;
# JP -> EN remains substantial because it also reflects comprehension.
NOTE_WEIGHT_EN_TO_JP = 0.60
NOTE_WEIGHT_JP_TO_EN = 0.40

# Include suspended cards in selected-mode scoring. False is the sensible
# default when suspension means "I know this already": a suspended direction
# is ignored, and a note with no remaining eligible direction is not selected.
INCLUDE_SUSPENDED_CARDS = False

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
PREVIEW_SELECTED_NOTES = 8

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
    card_queues_by_note: Dict[int, List[int]] = defaultdict(list)
    for card_row in connection.execute("SELECT nid, queue FROM cards"):
        card_queues_by_note[int(card_row["nid"])].append(
            int(card_row["queue"])
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

    print("\nEXTRACTING CONTENT")
    print("-" * 50)

    extracted: List[Dict[str, object]] = []

    for index, row in enumerate(rows, 1):
        note_id = int(row["id"])
        card_queues = card_queues_by_note.get(note_id, [])
        all_cards_suspended = bool(card_queues) and all(
            queue == SUSPENDED_QUEUE for queue in card_queues
        )
        generation_units = (
            FULL_MODE_UNITS_IF_FULLY_SUSPENDED
            if all_cards_suspended
            else FULL_MODE_UNITS_PER_NOTE
        )
        extracted.append(
            {
                "id": index,
                "source_note_id": note_id,
                **extract_note_content(row, fields_by_note_type),
                "generation_units": generation_units,
                "all_cards_suspended": all_cards_suspended,
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
        if (
            not INCLUDE_SUSPENDED_CARDS
            and int(row["card_queue"]) == SUSPENDED_QUEUE
        ):
            excluded["suspended"] += 1
            continue
        if len(reviews) < MIN_REVIEWS_PER_CARD:
            excluded["too_few_reviews"] += 1
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
        card_difficulty_score = sum(
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
                "card_difficulty_score": card_difficulty_score,
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
            -float(item["card_difficulty_score"]),
            -float(item["components"]["recent_again"]),
            -float(item["components"]["lifetime_again_rate"]),
            -float(item["components"]["mature_lapses"]),
            -float(item["components"]["hard_presses"]),
            int(item["source_card_id"]),
        )
    )
    for rank, card in enumerate(scored, 1):
        card["card_difficulty_rank"] = rank

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


def average_card_value(
    cards: Sequence[Dict[str, object]],
    field_name: str,
) -> float:
    """Return the arithmetic mean of one numeric card field."""
    return sum(float(card[field_name]) for card in cards) / len(cards)


def average_component(
    cards: Sequence[Dict[str, object]],
    component_name: str,
) -> float:
    """Return the mean value of a card difficulty component."""
    return sum(
        float(card["components"][component_name]) for card in cards
    ) / len(cards)


def aggregate_cards_into_notes(
    scored_cards: Sequence[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Combine sibling direction scores into one ranked score per note."""
    cards_by_note: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for card in scored_cards:
        cards_by_note[int(card["source_note_id"])].append(card)

    en_to_jp_name = safe_field_name("EN to JP")
    jp_to_en_name = safe_field_name("JP to EN")
    component_names = (
        "recent_again",
        "lifetime_again_rate",
        "mature_lapses",
        "hard_presses",
    )
    coverage = Counter()
    ranked_notes: List[Dict[str, object]] = []

    for note_id, cards in cards_by_note.items():
        cards = sorted(
            cards,
            key=lambda card: (
                int(card["card_ordinal"]),
                int(card["source_card_id"]),
            ),
        )
        en_to_jp_cards = [
            card for card in cards
            if safe_field_name(str(card["card_template"])) == en_to_jp_name
        ]
        jp_to_en_cards = [
            card for card in cards
            if safe_field_name(str(card["card_template"])) == jp_to_en_name
        ]

        direction_groups = []
        direction_scores: Dict[str, float] = {}
        if en_to_jp_cards:
            score = average_card_value(
                en_to_jp_cards,
                "card_difficulty_score",
            )
            direction_scores["EN to JP"] = score
            direction_groups.append(
                ("EN to JP", en_to_jp_cards, float(NOTE_WEIGHT_EN_TO_JP))
            )
        if jp_to_en_cards:
            score = average_card_value(
                jp_to_en_cards,
                "card_difficulty_score",
            )
            direction_scores["JP to EN"] = score
            direction_groups.append(
                ("JP to EN", jp_to_en_cards, float(NOTE_WEIGHT_JP_TO_EN))
            )

        if direction_groups:
            configured_weight = float(
                NOTE_WEIGHT_EN_TO_JP + NOTE_WEIGHT_JP_TO_EN
            )
            note_score = sum(
                direction_scores[name] * weight
                for name, _group_cards, weight in direction_groups
            ) / configured_weight
            combined_components = {
                component_name: sum(
                    average_component(group_cards, component_name) * weight
                    for _name, group_cards, weight in direction_groups
                ) / configured_weight
                for component_name in component_names
            }
            if en_to_jp_cards and jp_to_en_cards:
                coverage["both_directions"] += 1
            elif en_to_jp_cards:
                coverage["en_to_jp_only"] += 1
            else:
                coverage["jp_to_en_only"] += 1
            aggregation_method = (
                "fixed direction weights; unavailable directions score zero"
            )
        else:
            # Generic fallback for decks whose templates have other names.
            note_score = average_card_value(cards, "card_difficulty_score")
            combined_components = {
                name: average_component(cards, name)
                for name in component_names
            }
            direction_scores = {
                str(card["card_template"]): float(
                    card["card_difficulty_score"]
                )
                for card in cards
            }
            coverage["other_templates_only"] += 1
            aggregation_method = "equal mean of available cards"

        content_source = cards[0]
        ranked_notes.append(
            {
                "source_note_id": note_id,
                "difficulty_score": note_score,
                "combined_components": combined_components,
                "direction_scores": direction_scores,
                "aggregation_method": aggregation_method,
                "source_cards": cards,
                "japanese": content_source["japanese"],
                "english": content_source["english"],
                "explanation": content_source["explanation"],
                "example_japanese": content_source["example_japanese"],
                "example_english": content_source["example_english"],
            }
        )

    ranked_notes.sort(
        key=lambda note: (
            -float(note["difficulty_score"]),
            -float(note["direction_scores"].get("EN to JP", -1.0)),
            -float(note["direction_scores"].get("JP to EN", -1.0)),
            int(note["source_note_id"]),
        )
    )
    for rank, note in enumerate(ranked_notes, 1):
        note["difficulty_rank"] = rank

    diagnostics = {
        "notes_scored": len(ranked_notes),
        "note_direction_coverage": dict(coverage),
    }
    return ranked_notes, diagnostics


def select_output_notes(
    ranked_notes: Sequence[Dict[str, object]],
) -> Tuple[List[Tuple[Dict[str, object], int]], int]:
    """Put unique notes first and second copies of the hardest notes last."""
    unique_target = OUTPUT_COUNT - REPEAT_TOP_NOTES
    selected_unique = list(ranked_notes[:unique_target])
    repeated_count = min(REPEAT_TOP_NOTES, len(selected_unique))

    # This ordering places ranks 1..REPEAT_TOP_NOTES at the beginning, and
    # their intentional second instances at the very end.
    selections = [(note, 1) for note in selected_unique]
    selections.extend((note, 2) for note in selected_unique[:repeated_count])
    return selections, repeated_count


def card_audit_record(card: Dict[str, object]) -> Dict[str, object]:
    """Return compact, JSON-safe evidence for one contributing card."""
    return {
        "source_card_id": int(card["source_card_id"]),
        "card_ordinal": int(card["card_ordinal"]),
        "card_template": card["card_template"],
        "card_difficulty_score": round(
            float(card["card_difficulty_score"]), 8
        ),
        "review_count": int(card["review_count"]),
        "again_count": int(card["again_count"]),
        "hard_count": int(card["hard_count"]),
        "mature_lapse_count": int(card["mature_lapse_count"]),
        "difficulty_components": {
            name: round(float(value), 8)
            for name, value in card["components"].items()
        },
    }


def make_selected_records(
    selections: Sequence[Tuple[Dict[str, object], int]],
) -> List[Dict[str, object]]:
    """Create normal pipeline rows plus note-ranking audit fields."""
    records: List[Dict[str, object]] = []
    for output_id, (note, copy_number) in enumerate(selections, 1):
        source_cards = list(note["source_cards"])
        records.append(
            {
                # These seven fields are the unchanged V3 contract.
                "id": output_id,
                "source_note_id": int(note["source_note_id"]),
                "japanese": note["japanese"],
                "english": note["english"],
                "explanation": note["explanation"],
                "example_japanese": note["example_japanese"],
                "example_english": note["example_english"],
                # Stage 2 ignores these useful audit fields.
                "difficulty_rank": int(note["difficulty_rank"]),
                "difficulty_score": round(
                    float(note["difficulty_score"]), 8
                ),
                "output_copy": copy_number,
                "aggregation_method": note["aggregation_method"],
                "direction_scores": {
                    name: round(float(value), 8)
                    for name, value in note["direction_scores"].items()
                },
                "combined_difficulty_components": {
                    name: round(float(value), 8)
                    for name, value in note["combined_components"].items()
                },
                # Stage 2 honors this optional per-row generation target.
                "generation_units": UNITS_PER_SELECTED_ROW,
                "source_card_ids": [
                    int(card["source_card_id"]) for card in source_cards
                ],
                "card_score_details": [
                    card_audit_record(card) for card in source_cards
                ],
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
            "generation_allocation": {
                "normal_units_per_note": FULL_MODE_UNITS_PER_NOTE,
                "fully_suspended_units_per_note": (
                    FULL_MODE_UNITS_IF_FULLY_SUSPENDED
                ),
                "fully_suspended_notes": sum(
                    bool(note.get("all_cards_suspended")) for note in notes
                ),
            },
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
    unique_cards = {
        int(card_id)
        for row in records
        for card_id in row["source_card_ids"]
    }
    unique_notes = {int(row["source_note_id"]) for row in records}
    output = {
        "metadata": {
            "schema_version": 3,
            "extractor_version": SCRIPT_VERSION,
            "source_file": input_path.name,
            "collection_member": collection_member,
            "total_notes": len(records),
            "unique_contributing_cards": len(unique_cards),
            "unique_source_notes": len(unique_notes),
            "repeated_notes": repeated_count,
            "ranked_unit": "Anki note with combined direction scores",
            "ranking_formula": (
                "weighted mean of independently scored EN-to-JP and "
                "JP-to-EN cards; unavailable directions contribute zero"
            ),
            "output_order": (
                "unique notes in combined-score order, followed by second "
                "copies of the hardest notes"
            ),
            "selection_settings": {
                "selection": SELECTION,
                "output_count_including_repeats": OUTPUT_COUNT,
                "repeat_top_notes": REPEAT_TOP_NOTES,
                "units_per_selected_row": UNITS_PER_SELECTED_ROW,
                "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
                "old_review_weight_floor": OLD_REVIEW_WEIGHT_FLOOR,
                "smoothing_prior_reviews": SMOOTHING_PRIOR_REVIEWS,
                "minimum_reviews_per_card": MIN_REVIEWS_PER_CARD,
                "mature_interval_days": MATURE_INTERVAL_DAYS,
                "include_suspended_cards": INCLUDE_SUSPENDED_CARDS,
                "card_score_weights": diagnostics["normalized_weights"],
                "note_direction_weights": {
                    "EN to JP": NOTE_WEIGHT_EN_TO_JP,
                    "JP to EN": NOTE_WEIGHT_JP_TO_EN,
                },
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
    unit_settings = {
        "FULL_MODE_UNITS_PER_NOTE": FULL_MODE_UNITS_PER_NOTE,
        "FULL_MODE_UNITS_IF_FULLY_SUSPENDED": (
            FULL_MODE_UNITS_IF_FULLY_SUSPENDED
        ),
        "UNITS_PER_SELECTED_ROW": UNITS_PER_SELECTED_ROW,
    }
    for setting_name, value in unit_settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{setting_name} must be a positive integer")
    if (
        FULL_MODE_UNITS_IF_FULLY_SUSPENDED
        > FULL_MODE_UNITS_PER_NOTE
    ):
        raise ValueError(
            "FULL_MODE_UNITS_IF_FULLY_SUSPENDED cannot exceed "
            "FULL_MODE_UNITS_PER_NOTE"
        )
    if not SELECTION:
        return

    if isinstance(OUTPUT_COUNT, bool) or not isinstance(OUTPUT_COUNT, int):
        raise ValueError("OUTPUT_COUNT must be an integer")
    if OUTPUT_COUNT < 1:
        raise ValueError("OUTPUT_COUNT must be at least 1")
    if (
        isinstance(REPEAT_TOP_NOTES, bool)
        or not isinstance(REPEAT_TOP_NOTES, int)
    ):
        raise ValueError("REPEAT_TOP_NOTES must be an integer")
    if not 0 <= REPEAT_TOP_NOTES < OUTPUT_COUNT:
        raise ValueError(
            "REPEAT_TOP_NOTES must be between 0 and OUTPUT_COUNT - 1"
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
    note_direction_weights = (
        NOTE_WEIGHT_EN_TO_JP,
        NOTE_WEIGHT_JP_TO_EN,
    )
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or weight <= 0
        for weight in note_direction_weights
    ):
        raise ValueError(
            "Note direction weights must be finite numbers above 0"
        )
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

    generation_counts = Counter(
        int(note.get("generation_units", FULL_MODE_UNITS_PER_NOTE))
        for note in notes
    )
    fully_suspended = sum(
        bool(note.get("all_cards_suspended")) for note in notes
    )
    print(f"Fully suspended: {fully_suspended:,}")
    print(
        "Unit targets:    "
        + ", ".join(
            f"{units} unit(s): {count:,} notes"
            for units, count in sorted(generation_counts.items())
        )
    )

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
    """Print selected-mode counts and highest-ranked combined notes."""
    unique_records = [
        row for row in records if int(row["output_copy"]) == 1
    ]
    unique_cards = {
        int(card_id)
        for row in unique_records
        for card_id in row["source_card_ids"]
    }
    unique_notes = {int(row["source_note_id"]) for row in unique_records}
    second_copies = sum(int(row["output_copy"]) == 2 for row in records)
    coverage = Counter()
    for row in unique_records:
        directions = set(row["direction_scores"])
        if {"EN to JP", "JP to EN"}.issubset(directions):
            coverage["both directions"] += 1
        elif "EN to JP" in directions:
            coverage["EN to JP only"] += 1
        elif "JP to EN" in directions:
            coverage["JP to EN only"] += 1
        else:
            coverage["other templates"] += 1

    print("\n" + "=" * 64)
    print("DIFFICULT-NOTE EXTRACTION COMPLETE")
    print("=" * 64)
    print(f"Output rows:          {len(records):,}")
    print(
        "Stage-2 units:       "
        f"{sum(int(row['generation_units']) for row in records):,}"
    )
    print(f"Unique notes:        {len(unique_notes):,}")
    print(f"Contributing cards:  {len(unique_cards):,}")
    print(f"Repeated notes:      {second_copies:,}")
    print(f"Notes ranked:        {diagnostics['notes_scored']:,}")
    print(f"Cards scored first:  {diagnostics['cards_scored']:,}")
    print(f"Output file:         {output_path}")

    print("\nDIRECTION COVERAGE (unique selected notes)")
    for label, count in coverage.most_common():
        print(f"  {label}: {count:,}")

    if len(records) < OUTPUT_COUNT:
        print(
            f"\nWARNING: Requested {OUTPUT_COUNT:,} rows, but only "
            f"{len(records):,} could be produced from eligible notes."
        )

    if unique_records and PREVIEW_SELECTED_NOTES > 0:
        print("\nHIGHEST-RANKED UNIQUE NOTES")
        print("-" * 64)
        for record in unique_records[:PREVIEW_SELECTED_NOTES]:
            direction_text = ", ".join(
                f"{name}={float(score):.4f}"
                for name, score in record["direction_scores"].items()
            )
            print(
                f"#{record['difficulty_rank']:>3}  "
                f"score={record['difficulty_score']:.4f}  "
                f"[{direction_text}]"
            )
            print(f"     {record['japanese']} / {record['english']}")

    print("\nORDER CHECK")
    print(
        f"  First {second_copies:,} rows: first instances of the hardest notes"
    )
    print(
        f"  Last {second_copies:,} rows: second instances of those same notes"
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
        + ("SELECTED DIFFICULT NOTES" if SELECTION else "FULL DECK")
    )
    print(f"Input:          {input_path}")
    print(f"Output folder:  {get_output_root()}")
    print(f"Output:         {output_path}")
    if SELECTION:
        print(f"Output rows:    {OUTPUT_COUNT} (including repeats)")
        print(f"Repeat top:     {REPEAT_TOP_NOTES} notes")
        print(f"Units per row:  {UNITS_PER_SELECTED_ROW}")
        print(f"Recency:        {RECENCY_HALF_LIFE_DAYS:g}-day half-life")
        print(
            "Note combine:   "
            f"EN->JP {NOTE_WEIGHT_EN_TO_JP:g}, "
            f"JP->EN {NOTE_WEIGHT_JP_TO_EN:g}"
        )
    else:
        print(f"Sort order:     {SORT_ORDER}")
        print(
            "Units per note: "
            f"{FULL_MODE_UNITS_PER_NOTE} active / "
            f"{FULL_MODE_UNITS_IF_FULLY_SUSPENDED} fully suspended"
        )
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
                    scored_cards, diagnostics = score_cards(
                        connection,
                        fields_by_note_type,
                        template_names,
                    )
                    print("\nCOMBINING CARD SCORES INTO NOTE SCORES")
                    print("-" * 60)
                    ranked_notes, note_diagnostics = (
                        aggregate_cards_into_notes(scored_cards)
                    )
                    diagnostics.update(note_diagnostics)
                    selections, repeated_count = select_output_notes(
                        ranked_notes
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
                print("\nERROR: No eligible reviewed notes were found.")
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
