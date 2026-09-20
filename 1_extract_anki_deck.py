#!/usr/bin/env python3
"""
ANKI CONTENT EXTRACTOR V3

Reads an Anki .apkg deck and creates one compact JSON file containing only
the material needed for the later AI sentence-generation step.

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

The script extracts Anki NOTES rather than generated CARDS. In this deck, one
note normally creates both a JP-to-EN and an EN-to-JP card. Extracting notes
avoids sending the same information to the AI twice.

For modern Anki packages, install the only non-standard dependency with:
    pip install zstandard
"""

import html
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_VERSION = "3.0-SPYDER"


# ===================== CONFIGURATION =====================

# Put the APKG in the same folder as this script, or enter an absolute path.
INPUT_FILE = "Base.apkg"
OUTPUT_ROOT_DIR_NAME = "anki_audio_output_v3"
OUTPUT_FILE = "anki_content_v3.json"

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

# =========================================================


FIELD_SEPARATOR = "\x1f"


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

    configured_fields = [
        JAPANESE_FIELD,
        ENGLISH_FIELD,
        EXPLANATION_FIELD,
        EXAMPLE_JAPANESE_FIELD,
        EXAMPLE_ENGLISH_FIELD,
    ]

    used_note_types = set(note_type_counts)
    for note_type_id in used_note_types:
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

    print("\nEXTRACTING CONTENT")
    print("-" * 50)

    extracted: List[Dict[str, object]] = []

    for index, row in enumerate(rows, 1):
        note_id = int(row["id"])
        note_type_id = int(row["mid"])
        field_names = fields_by_note_type.get(note_type_id, [])
        field_values = (row["flds"] or "").split(FIELD_SEPARATOR)

        # Fill missing stored fields with empty strings. Any unexpected extras
        # are deliberately ignored because this is a content-only export.
        if len(field_values) < len(field_names):
            field_values.extend(
                "" for _ in range(len(field_names) - len(field_values))
            )

        fields = {
            field_name: field_value
            for field_name, field_value in zip(field_names, field_values)
        }

        extracted.append(
            {
                "id": index,
                "source_note_id": note_id,
                "japanese": get_field(fields, JAPANESE_FIELD),
                "english": get_field(fields, ENGLISH_FIELD),
                "explanation": get_field(fields, EXPLANATION_FIELD),
                "example_japanese": get_field(
                    fields, EXAMPLE_JAPANESE_FIELD
                ),
                "example_english": get_field(
                    fields, EXAMPLE_ENGLISH_FIELD
                ),
            }
        )

        if PROGRESS_EVERY and index % PROGRESS_EVERY == 0:
            print(f"  Extracted {index:,}/{len(rows):,} notes...")

    return extracted


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


def main() -> bool:
    """Main function, designed for direct execution in Spyder."""
    print("=" * 60)
    print("ANKI CONTENT EXTRACTOR V3")
    print(f"Version: {SCRIPT_VERSION} (no command-line arguments required)")
    print("=" * 60)

    input_path = resolve_input_path(INPUT_FILE)
    output_path = resolve_output_path(OUTPUT_FILE)

    print(f"Input:          {input_path}")
    print(f"Output folder:  {get_output_root()}")
    print(f"Output:         {output_path}")
    print(f"Sort order:     {SORT_ORDER}")
    print(f"Remove furigana: {REMOVE_FURIGANA}")
    print("=" * 60)

    try:
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

                notes = extract_notes(
                    connection,
                    note_type_names,
                    fields_by_note_type,
                )
            finally:
                connection.close()

        if not notes:
            print("\nERROR: No notes were found in the deck.")
            return False

        print("\nSaving compact JSON output...")
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
