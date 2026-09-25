#!/usr/bin/env python3
"""
ANKI JAPANESE AUDIO-LEARNING GENERATOR V3

Reads ``translated_output_v3.json`` and creates resumable Japanese-first audio,
one or more chaptered M4B audiobooks, and a phone-friendly HTML text companion
beside every finished M4B.

Each learning unit becomes exactly one M4B chapter with this sequence:

  1. Playback number spoken naturally in Japanese with 番目
  2. Japanese sentence: slow Edge voices
  3. Close-structure English translation
  4. Japanese sentence: moderately paced Edge voices

There are no explanations, breakdowns, or separate literal translations.

Edge TTS is used for Japanese and English. The two available Japanese neural
voices rotate deterministically, so reruns and resumes produce the same plan.

The Japanese/English body parts are generated and saved per immutable unit ID,
so TTS work remains resumable while a run is incomplete. Final playback order
is a deterministic shuffle. The spoken number follows that shuffled playback
position rather than the internal unit ID. Completed units are packed into
approximately seven-hour M4B volumes, with an eight-hour ceiling where normal
unit boundaries permit it. Each volume also contains at most 255 chapters so
both M4B chapter formats cover every example. Every volume contains whole
units only. After all
M4Bs and HTML companions verify successfully, temporary MP3/cache/build files
are deleted.

Designed for direct execution in Spyder:
  1. Keep this script beside the other V3 scripts. It reads
     ``anki_audio_output_v3/translated_output_v3.json``.
  2. Set UNIT_LIMIT to 5 or 20 for a limited test, or None for all unfinished
     units.
  3. Press Run. Progress is saved after every unit and resumes automatically.

Requirements:
    pip install edge-tts pydub

FFmpeg and FFprobe must be installed and available on PATH.

"""

import asyncio
import hashlib
import html
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from pydub import AudioSegment, effects
except ImportError:
    AudioSegment = None
    effects = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

SCRIPT_VERSION = "3.1-EDGE-JA-HTML-M4B-SPYDER"


# ===================== ONLY USER SETTING =====================

# None = process every unfinished unit.
# 5 or 20 = process only the next 5 or 20 unfinished units this run.
UNIT_LIMIT = None

# Edge currently exposes two Japanese neural voices. Each sentence uses them in
# a deterministic alternating order. To prioritize one familiar voice, replace
# the tuple with, for example, ("ja-JP-NanamiNeural",).
JAPANESE_VOICES = (
    "ja-JP-KeitaNeural",
    "ja-JP-NanamiNeural",
)

# Changing this creates a different deterministic playback order. It does not
# invalidate generated unit body audio; it only rebuilds audiobook layout.
SHUFFLE_SEED = 20260925

# ===================== FIXED INTERNAL SETTINGS =====================

OUTPUT_ROOT_DIR_NAME = "anki_audio_output_v3"
INPUT_JSON = "translated_output_v3.json"
PROGRESS_FILE = "audio_progress_v3.json"
MANIFEST_FILE = "audio_manifest_v3.json"

# All intermediate audio is isolated in one disposable work folder. It remains
# during partial runs for resumability and is removed after verified M4B output.
WORK_DIR_NAME = "_audio_work"
TTS_CACHE_DIR_NAME = "tts_cache"
BUILD_TEMP_AUDIO_DIR_NAME = "assembly"
UNIT_AUDIO_DIR_NAME = "unit_parts"
UNIT_NUMBER_AUDIO_DIR_NAME = "number_parts"
AUDIOBOOK_OUTPUT_DIR_NAME = "."  # Final M4Bs live directly in the output root.
AUDIOBOOK_BASENAME = "japanese_audio_learning_v3"
now = datetime.now().astimezone()
AUDIOBOOK_TITLE = (
    f"アンキ発日本語リスニング・"
    f"{now.year}年{now.month}月{now.day}日"
)
HTML_COMPANION_VERSION = 1

# Pauses preserve the intent of the older generator.
PAUSE_IN_INITIAL_JAPANESE_MS = 1300
PAUSE_AFTER_INITIAL_JAPANESE_MS = 1600
PAUSE_IN_FINAL_JAPANESE_MS = 800
PAUSE_END_SILENCE_MS = 900

# Edge rates retain the natural timing of the original generator.
JA_RATE_UNIT_NUMBER = "+0%"
JA_RATE_JAPANESE_SLOW = "-30%"
JA_RATE_JAPANESE_FINAL = "-10%"
EN_RATE = "+0%"

JAPANESE_NUMBER_MALE_VOICE = "ja-JP-KeitaNeural"
JAPANESE_NUMBER_FEMALE_VOICE = "ja-JP-NanamiNeural"
ENGLISH_MALE_VOICE = "en-US-SteffanNeural"

JAPANESE_VOICE_KEYS = tuple(
    f"ja_edge_{index + 1}" for index in range(len(JAPANESE_VOICES))
)
VOICE_MAP = {
    **{
        key: {"provider": "edge-tts", "voice": voice}
        for key, voice in zip(JAPANESE_VOICE_KEYS, JAPANESE_VOICES)
    },
    "ja_number_male": {
        "provider": "edge-tts",
        "voice": JAPANESE_NUMBER_MALE_VOICE,
    },
    "ja_number_female": {
        "provider": "edge-tts",
        "voice": JAPANESE_NUMBER_FEMALE_VOICE,
    },
    "en_male": {"provider": "edge-tts", "voice": ENGLISH_MALE_VOICE},
}

# Durable intermediate MP3 settings.
MP3_BITRATE = "192k"
OUTPUT_FRAME_RATE = 24000
OUTPUT_CHANNELS = 1
OUTPUT_SAMPLE_WIDTH = 2
NORMALIZE_HEADROOM_DB = 1.0
MIN_MP3_BYTES = 120
MIN_AUDIO_DURATION_MS = 40

# Final M4B encoding: a balanced speech setting for spoken learning material.
AUDIOBOOK_AAC_CODEC = "aac"
AUDIOBOOK_AAC_PROFILE = "aac_low"
AUDIOBOOK_AAC_BITRATE = "80k"
AUDIOBOOK_FRAME_RATE = OUTPUT_FRAME_RATE
AUDIOBOOK_CHANNELS = OUTPUT_CHANNELS
AUDIOBOOK_CHAPTER_TIMEBASE = 1000
MIN_M4B_BYTES = 1024
CHAPTER_TIME_TOLERANCE_SECONDS = 0.35
VOLUME_DURATION_TOLERANCE_SECONDS = 1.25

# Volumes target seven hours. Full volumes are intended to remain between six
# and eight hours; the final volume may be shorter when the total cannot divide
# evenly. A chapter-limited volume may also be shorter. A single unusually
# long unit is never split.
MIN_VOLUME_HOURS = 6.0
TARGET_VOLUME_HOURS = 7.0
MAX_VOLUME_HOURS = 8.0
# The Nero chapter index in M4B holds at most 255 entries. Keep both Nero and
# QuickTime chapter indexes complete by splitting only between whole units.
MAX_CHAPTERS_PER_VOLUME = 255

# Cleanup. Keep work files only while a run is incomplete or has failed.
CLEAN_BUILD_ARTIFACTS_AFTER_SUCCESS = True

# Robustness.
MAX_TTS_RETRIES = 5
RETRY_BASE_SLEEP = 0.7
RETRY_JITTER = 0.4
STOP_AFTER_CONSECUTIVE_FAILURES = 5
BETWEEN_UNITS_SLEEP = 0.1

# Number-clip progress is printed at the first clip, at this interval, and at
# completion. Body-phase ETA treats one number clip as one TTS component; the
# estimate becomes measured directly once number-clip generation begins.
NUMBER_CLIP_PROGRESS_EVERY = 250

# =========================================================


JAPANESE_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_base_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


BASE_DIR = get_base_dir()
OUTPUT_ROOT_DIR = BASE_DIR / OUTPUT_ROOT_DIR_NAME
INPUT_JSON_PATH = OUTPUT_ROOT_DIR / INPUT_JSON
PROGRESS_PATH = OUTPUT_ROOT_DIR / PROGRESS_FILE
MANIFEST_PATH = OUTPUT_ROOT_DIR / MANIFEST_FILE
WORK_DIR = OUTPUT_ROOT_DIR / WORK_DIR_NAME
TTS_CACHE_DIR = WORK_DIR / TTS_CACHE_DIR_NAME
BUILD_TEMP_AUDIO_DIR = WORK_DIR / BUILD_TEMP_AUDIO_DIR_NAME
UNIT_AUDIO_DIR = WORK_DIR / UNIT_AUDIO_DIR_NAME
UNIT_NUMBER_AUDIO_DIR = WORK_DIR / UNIT_NUMBER_AUDIO_DIR_NAME
AUDIOBOOK_OUTPUT_DIR = OUTPUT_ROOT_DIR / AUDIOBOOK_OUTPUT_DIR_NAME


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace('"', ",")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(data) -> str:
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(str(temporary_path), str(path))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    os.replace(str(temporary_path), str(path))


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def standardize_audio(audio):
    return (
        audio.set_frame_rate(OUTPUT_FRAME_RATE)
        .set_channels(OUTPUT_CHANNELS)
        .set_sample_width(OUTPUT_SAMPLE_WIDTH)
    )


def make_silence(duration_ms: int):
    silence = AudioSegment.silent(
        duration=max(0, int(duration_ms)),
        frame_rate=OUTPUT_FRAME_RATE,
    )
    return standardize_audio(silence)


def load_audio(path: Path):
    audio = AudioSegment.from_file(str(path), format="mp3")
    return standardize_audio(audio)


def audio_file_is_valid(
    path: Path,
    decode: bool = False,
    expected_size: Optional[int] = None,
) -> bool:
    try:
        if not path.exists() or not path.is_file():
            return False
        if path.stat().st_size <= MIN_MP3_BYTES:
            return False
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        if decode:
            audio = AudioSegment.from_file(str(path), format="mp3")
            if len(audio) < MIN_AUDIO_DURATION_MS:
                return False
        return True
    except Exception:
        return False


def atomic_export_audio(
    audio,
    output_path: Path,
    normalize_audio: bool = True,
) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        output_path.stem + ".tmp" + output_path.suffix
    )
    safe_unlink(temporary_path)

    try:
        prepared = standardize_audio(audio)
        if normalize_audio and len(prepared) > 0:
            if not math.isinf(prepared.max_dBFS):
                prepared = effects.normalize(
                    prepared,
                    headroom=NORMALIZE_HEADROOM_DB,
                )

        prepared.export(
            str(temporary_path),
            format="mp3",
            bitrate=MP3_BITRATE,
        )
        if not audio_file_is_valid(temporary_path, decode=True):
            safe_unlink(temporary_path)
            return False
        os.replace(str(temporary_path), str(output_path))
        return audio_file_is_valid(output_path)
    except Exception:
        safe_unlink(temporary_path)
        return False


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(float(seconds) + 0.5))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock


def local_now() -> datetime:
    return datetime.now().astimezone()


def format_local_time(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S %Z")


def format_eta(
    average_seconds: Optional[float],
    remaining_items: float,
    now: datetime,
) -> str:
    if average_seconds is None:
        return "calculating"
    remaining_seconds = max(
        0,
        int(average_seconds * remaining_items + 0.5),
    )
    arrival = now.replace(microsecond=0) + timedelta(
        seconds=remaining_seconds
    )
    return (
        f"{format_duration(remaining_seconds)} "
        f"({arrival.strftime('%H:%M %Z')})"
    )


def format_unit_id(unit_id: str) -> str:
    return f"{int(unit_id):05d}"


def number_clip_output_path(playback_number: int) -> Path:
    """Return the deterministic output path for one playback number."""
    return (
        UNIT_NUMBER_AUDIO_DIR
        / f"unit_number_{int(playback_number):05d}.mp3"
    )


def count_pending_number_clips(total_units: int) -> int:
    """Quickly count clips that still appear to need generation."""
    return sum(
        not audio_file_is_valid(
            number_clip_output_path(playback_number),
            decode=False,
        )
        for playback_number in range(1, total_units + 1)
    )


def average_unit_tts_components(
    plans: Dict[str, Dict],
    unit_ids: Sequence[str],
) -> float:
    """Return the mean TTS-component count in the selected body plans."""
    component_counts = [
        sum(len(part.get("components", [])) for part in plans[unit_id]["parts"])
        for unit_id in unit_ids
    ]
    if not component_counts:
        return 1.0
    return sum(component_counts) / len(component_counts)


def estimated_remaining_unit_equivalents(
    remaining_units: int,
    pending_number_clips: int,
    components_per_unit: float,
) -> float:
    """Combine body units and future number clips for a rough overall ETA."""
    safe_component_count = max(1.0, float(components_per_unit))
    return max(0, remaining_units) + (
        max(0, pending_number_clips) / safe_component_count
    )


def validate_and_load_input(path: Path) -> Tuple[Dict[str, Dict], List[str]]:
    """Load translated_output_v3.json and validate its audio-facing schema."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        root = json.load(file)

    if not isinstance(root, dict):
        raise ValueError("Input JSON root must be an object")

    metadata = root.get("metadata", {})
    if isinstance(metadata, dict):
        status = str(metadata.get("status", "")).strip().casefold()
        if status and status != "complete":
            raise ValueError(
                "translated_output_v3.json is partial. Complete the V3 "
                "content-generation run before starting audio, otherwise "
                "the input hash and unit set would change during resume."
            )

    raw_units = root.get("units")
    if not isinstance(raw_units, dict) or not raw_units:
        raise ValueError("Input JSON must contain a non-empty 'units' object")

    cleaned_units: Dict[str, Dict] = {}
    numeric_ids = set()

    for raw_id, raw_row in raw_units.items():
        unit_id = str(raw_id)
        if not unit_id.isdigit():
            raise ValueError(f"Unit key must be numeric: {unit_id!r}")

        numeric_id = int(unit_id)
        if numeric_id < 1:
            raise ValueError(f"Unit key must be positive: {unit_id!r}")
        if numeric_id in numeric_ids:
            raise ValueError(f"Numerically duplicated unit key: {unit_id!r}")
        numeric_ids.add(numeric_id)

        if not isinstance(raw_row, dict):
            raise ValueError(f"Unit {unit_id} is not an object")

        japanese = clean_text(raw_row.get("japanese", ""))
        english = clean_text(raw_row.get("english", ""))
        if not japanese:
            raise ValueError(f"Unit {unit_id} has empty Japanese")
        if not JAPANESE_RE.search(japanese):
            raise ValueError(f"Unit {unit_id} contains no recognizable Japanese")
        if not english:
            raise ValueError(f"Unit {unit_id} has empty English")
        if JAPANESE_RE.search(english):
            raise ValueError(
                f"Unit {unit_id} English contains Japanese characters"
            )

        try:
            primary_source = int(
                raw_row.get("primary_source_note_number", 0)
            )
        except (TypeError, ValueError):
            primary_source = 0
        if primary_source < 1:
            raise ValueError(
                f"Unit {unit_id} has invalid primary source note"
            )

        supporting_raw = raw_row.get("supporting_source_note_numbers", [])
        if not isinstance(supporting_raw, list):
            raise ValueError(
                f"Unit {unit_id} supporting_source_note_numbers is not a list"
            )
        supporting: List[int] = []
        for value in supporting_raw:
            try:
                number = int(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Unit {unit_id} has an invalid supporting source note"
                ) from error
            if number > 0 and number != primary_source and number not in supporting:
                supporting.append(number)

        cleaned_row = dict(raw_row)
        cleaned_row["japanese"] = japanese
        cleaned_row["english"] = english
        cleaned_row["primary_source_note_number"] = primary_source
        cleaned_row["supporting_source_note_numbers"] = supporting
        cleaned_units[str(numeric_id)] = cleaned_row

    ordered_ids = sorted(cleaned_units, key=int)
    return cleaned_units, ordered_ids


def component(
    text: str,
    language: str,
    voice_key: str,
    rate: str,
    pause_after_ms: int = 0,
) -> Dict:
    voice = VOICE_MAP[voice_key]
    return {
        "text": clean_text(text),
        "language": language,
        "voice_key": voice_key,
        "provider": voice["provider"],
        "voice": voice["voice"],
        "rate": rate,
        "pause_after_ms": int(pause_after_ms),
    }


def japanese_voice_keys_for_unit(unit_id: str) -> List[str]:
    """Rotate the configured Edge voices reproducibly for one unit."""
    if not JAPANESE_VOICE_KEYS:
        raise ValueError("JAPANESE_VOICES must contain at least one voice")
    digest = hashlib.sha256(f"edge-voices:{unit_id}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % len(JAPANESE_VOICE_KEYS)
    return [
        JAPANESE_VOICE_KEYS[(offset + index) % len(JAPANESE_VOICE_KEYS)]
        for index in range(4)
    ]


def build_unit_plan(unit_id: str, row: Dict) -> Dict:
    """Create the durable body plan; shuffled numbering is added at assembly."""
    padded = format_unit_id(unit_id)
    japanese = clean_text(row["japanese"])
    english = clean_text(row["english"])
    japanese_voice_keys = japanese_voice_keys_for_unit(unit_id)

    parts = [
        {
            "number": 2,
            "name": "japanese_slow_varied_voices",
            "output_file": f"unit_{padded}_2_japanese_slow_varied.mp3",
            "components": [
                component(
                    japanese,
                    "ja",
                    japanese_voice_keys[0],
                    JA_RATE_JAPANESE_SLOW,
                    pause_after_ms=PAUSE_IN_INITIAL_JAPANESE_MS,
                ),
                component(
                    japanese,
                    "ja",
                    japanese_voice_keys[1],
                    JA_RATE_JAPANESE_SLOW,
                ),
            ],
            "end_silence_ms": PAUSE_AFTER_INITIAL_JAPANESE_MS,
        },
        {
            "number": 3,
            "name": "english_close_translation",
            "output_file": f"unit_{padded}_3_english_close_translation.mp3",
            "components": [component(english, "en", "en_male", EN_RATE)],
            "end_silence_ms": PAUSE_END_SILENCE_MS,
        },
        {
            "number": 4,
            "name": "japanese_final_varied_voices",
            "output_file": f"unit_{padded}_4_japanese_final_varied.mp3",
            "components": [
                component(
                    japanese,
                    "ja",
                    japanese_voice_keys[2],
                    JA_RATE_JAPANESE_FINAL,
                    pause_after_ms=PAUSE_IN_FINAL_JAPANESE_MS,
                ),
                component(
                    japanese,
                    "ja",
                    japanese_voice_keys[3],
                    JA_RATE_JAPANESE_FINAL,
                ),
            ],
            "end_silence_ms": PAUSE_END_SILENCE_MS,
        },
    ]

    plan = {
        "unit_id": unit_id,
        "tts_provider": "edge-tts",
        "parts": parts,
        "render_settings": {
            "bitrate": MP3_BITRATE,
            "frame_rate": OUTPUT_FRAME_RATE,
            "channels": OUTPUT_CHANNELS,
            "sample_width": OUTPUT_SAMPLE_WIDTH,
            "normalization_headroom_db": NORMALIZE_HEADROOM_DB,
        },
    }
    plan["record_hash"] = sha256_json(plan)
    return plan


def unit_number_voice_key(playback_number: int) -> str:
    digest = hashlib.sha256(str(playback_number).encode("utf-8")).digest()
    return (
        "ja_number_male"
        if digest[0] % 2 == 0
        else "ja_number_female"
    )


def spoken_unit_number(playback_number: int) -> str:
    """Return a concise natural ordinal such as 1番目, without ユニット."""
    return f"{int(playback_number)}番目"


def build_number_part(playback_number: int) -> Dict:
    voice_key = unit_number_voice_key(playback_number)
    return {
        "number": 1,
        "name": "playback_unit_number",
        "output_file": number_clip_output_path(playback_number).name,
        "components": [
            component(
                spoken_unit_number(playback_number),
                "ja",
                voice_key,
                JA_RATE_UNIT_NUMBER,
            )
        ],
        "end_silence_ms": PAUSE_END_SILENCE_MS,
    }


def progress_settings() -> Dict:
    return {
        "script_version": SCRIPT_VERSION,
        "tts_provider": "edge-tts",
        "voices": VOICE_MAP,
        "rates": {
            "unit_number": JA_RATE_UNIT_NUMBER,
            "initial_japanese": JA_RATE_JAPANESE_SLOW,
            "final_japanese": JA_RATE_JAPANESE_FINAL,
            "english": EN_RATE,
        },
        "pauses_ms": {
            "initial_japanese": PAUSE_IN_INITIAL_JAPANESE_MS,
            "after_initial_japanese": PAUSE_AFTER_INITIAL_JAPANESE_MS,
            "final_japanese": PAUSE_IN_FINAL_JAPANESE_MS,
            "part_end": PAUSE_END_SILENCE_MS,
        },
        "audio": {
            "bitrate": MP3_BITRATE,
            "frame_rate": OUTPUT_FRAME_RATE,
            "channels": OUTPUT_CHANNELS,
            "sample_width": OUTPUT_SAMPLE_WIDTH,
            "normalization_headroom_db": NORMALIZE_HEADROOM_DB,
        },
        "unit_body_sequence": [
            "japanese_slow_varied_voices",
            "english_close_translation",
            "japanese_final_varied_voices",
        ],
    }


def audiobook_settings() -> Dict:
    return {
        "container": "m4b",
        "ffmpeg_muxer": "ipod",
        "audio_codec": AUDIOBOOK_AAC_CODEC,
        "audio_profile": AUDIOBOOK_AAC_PROFILE,
        "bitrate": AUDIOBOOK_AAC_BITRATE,
        "frame_rate": AUDIOBOOK_FRAME_RATE,
        "channels": AUDIOBOOK_CHANNELS,
        "chapter_timebase": AUDIOBOOK_CHAPTER_TIMEBASE,
        "chapter_unit": "complete learning unit",
        "shuffle_seed": SHUFFLE_SEED,
        "minimum_volume_hours": MIN_VOLUME_HOURS,
        "target_volume_hours": TARGET_VOLUME_HOURS,
        "maximum_volume_hours": MAX_VOLUME_HOURS,
        "maximum_chapters_per_volume": MAX_CHAPTERS_PER_VOLUME,
        "numbering": "global shuffled playback position spoken with 番目",
        "html_companion": {
            "version": HTML_COMPANION_VERSION,
            "format": "self-contained searchable text index per volume",
        },
        "title": AUDIOBOOK_TITLE,
    }


def new_progress(input_hash: str, total_units: int) -> Dict:
    return {
        "version": 3,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "input_sha256": input_hash,
        "settings": progress_settings(),
        "total_units": total_units,
        "completed_units": {},
        "failed_units": {},
    }


def load_or_create_progress(
    path: Path,
    input_hash: str,
    total_units: int,
) -> Dict:
    if not path.exists():
        return new_progress(input_hash, total_units)

    with path.open("r", encoding="utf-8") as file:
        progress = json.load(file)

    if progress.get("input_sha256") != input_hash:
        raise ValueError(
            "The V3 audio progress file belongs to different input data. "
            "Rename the V3 progress, manifest, and output folders before "
            "starting audio for changed content."
        )
    if progress.get("settings") != progress_settings():
        raise ValueError(
            "Unit body audio settings changed since V3 progress was created. "
            "Restore the settings or rename the V3 progress/output files."
        )
    if int(progress.get("total_units", 0)) != total_units:
        raise ValueError("The unit count no longer matches V3 audio progress")

    progress.setdefault("completed_units", {})
    progress.setdefault("failed_units", {})
    return progress


def new_manifest(input_hash: str) -> Dict:
    return {
        "version": 3,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "input_sha256": input_hash,
        "unit_settings_sha256": sha256_json(progress_settings()),
        "units": {},
        "audiobooks": {},
    }


def load_or_create_manifest(path: Path, input_hash: str) -> Dict:
    if not path.exists():
        return new_manifest(input_hash)

    try:
        with path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except Exception:
        return new_manifest(input_hash)

    if (
        manifest.get("input_sha256") != input_hash
        or manifest.get("unit_settings_sha256")
        != sha256_json(progress_settings())
    ):
        return new_manifest(input_hash)

    manifest.setdefault("units", {})
    manifest.setdefault("audiobooks", {})
    return manifest


class EdgeTTSProvider:
    name = "edge-tts"

    def __init__(self) -> None:
        self.last_error = ""

    async def _generate_async(
        self,
        item: Dict,
        output_file: Path,
    ) -> bool:
        try:
            communicate = edge_tts.Communicate(
                text=item["text"],
                voice=item["voice"],
                rate=item["rate"],
            )
            await communicate.save(str(output_file))
            return audio_file_is_valid(output_file)
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            return False

    def generate(
        self,
        item: Dict,
        output_file: Path,
    ) -> bool:
        self.last_error = ""
        result = {"ok": False}

        def runner() -> None:
            try:
                result["ok"] = asyncio.run(
                    self._generate_async(item, output_file)
                )
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                result["ok"] = False

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        return bool(result["ok"])


class TTSCache:
    """Content-addressed Edge TTS cache."""

    def __init__(self, cache_dir: Path, provider) -> None:
        self.cache_dir = cache_dir
        self.provider = provider
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.generated_count = 0
        self.reused_count = 0

    def cache_spec(self, item: Dict) -> Dict:
        spec = {
            "provider": item["provider"],
            "text": item["text"],
            "voice": item["voice"],
            "rate": item["rate"],
        }
        return spec

    def cache_key(self, item: Dict) -> str:
        return sha256_json(self.cache_spec(item))

    def cache_path(self, item: Dict) -> Path:
        return self.cache_dir / f"{self.cache_key(item)}.mp3"

    def get_or_create(self, item: Dict) -> Path:
        cache_path = self.cache_path(item)
        if audio_file_is_valid(cache_path, decode=True):
            self.reused_count += 1
            return cache_path

        safe_unlink(cache_path)
        raw_path = cache_path.with_name(cache_path.stem + ".raw.tmp.mp3")
        if audio_file_is_valid(raw_path, decode=True):
            os.replace(str(raw_path), str(cache_path))
            self.reused_count += 1
            return cache_path

        for attempt in range(1, MAX_TTS_RETRIES + 1):
            safe_unlink(raw_path)
            ok = self.provider.generate(item, raw_path)
            if ok and audio_file_is_valid(raw_path, decode=True):
                os.replace(str(raw_path), str(cache_path))
                self.generated_count += 1
                return cache_path

            if attempt < MAX_TTS_RETRIES:
                sleep_seconds = (
                    RETRY_BASE_SLEEP * (2 ** (attempt - 1))
                    + random.uniform(0, RETRY_JITTER)
                )
                time.sleep(sleep_seconds)

        safe_unlink(raw_path)
        detail = str(getattr(self.provider, "last_error", "")).strip()
        raise RuntimeError(
            "TTS failed after retries for voice "
            f"{item['voice']}: {item['text'][:120]}"
            + (f" | {detail}" if detail else "")
        )


class PartRenderer:
    def __init__(self, tts_cache: TTSCache) -> None:
        self.tts_cache = tts_cache

    def build_part(self, part: Dict, output_path: Path) -> None:
        combined = AudioSegment.empty()
        for item in part["components"]:
            cache_path = self.tts_cache.get_or_create(item)
            combined += load_audio(cache_path)
            pause_after_ms = int(item.get("pause_after_ms", 0))
            if pause_after_ms > 0:
                combined += make_silence(pause_after_ms)

        end_silence_ms = int(part.get("end_silence_ms", 0))
        if end_silence_ms > 0:
            combined += make_silence(end_silence_ms)

        if not atomic_export_audio(combined, output_path, normalize_audio=True):
            raise RuntimeError(f"Could not create audio part: {output_path}")


class UnitRenderer:
    def __init__(self, tts_cache: TTSCache) -> None:
        self.part_renderer = PartRenderer(tts_cache)

    def render(self, plan: Dict, trust_existing_parts: bool) -> Dict:
        part_metadata: List[Dict] = []
        duration_ms = 0

        for part in plan["parts"]:
            output_path = UNIT_AUDIO_DIR / part["output_file"]
            if not (
                trust_existing_parts
                and audio_file_is_valid(output_path, decode=True)
            ):
                self.part_renderer.build_part(part, output_path)

            if not audio_file_is_valid(output_path, decode=True):
                raise RuntimeError(f"Invalid unit part file: {output_path}")

            part_duration = len(load_audio(output_path))
            duration_ms += part_duration
            part_metadata.append(
                {
                    "file": output_path.name,
                    "size_bytes": output_path.stat().st_size,
                    "duration_ms": part_duration,
                    "sha256": sha256_file(output_path),
                }
            )

        return {
            "record_hash": plan["record_hash"],
            "part_files": part_metadata,
            "body_duration_ms": duration_ms,
        }


def part_metadata_by_name(entry: Dict) -> Dict[str, Dict]:
    normalized: Dict[str, Dict] = {}
    raw_parts = entry.get("part_files", [])
    if not isinstance(raw_parts, list):
        return normalized

    for item in raw_parts:
        if isinstance(item, dict):
            filename = str(item.get("file", "")).strip()
            if filename:
                normalized[filename] = item
    return normalized


def plan_part_files_are_valid(
    plan: Dict,
    entry: Optional[Dict] = None,
    decode: bool = False,
) -> bool:
    metadata = part_metadata_by_name(entry or {})
    for part in plan["parts"]:
        filename = part["output_file"]
        path = UNIT_AUDIO_DIR / filename
        expected_size = metadata.get(filename, {}).get("size_bytes")
        try:
            expected_size_int = int(expected_size)
        except (TypeError, ValueError):
            expected_size_int = None
        if not audio_file_is_valid(
            path,
            decode=decode,
            expected_size=expected_size_int,
        ):
            return False
    return True


def completed_entry_is_valid(
    unit_id: str,
    plan: Dict,
    progress: Dict,
) -> bool:
    entry = progress.get("completed_units", {}).get(unit_id)
    if not isinstance(entry, dict):
        return False
    if entry.get("record_hash") != plan["record_hash"]:
        return False
    return plan_part_files_are_valid(plan, entry=entry, decode=False)


def trusted_partial_entry_matches(
    unit_id: str,
    plan: Dict,
    progress: Dict,
) -> bool:
    completed = progress.get("completed_units", {}).get(unit_id, {})
    failed = progress.get("failed_units", {}).get(unit_id, {})
    return (
        completed.get("record_hash") == plan["record_hash"]
        or failed.get("record_hash") == plan["record_hash"]
    )


def remove_obsolete_unit_files(plans: Dict[str, Dict]) -> int:
    expected = {
        part["output_file"]
        for plan in plans.values()
        for part in plan["parts"]
    }
    removed = 0
    for path in UNIT_AUDIO_DIR.glob("unit_*.mp3"):
        if path.name in expected:
            continue
        try:
            path.unlink()
            removed += 1
        except Exception:
            pass
    return removed


def remove_obsolete_number_files(total_units: int) -> int:
    expected = {
        build_number_part(number)["output_file"]
        for number in range(1, total_units + 1)
    }
    removed = 0
    for path in UNIT_NUMBER_AUDIO_DIR.glob("unit_number_*.mp3"):
        if path.name in expected:
            continue
        try:
            path.unlink()
            removed += 1
        except Exception:
            pass
    return removed


def compact_plan_for_manifest(plan: Dict, tts_cache: TTSCache) -> List[Dict]:
    output = []
    for part in plan["parts"]:
        components = []
        for item in part["components"]:
            components.append(
                {
                    "language": item["language"],
                    "provider": item["provider"],
                    "voice": item["voice"],
                    "rate": item["rate"],
                    "text": item["text"],
                    "pause_after_ms": item["pause_after_ms"],
                    "tts_cache_file": tts_cache.cache_path(item).name,
                }
            )
        output.append(
            {
                "number": part["number"],
                "name": part["name"],
                "output_file": part["output_file"],
                "end_silence_ms": part["end_silence_ms"],
                "components": components,
            }
        )
    return output


def update_unit_manifest(
    manifest: Dict,
    unit_id: str,
    row: Dict,
    plan: Dict,
    status: str,
    tts_cache: TTSCache,
    result: Optional[Dict] = None,
    error: str = "",
) -> None:
    entry = {
        "status": status,
        "updated_at": utc_now(),
        "record_hash": plan["record_hash"],
        "japanese": row["japanese"],
        "english": row["english"],
        "primary_source_note_number": row[
            "primary_source_note_number"
        ],
        "supporting_source_note_numbers": row.get(
            "supporting_source_note_numbers", []
        ),
        "parts": compact_plan_for_manifest(plan, tts_cache),
    }
    if result:
        entry["body_duration_ms"] = result.get("body_duration_ms", 0)
        entry["part_files"] = result.get("part_files", [])
    if error:
        entry["last_error"] = error
    manifest["units"][unit_id] = entry
    manifest["updated_at"] = utc_now()


def save_progress_and_manifest(progress: Dict, manifest: Dict) -> None:
    progress["updated_at"] = utc_now()
    manifest["updated_at"] = utc_now()
    atomic_save_json(PROGRESS_PATH, progress)
    atomic_save_json(MANIFEST_PATH, manifest)


def ensure_number_parts(
    total_units: int,
    tts_cache: TTSCache,
) -> Tuple[Dict[int, Dict], Dict[str, object]]:
    """Create/reuse global shuffled-playback number clips."""
    renderer = PartRenderer(tts_cache)
    metadata: Dict[int, Dict] = {}
    UNIT_NUMBER_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    phase_start = time.perf_counter()
    built_files = 0
    reused_files = 0

    print("\nPLAYBACK NUMBER AUDIO")
    print("-" * 64)
    for playback_number in range(1, total_units + 1):
        part = build_number_part(playback_number)
        path = UNIT_NUMBER_AUDIO_DIR / part["output_file"]
        already_ready = audio_file_is_valid(path, decode=True)
        if not already_ready:
            renderer.build_part(part, path)
            built_files += 1
        else:
            reused_files += 1
        if not audio_file_is_valid(path, decode=True):
            raise RuntimeError(f"Invalid playback number audio: {path}")

        metadata[playback_number] = {
            "file": path.name,
            "path": path,
            "size_bytes": path.stat().st_size,
            "duration_ms": len(load_audio(path)),
            "sha256": sha256_file(path),
            "spoken_text": spoken_unit_number(playback_number),
            "voice": part["components"][0]["voice"],
        }

        if (
            playback_number == 1
            or playback_number % NUMBER_CLIP_PROGRESS_EVERY == 0
            or playback_number == total_units
        ):
            elapsed = time.perf_counter() - phase_start
            average_seconds = elapsed / playback_number
            remaining = total_units - playback_number
            now = local_now()
            print(
                f"  Ready: {playback_number:,}/{total_units:,} "
                "number clips | "
                f"{remaining:,} left | "
                f"elapsed {format_duration(elapsed)} | "
                f"ETA {format_eta(average_seconds, remaining, now)}",
                flush=True,
            )

    elapsed = time.perf_counter() - phase_start
    return metadata, {
        "total_clips": total_units,
        "built_files": built_files,
        "reused_files": reused_files,
        "elapsed_seconds": round(elapsed, 2),
        "average_seconds_per_clip": round(
            elapsed / total_units if total_units else 0.0,
            4,
        ),
    }


def stable_shuffled_unit_ids(
    unit_ids: Sequence[str],
    units: Dict[str, Dict],
) -> List[str]:
    """Deterministically shuffle and reduce same-primary adjacency."""
    ordered = sorted(
        unit_ids,
        key=lambda unit_id: hashlib.sha256(
            f"{SHUFFLE_SEED}:{unit_id}".encode("utf-8")
        ).hexdigest(),
    )

    # The hash order is the random basis. A light deterministic repair swaps
    # only adjacent units with the same primary note. This remains efficient
    # for large decks and leaves the order otherwise untouched.
    for index in range(1, len(ordered)):
        previous_primary = int(
            units[ordered[index - 1]]["primary_source_note_number"]
        )
        current_primary = int(
            units[ordered[index]]["primary_source_note_number"]
        )
        if current_primary != previous_primary:
            continue

        swap_index = None
        for candidate_index in range(index + 1, len(ordered)):
            candidate_primary = int(
                units[ordered[candidate_index]][
                    "primary_source_note_number"
                ]
            )
            if candidate_primary != previous_primary:
                swap_index = candidate_index
                break

        if swap_index is not None:
            ordered[index], ordered[swap_index] = (
                ordered[swap_index],
                ordered[index],
            )

    return ordered


def completed_part_duration_ms(path: Path, saved: Dict) -> int:
    try:
        duration = int(saved.get("duration_ms", 0))
    except (TypeError, ValueError):
        duration = 0
    if duration >= MIN_AUDIO_DURATION_MS:
        return duration
    if not audio_file_is_valid(path, decode=True):
        raise RuntimeError(f"Cannot measure invalid unit part: {path}")
    duration = len(load_audio(path))
    if duration < MIN_AUDIO_DURATION_MS:
        raise RuntimeError(f"Unit part has no usable duration: {path}")
    return duration


def build_chapter_entries(
    shuffled_ids: Sequence[str],
    units: Dict[str, Dict],
    plans: Dict[str, Dict],
    progress: Dict,
    number_metadata: Dict[int, Dict],
) -> List[Dict]:
    chapters: List[Dict] = []

    for playback_number, unit_id in enumerate(shuffled_ids, 1):
        plan = plans[unit_id]
        if not completed_entry_is_valid(unit_id, plan, progress):
            raise RuntimeError(
                f"Unit {unit_id} is incomplete; M4B assembly requires all "
                "units to be complete"
            )

        entry = progress["completed_units"][unit_id]
        saved_parts = part_metadata_by_name(entry)
        sources: List[Dict] = []
        body_duration_ms = 0

        number_info = number_metadata[playback_number]
        sources.append(
            {
                "kind": "number",
                "file": number_info["file"],
                "path": number_info["path"],
                "size_bytes": number_info["size_bytes"],
                "duration_ms": number_info["duration_ms"],
                "sha256": number_info["sha256"],
            }
        )

        updated_part_metadata = []
        for part in plan["parts"]:
            filename = part["output_file"]
            path = UNIT_AUDIO_DIR / filename
            saved = dict(saved_parts.get(filename, {}))
            duration_ms = completed_part_duration_ms(path, saved)
            body_duration_ms += duration_ms
            current = {
                **saved,
                "file": filename,
                "size_bytes": path.stat().st_size,
                "duration_ms": duration_ms,
            }
            if not current.get("sha256"):
                current["sha256"] = sha256_file(path)
            updated_part_metadata.append(current)
            sources.append(
                {
                    "kind": "unit_body",
                    "file": filename,
                    "path": path,
                    "size_bytes": current["size_bytes"],
                    "duration_ms": duration_ms,
                    "sha256": current["sha256"],
                    "record_hash": plan["record_hash"],
                }
            )

        entry["part_files"] = updated_part_metadata
        entry["body_duration_ms"] = body_duration_ms
        total_duration_ms = number_info["duration_ms"] + body_duration_ms
        row = units[unit_id]
        chapters.append(
            {
                "playback_number": playback_number,
                "unit_id": unit_id,
                "title": f"例文{playback_number:05d}",
                "japanese": row["japanese"],
                "english": row["english"],
                "primary_source_note_number": row[
                    "primary_source_note_number"
                ],
                "supporting_source_note_numbers": row.get(
                    "supporting_source_note_numbers", []
                ),
                "duration_ms": total_duration_ms,
                "number_duration_ms": number_info["duration_ms"],
                "body_duration_ms": body_duration_ms,
                "sources": sources,
            }
        )

    return chapters


def split_chapters_into_volumes(chapters: Sequence[Dict]) -> List[List[Dict]]:
    min_ms = int(MIN_VOLUME_HOURS * 60 * 60 * 1000)
    target_ms = int(TARGET_VOLUME_HOURS * 60 * 60 * 1000)
    max_ms = int(MAX_VOLUME_HOURS * 60 * 60 * 1000)

    volumes: List[List[Dict]] = []
    current: List[Dict] = []
    current_ms = 0

    for chapter in chapters:
        duration_ms = int(chapter["duration_ms"])
        should_cut = bool(current) and (
            current_ms >= target_ms
            or current_ms + duration_ms > max_ms
            or len(current) >= MAX_CHAPTERS_PER_VOLUME
        )
        if should_cut:
            volumes.append(current)
            current = []
            current_ms = 0

        current.append(chapter)
        current_ms += duration_ms

    if current:
        volumes.append(current)

    # Rebalance only the short final volume while preserving global order.
    if len(volumes) >= 2:
        previous = volumes[-2]
        final = volumes[-1]

        def duration(items: Sequence[Dict]) -> int:
            return sum(int(item["duration_ms"]) for item in items)

        while (
            final
            and len(final) < MAX_CHAPTERS_PER_VOLUME
            and duration(final) < min_ms
            and previous
        ):
            candidate = previous[-1]
            remaining_previous = duration(previous) - int(
                candidate["duration_ms"]
            )
            if remaining_previous < min_ms:
                break
            final.insert(0, previous.pop())

    return volumes


def ffmetadata_escape(value: str) -> str:
    text = str(value).replace("\\", "\\\\")
    for character in ("=", ";", "#"):
        text = text.replace(character, "\\" + character)
    return text.replace("\r", " ").replace("\n", " ")


def ffconcat_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def run_checked_process(command: Sequence[str], description: str) -> str:
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        if len(details) > 4000:
            details = details[-4000:]
        raise RuntimeError(
            f"{description} failed with exit code "
            f"{completed.returncode}: {details or 'no diagnostic output'}"
        )
    return completed.stdout


def build_volume_plan(
    volume_number: int,
    total_volumes: int,
    chapters: Sequence[Dict],
    input_hash: str,
) -> Dict:
    cursor_ms = 0
    planned_chapters: List[Dict] = []
    concat_sources: List[Dict] = []
    fingerprints: List[Dict] = []

    for chapter in chapters:
        start_ms = cursor_ms
        end_ms = start_ms + int(chapter["duration_ms"])
        planned = {
            **chapter,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
        planned_chapters.append(planned)
        cursor_ms = end_ms

        for source in chapter["sources"]:
            concat_sources.append(source)
            fingerprints.append(
                {
                    "kind": source["kind"],
                    "file": source["file"],
                    "size_bytes": source["size_bytes"],
                    "duration_ms": source["duration_ms"],
                    "sha256": source.get("sha256", ""),
                    "record_hash": source.get("record_hash", ""),
                }
            )

    width = max(2, len(str(total_volumes)))
    file_name = (
        f"{AUDIOBOOK_BASENAME}_{volume_number:0{width}d}.m4b"
    )
    volume_title = (
        f"{AUDIOBOOK_TITLE}・"
        f"第{volume_number}巻（全{total_volumes}巻）"
    )
    fingerprint = {
        "input_sha256": input_hash,
        "settings": audiobook_settings(),
        "volume_number": volume_number,
        "total_volumes": total_volumes,
        "file_name": file_name,
        "source_files": fingerprints,
        "chapters": [
            {
                "playback_number": item["playback_number"],
                "unit_id": item["unit_id"],
                "title": item["title"],
                "start_ms": item["start_ms"],
                "end_ms": item["end_ms"],
            }
            for item in planned_chapters
        ],
        "total_duration_ms": cursor_ms,
    }

    return {
        "assembly_hash": sha256_json(fingerprint),
        "volume_number": volume_number,
        "total_volumes": total_volumes,
        "file_name": file_name,
        "output_path": AUDIOBOOK_OUTPUT_DIR / file_name,
        "title": volume_title,
        "concat_sources": concat_sources,
        "chapters": planned_chapters,
        "total_duration_ms": cursor_ms,
        "settings": audiobook_settings(),
    }


def format_chapter_time(milliseconds: int) -> str:
    total_seconds = max(0, int(milliseconds) // 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def html_companion_text(volume_plan: Dict) -> str:
    """Return a self-contained, searchable mobile companion page."""
    title = html.escape(str(volume_plan["title"]))
    audio_file = html.escape(str(volume_plan["file_name"]), quote=True)
    chapters = volume_plan["chapters"]
    first_number = int(chapters[0]["playback_number"])
    last_number = int(chapters[-1]["playback_number"])
    duration = format_duration(volume_plan["total_duration_ms"] / 1000)

    cards: List[str] = []
    for chapter in chapters:
        number = int(chapter["playback_number"])
        japanese = str(chapter["japanese"])
        english_text = str(chapter["english"])
        primary = int(chapter["primary_source_note_number"])
        supporting = [
            int(value)
            for value in chapter.get("supporting_source_note_numbers", [])
        ]
        source_text = f"Source note {primary}"
        if supporting:
            source_text += "; supporting " + ", ".join(
                str(value) for value in supporting
            )
        search_text = f"{number} {japanese} {english_text}".casefold()
        cards.append(
            "\n".join(
                [
                    (
                        f'<article class="sentence" id="sentence-{number:05d}" '
                        f'data-number="{number}" '
                        f'data-search="{html.escape(search_text, quote=True)}">'
                    ),
                    '  <div class="sentence-meta">',
                    (
                        f'    <a href="#sentence-{number:05d}">'
                        f'#{number:05d}</a>'
                    ),
                    (
                        "    <span>"
                        + format_chapter_time(int(chapter["start_ms"]))
                        + "</span>"
                    ),
                    f"    <span>{html.escape(source_text)}</span>",
                    "  </div>",
                    (
                        '  <p class="japanese" lang="ja">'
                        + html.escape(japanese)
                        + "</p>"
                    ),
                    (
                        '  <p class="english" lang="en">'
                        + html.escape(english_text)
                        + "</p>"
                    ),
                    "</article>",
                ]
            )
        )

    document_start = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>__TITLE__ — Text companion</title>
  <style>
    :root { color-scheme: light dark; font-family: system-ui, -apple-system,
      "Segoe UI", sans-serif; --bg: #f5f6f8; --card: #fff; --text: #17202a;
      --muted: #68717d; --line: #d9dde3; --accent: #315dca; }
    @media (prefers-color-scheme: dark) {
      :root { --bg: #101318; --card: #191e26; --text: #edf1f7;
        --muted: #a9b2bf; --line: #343c48; --accent: #91b2ff; }
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    header { position: sticky; top: 0; z-index: 2; padding: 1rem;
      background: var(--bg);
      background: color-mix(in srgb, var(--bg) 92%, transparent);
      border-bottom: 1px solid var(--line); backdrop-filter: blur(12px); }
    .header-inner, main { width: min(52rem, 100%); margin: auto; }
    h1 { margin: 0 0 .35rem; font-size: clamp(1.1rem, 4vw, 1.5rem); }
    .summary { margin: 0 0 .75rem; color: var(--muted); font-size: .9rem; }
    .tools { display: grid; grid-template-columns: 1fr auto; gap: .55rem; }
    input, button { min-height: 2.75rem; border: 1px solid var(--line);
      border-radius: .7rem; font: inherit; }
    input { width: 100%; padding: .65rem .8rem; background: var(--card);
      color: var(--text); }
    button { padding: .55rem .85rem; background: var(--accent); color: white;
      border-color: transparent; }
    #result { min-height: 1.2rem; margin: .45rem 0 0; color: var(--muted);
      font-size: .85rem; }
    main { padding: .8rem; }
    .sentence { scroll-margin-top: 10rem; margin: 0 0 .75rem; padding: 1rem;
      border: 1px solid var(--line); border-radius: .9rem;
      background: var(--card); box-shadow: 0 1px 2px rgb(0 0 0 / .05); }
    .sentence:target { outline: 3px solid var(--accent); }
    .sentence-meta { display: flex; flex-wrap: wrap; gap: .45rem .8rem;
      color: var(--muted); font-size: .8rem; }
    .sentence-meta a, .audio-link { color: var(--accent); font-weight: 700; }
    .japanese { margin: .65rem 0 .4rem; font-family: "Hiragino Sans",
      "Yu Gothic", "Noto Sans JP", sans-serif; font-size: 1.35rem;
      line-height: 1.65; }
    .english { margin: 0; color: var(--muted); font-size: 1rem;
      line-height: 1.5; }
    [hidden] { display: none !important; }
    noscript { display: block; margin-top: .5rem; color: var(--muted); }
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>__TITLE__</h1>
      <p class="summary">Sentences __FIRST__–__LAST__ · __COUNT__ chapters ·
        __DURATION__ · <a class="audio-link" href="__AUDIO__">open M4B</a></p>
      <div class="tools">
        <input id="search" type="search" inputmode="search"
          placeholder="Sentence number, Japanese, or English"
          aria-label="Filter sentences">
        <button id="clear" type="button">Clear</button>
      </div>
      <p id="result" aria-live="polite">__COUNT__ sentences</p>
      <noscript>Search requires JavaScript; the complete list is below.</noscript>
    </div>
  </header>
  <main id="sentences">
"""
    document_start = (
        document_start.replace("__TITLE__", title)
        .replace("__FIRST__", f"{first_number:05d}")
        .replace("__LAST__", f"{last_number:05d}")
        .replace("__COUNT__", f"{len(chapters):,}")
        .replace("__DURATION__", html.escape(duration))
        .replace("__AUDIO__", audio_file)
    )
    document_end = """  </main>
  <script>
    const search = document.getElementById("search");
    const clear = document.getElementById("clear");
    const result = document.getElementById("result");
    const cards = [...document.querySelectorAll(".sentence")];
    function applyFilter() {
      const query = search.value.trim().toLocaleLowerCase();
      const numeric = /^#?\\d+$/.test(query) ? query.replace("#", "") : null;
      let visible = 0;
      for (const card of cards) {
        const match = !query || (numeric !== null
          ? card.dataset.number === String(Number(numeric))
          : card.dataset.search.toLocaleLowerCase().includes(query));
        card.hidden = !match;
        if (match) visible += 1;
      }
      result.textContent = `${visible.toLocaleString()} sentence${visible === 1 ? "" : "s"}`;
      if (numeric !== null && visible === 1) {
        const target = cards.find(card => !card.hidden);
        if (target) target.scrollIntoView({ block: "center" });
      }
    }
    search.addEventListener("input", applyFilter);
    clear.addEventListener("click", () => {
      search.value = "";
      applyFilter();
      search.focus();
    });
  </script>
</body>
</html>
"""
    return document_start + "\n".join(cards) + "\n" + document_end


def write_html_companion(volume_plan: Dict) -> Dict:
    output_path = Path(volume_plan["output_path"]).with_suffix(".html")
    content = html_companion_text(volume_plan)
    atomic_write_text(output_path, content)
    return {
        "version": HTML_COMPANION_VERSION,
        "file": output_path.name,
        "relative_path": str(
            Path(OUTPUT_ROOT_DIR_NAME)
            / AUDIOBOOK_OUTPUT_DIR_NAME
            / output_path.name
        ),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def instruction_paths(volume_number: int) -> Tuple[Path, Path]:
    return (
        BUILD_TEMP_AUDIO_DIR
        / f"volume_{volume_number:03d}_concat.ffconcat",
        BUILD_TEMP_AUDIO_DIR
        / f"volume_{volume_number:03d}_chapters.ffmetadata",
    )


def write_volume_instruction_files(volume_plan: Dict) -> Tuple[Path, Path]:
    concat_path, metadata_path = instruction_paths(
        int(volume_plan["volume_number"])
    )
    concat_lines = ["ffconcat version 1.0"]
    for source in volume_plan["concat_sources"]:
        concat_lines.append(
            f"file {ffconcat_quote(Path(source['path']).resolve().as_posix())}"
        )
        concat_lines.append(
            f"duration {source['duration_ms'] / 1000.0:.6f}"
        )

    metadata_lines = [
        ";FFMETADATA1",
        f"title={ffmetadata_escape(volume_plan['title'])}",
        f"album={ffmetadata_escape(AUDIOBOOK_TITLE)}",
        f"track={volume_plan['volume_number']}/{volume_plan['total_volumes']}",
        "genre=Audiobook",
        f"encoder={ffmetadata_escape('Generator ' + SCRIPT_VERSION)}",
    ]
    for chapter in volume_plan["chapters"]:
        metadata_lines.extend(
            [
                "",
                "[CHAPTER]",
                f"TIMEBASE=1/{AUDIOBOOK_CHAPTER_TIMEBASE}",
                f"START={chapter['start_ms']}",
                f"END={chapter['end_ms']}",
                f"title={ffmetadata_escape(chapter['title'])}",
            ]
        )

    atomic_write_text(concat_path, "\n".join(concat_lines) + "\n")
    atomic_write_text(metadata_path, "\n".join(metadata_lines) + "\n")
    return concat_path, metadata_path


def find_ffmpeg() -> Optional[str]:
    candidates: List[str] = []
    if AudioSegment is not None:
        converter = getattr(AudioSegment, "converter", "")
        if converter:
            candidates.append(str(converter))
    candidates.extend(["ffmpeg", "ffmpeg.exe"])

    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)
    return None


def find_ffprobe() -> Optional[str]:
    candidates: List[str] = []
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        binary = Path(ffmpeg_path)
        sibling = "ffprobe.exe" if binary.name.lower().endswith(".exe") else "ffprobe"
        candidates.append(str(binary.with_name(sibling)))
    candidates.extend(["ffprobe", "ffprobe.exe"])

    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)
    return None


def probe_audiobook(path: Path) -> Dict:
    ffprobe_path = find_ffprobe()
    if not ffprobe_path:
        raise RuntimeError("FFprobe was not found")
    output = run_checked_process(
        [
            ffprobe_path,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(path),
        ],
        "FFprobe audiobook verification",
    )
    return json.loads(output)


def validate_volume_file(
    path: Path,
    volume_plan: Dict,
    expected_size: Optional[int] = None,
) -> Tuple[bool, str, Optional[Dict]]:
    try:
        if not path.exists() or not path.is_file():
            return False, "output file does not exist", None
        if path.stat().st_size < MIN_M4B_BYTES:
            return False, "output file is too small", None
        if expected_size is not None and path.stat().st_size != expected_size:
            return False, "output size differs from manifest", None

        probe = probe_audiobook(path)
        audio_streams = [
            stream
            for stream in probe.get("streams", [])
            if stream.get("codec_type") == "audio"
        ]
        if len(audio_streams) != 1:
            return False, "expected exactly one audio stream", probe

        audio = audio_streams[0]
        if audio.get("codec_name") != AUDIOBOOK_AAC_CODEC:
            return False, "audio stream is not AAC", probe
        if int(audio.get("channels", 0)) != AUDIOBOOK_CHANNELS:
            return False, "audio stream is not mono", probe
        if int(audio.get("sample_rate", 0)) != AUDIOBOOK_FRAME_RATE:
            return False, "unexpected sample rate", probe

        actual_chapters = probe.get("chapters", [])
        expected_chapters = volume_plan["chapters"]
        if len(actual_chapters) != len(expected_chapters):
            return False, "chapter count does not match", probe

        for actual, expected in zip(actual_chapters, expected_chapters):
            actual_title = str(actual.get("tags", {}).get("title", ""))
            if actual_title != expected["title"]:
                return False, "chapter title or order differs", probe

            actual_start = float(actual.get("start_time", 0.0))
            actual_end = float(actual.get("end_time", 0.0))
            expected_start = expected["start_ms"] / 1000.0
            expected_end = expected["end_ms"] / 1000.0
            if abs(actual_start - expected_start) > CHAPTER_TIME_TOLERANCE_SECONDS:
                return False, "chapter start differs", probe
            if abs(actual_end - expected_end) > CHAPTER_TIME_TOLERANCE_SECONDS:
                return False, "chapter end differs", probe
            if actual_end <= actual_start:
                return False, "chapter has non-positive duration", probe

        actual_duration = float(probe.get("format", {}).get("duration", 0.0))
        expected_duration = volume_plan["total_duration_ms"] / 1000.0
        if abs(actual_duration - expected_duration) > VOLUME_DURATION_TOLERANCE_SECONDS:
            return False, "volume duration differs", probe

        return True, "", probe
    except Exception as error:
        return False, f"{type(error).__name__}: {error}", None


def create_or_reuse_volume(
    volume_plan: Dict,
    saved_entry: Dict,
) -> Dict:
    output_path = Path(volume_plan["output_path"])
    try:
        saved_size = int(saved_entry.get("size_bytes", 0)) or None
    except (TypeError, ValueError):
        saved_size = None

    if saved_entry.get("assembly_hash") == volume_plan["assembly_hash"]:
        valid, error, probe = validate_volume_file(
            output_path,
            volume_plan,
            expected_size=saved_size,
        )
        if valid:
            return {
                **saved_entry,
                "status": "complete",
                "result": "reused",
            }
        print(f"  Existing file will be rebuilt: {error}")

    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg was not found for M4B assembly")
    if not find_ffprobe():
        raise RuntimeError("FFprobe was not found for M4B verification")

    concat_path, metadata_path = write_volume_instruction_files(volume_plan)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(
        output_path.stem + ".tmp" + output_path.suffix
    )
    safe_unlink(temporary_output)

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-f",
        "ffmetadata",
        "-i",
        str(metadata_path),
        "-map",
        "0:a:0",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-vn",
        "-c:a",
        AUDIOBOOK_AAC_CODEC,
        "-profile:a",
        AUDIOBOOK_AAC_PROFILE,
        "-b:a",
        AUDIOBOOK_AAC_BITRATE,
        "-ar",
        str(AUDIOBOOK_FRAME_RATE),
        "-ac",
        str(AUDIOBOOK_CHANNELS),
        "-metadata",
        f"title={volume_plan['title']}",
        "-metadata",
        f"album={AUDIOBOOK_TITLE}",
        "-metadata",
        "genre=Audiobook",
        "-metadata:s:a:0",
        "language=mul",
        "-movflags",
        "+faststart",
        "-f",
        "ipod",
        str(temporary_output),
    ]

    started = time.perf_counter()
    try:
        run_checked_process(command, "FFmpeg M4B volume assembly")
        valid, error, probe = validate_volume_file(
            temporary_output,
            volume_plan,
        )
        if not valid:
            raise RuntimeError(f"Finished M4B did not verify: {error}")

        os.replace(str(temporary_output), str(output_path))
        elapsed = time.perf_counter() - started
        return {
            "status": "complete",
            "result": "created",
            "updated_at": utc_now(),
            "assembly_hash": volume_plan["assembly_hash"],
            "volume_number": volume_plan["volume_number"],
            "total_volumes": volume_plan["total_volumes"],
            "file": output_path.name,
            "relative_path": str(
                Path(OUTPUT_ROOT_DIR_NAME)
                / AUDIOBOOK_OUTPUT_DIR_NAME
                / output_path.name
            ),
            "title": volume_plan["title"],
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "duration_ms": volume_plan["total_duration_ms"],
            "chapter_count": len(volume_plan["chapters"]),
            "first_playback_number": volume_plan["chapters"][0][
                "playback_number"
            ],
            "last_playback_number": volume_plan["chapters"][-1][
                "playback_number"
            ],
            "unit_ids": [
                chapter["unit_id"] for chapter in volume_plan["chapters"]
            ],
            "elapsed_seconds": round(elapsed, 2),
        }
    except Exception:
        safe_unlink(temporary_output)
        raise
    finally:
        safe_unlink(concat_path)
        safe_unlink(metadata_path)


def remove_stale_audiobook_outputs(expected_files: Sequence[str]) -> int:
    expected = set(expected_files)
    expected.update(str(Path(name).with_suffix(".html")) for name in expected_files)
    removed = 0
    for suffix in ("m4b", "html"):
        for path in AUDIOBOOK_OUTPUT_DIR.glob(
            f"{AUDIOBOOK_BASENAME}_*.{suffix}"
        ):
            if path.name in expected:
                continue
            try:
                path.unlink()
                removed += 1
            except Exception:
                pass
    return removed


def assemble_m4b_volumes(
    unit_ids: Sequence[str],
    units: Dict[str, Dict],
    plans: Dict[str, Dict],
    progress: Dict,
    manifest: Dict,
    provider,
    input_hash: str,
) -> Dict:
    """Build deterministic shuffled multi-volume M4B output."""
    tts_cache = TTSCache(TTS_CACHE_DIR, provider)
    number_metadata, number_phase = ensure_number_parts(
        len(unit_ids),
        tts_cache,
    )
    remove_obsolete_number_files(len(unit_ids))

    shuffled_ids = stable_shuffled_unit_ids(unit_ids, units)
    chapters = build_chapter_entries(
        shuffled_ids,
        units,
        plans,
        progress,
        number_metadata,
    )
    volume_groups = split_chapters_into_volumes(chapters)
    total_volumes = len(volume_groups)
    volume_plans = [
        build_volume_plan(
            volume_number=index,
            total_volumes=total_volumes,
            chapters=group,
            input_hash=input_hash,
        )
        for index, group in enumerate(volume_groups, 1)
    ]

    layout_fingerprint = {
        "settings": audiobook_settings(),
        "shuffled_unit_ids": shuffled_ids,
        "volume_boundaries": [
            [chapter["unit_id"] for chapter in plan["chapters"]]
            for plan in volume_plans
        ],
    }
    layout_hash = sha256_json(layout_fingerprint)

    saved_root = manifest.get("audiobooks", {})
    saved_volumes = {}
    if isinstance(saved_root, dict):
        for item in saved_root.get("volumes", []):
            if isinstance(item, dict):
                saved_volumes[str(item.get("volume_number"))] = item

    print("\nM4B VOLUME ASSEMBLY")
    print("-" * 64)
    print(f"Shuffle seed: {SHUFFLE_SEED}")
    print(f"Volumes:      {total_volumes}")
    print(
        f"Encoding:     AAC-LC {AUDIOBOOK_AAC_BITRATE}, "
        f"{AUDIOBOOK_FRAME_RATE:,} Hz, mono"
    )

    output_entries = []
    for plan in volume_plans:
        volume_number = int(plan["volume_number"])
        duration_text = format_duration(plan["total_duration_ms"] / 1000)
        print(
            f"\n[{volume_number:02d}/{total_volumes:02d}] "
            f"{len(plan['chapters']):,} chapters | {duration_text}"
        )
        entry = create_or_reuse_volume(
            plan,
            saved_volumes.get(str(volume_number), {}),
        )
        entry["html_companion"] = write_html_companion(plan)
        output_entries.append(entry)
        print(
            f"  {entry['result'].capitalize()}: {entry['file']} | "
            f"{entry['size_bytes'] / (1024 * 1024):.1f} MiB"
        )
        print(f"  Text companion: {entry['html_companion']['file']}")

        manifest["audiobooks"] = {
            "status": "building",
            "updated_at": utc_now(),
            "layout_hash": layout_hash,
            "settings": audiobook_settings(),
            "total_units": len(unit_ids),
            "total_volumes": total_volumes,
            "shuffled_unit_ids": shuffled_ids,
            "volumes": output_entries,
            "number_audio_phase": number_phase,
        }
        save_progress_and_manifest(progress, manifest)

    removed_stale = remove_stale_audiobook_outputs(
        [plan["file_name"] for plan in volume_plans]
    )
    manifest["audiobooks"] = {
        "status": "complete",
        "updated_at": utc_now(),
        "layout_hash": layout_hash,
        "settings": audiobook_settings(),
        "total_units": len(unit_ids),
        "total_volumes": total_volumes,
        "total_duration_ms": sum(
            int(plan["total_duration_ms"]) for plan in volume_plans
        ),
        "shuffled_unit_ids": shuffled_ids,
        "volumes": output_entries,
        "removed_stale_files": removed_stale,
        "number_tts_generated": tts_cache.generated_count,
        "number_tts_reused": tts_cache.reused_count,
        "number_audio_phase": number_phase,
    }
    save_progress_and_manifest(progress, manifest)

    return manifest["audiobooks"]


def cleanup_stale_temporary_files() -> int:
    """Remove crash leftovers without touching resumable completed MP3 parts."""
    removed = 0
    if OUTPUT_ROOT_DIR.exists():
        for path in OUTPUT_ROOT_DIR.rglob("*.tmp"):
            if path.is_file():
                try:
                    path.unlink()
                    removed += 1
                except Exception:
                    pass
        for path in OUTPUT_ROOT_DIR.rglob("*.raw.tmp.mp3"):
            if path.is_file():
                try:
                    path.unlink()
                    removed += 1
                except Exception:
                    pass
        for path in AUDIOBOOK_OUTPUT_DIR.glob("*.tmp.m4b"):
            try:
                path.unlink()
                removed += 1
            except Exception:
                pass
    return removed


def cleanup_build_artifacts() -> Dict[str, int]:
    """Delete all generated MP3/cache/instruction files after final success."""
    result = {"files": 0, "bytes": 0}
    if not WORK_DIR.exists():
        return result

    for path in WORK_DIR.rglob("*"):
        if path.is_file():
            try:
                result["files"] += 1
                result["bytes"] += path.stat().st_size
            except Exception:
                pass

    shutil.rmtree(WORK_DIR, ignore_errors=False)
    return result


def verify_saved_audiobooks(
    manifest: Dict,
    total_units: int,
) -> Tuple[bool, str, Optional[Dict]]:
    """Validate final M4Bs and HTML without deleted intermediate MP3 files."""
    root = manifest.get("audiobooks", {})
    if not isinstance(root, dict) or root.get("status") != "complete":
        return False, "manifest has no complete audiobook build", None
    if root.get("settings") != audiobook_settings():
        return False, "audiobook settings have changed", None
    if int(root.get("total_units", 0)) != total_units:
        return False, "audiobook unit count differs", None

    volumes = root.get("volumes", [])
    try:
        expected_volume_count = int(root.get("total_volumes", 0))
    except (TypeError, ValueError):
        expected_volume_count = 0
    if not isinstance(volumes, list) or len(volumes) != expected_volume_count:
        return False, "audiobook volume list is incomplete", None

    for entry in volumes:
        if not isinstance(entry, dict) or entry.get("status") != "complete":
            return False, "a volume is not marked complete", None
        filename = str(entry.get("file", "")).strip()
        if not filename:
            return False, "a volume filename is missing", None
        path = AUDIOBOOK_OUTPUT_DIR / filename
        if not path.exists() or not path.is_file():
            return False, f"missing audiobook file: {filename}", None
        try:
            if path.stat().st_size != int(entry.get("size_bytes", -1)):
                return False, f"size mismatch for {filename}", None
        except (TypeError, ValueError):
            return False, f"invalid saved size for {filename}", None
        expected_hash = str(entry.get("sha256", ""))
        if not expected_hash or sha256_file(path) != expected_hash:
            return False, f"checksum mismatch for {filename}", None

        try:
            probe = probe_audiobook(path)
        except Exception as error:
            return (
                False,
                f"could not verify {filename}: {type(error).__name__}: {error}",
                None,
            )
        audio_streams = [
            stream
            for stream in probe.get("streams", [])
            if stream.get("codec_type") == "audio"
        ]
        if len(audio_streams) != 1:
            return False, f"invalid audio stream count in {filename}", None
        audio = audio_streams[0]
        if audio.get("codec_name") != AUDIOBOOK_AAC_CODEC:
            return False, f"non-AAC audio in {filename}", None
        if int(audio.get("channels", 0)) != AUDIOBOOK_CHANNELS:
            return False, f"wrong channel count in {filename}", None
        if int(audio.get("sample_rate", 0)) != AUDIOBOOK_FRAME_RATE:
            return False, f"wrong sample rate in {filename}", None
        if len(probe.get("chapters", [])) != int(entry.get("chapter_count", -1)):
            return False, f"chapter count mismatch in {filename}", None

        companion = entry.get("html_companion", {})
        if not isinstance(companion, dict):
            return False, f"HTML companion metadata missing for {filename}", None
        try:
            companion_version = int(companion.get("version", 0))
        except (TypeError, ValueError):
            companion_version = 0
        if companion_version != HTML_COMPANION_VERSION:
            return False, f"HTML companion version differs for {filename}", None
        companion_name = str(companion.get("file", "")).strip()
        expected_name = Path(filename).with_suffix(".html").name
        if companion_name != expected_name:
            return False, f"HTML companion filename differs for {filename}", None
        companion_path = AUDIOBOOK_OUTPUT_DIR / companion_name
        if not companion_path.exists() or not companion_path.is_file():
            return False, f"missing HTML companion: {companion_name}", None
        try:
            if companion_path.stat().st_size != int(
                companion.get("size_bytes", -1)
            ):
                return False, f"size mismatch for {companion_name}", None
        except (TypeError, ValueError):
            return False, f"invalid saved size for {companion_name}", None
        companion_hash = str(companion.get("sha256", ""))
        if not companion_hash or sha256_file(companion_path) != companion_hash:
            return False, f"checksum mismatch for {companion_name}", None

    return True, "", root


def validate_configuration() -> None:
    if UNIT_LIMIT is not None and (
        isinstance(UNIT_LIMIT, bool)
        or not isinstance(UNIT_LIMIT, int)
        or UNIT_LIMIT < 1
    ):
        raise ValueError("UNIT_LIMIT must be None or a positive integer")
    if not (0 < MIN_VOLUME_HOURS <= TARGET_VOLUME_HOURS <= MAX_VOLUME_HOURS):
        raise ValueError(
            "Volume hours must satisfy 0 < minimum <= target <= maximum"
        )
    if MAX_TTS_RETRIES < 1:
        raise ValueError("MAX_TTS_RETRIES must be at least 1")
    if (
        isinstance(NUMBER_CLIP_PROGRESS_EVERY, bool)
        or not isinstance(NUMBER_CLIP_PROGRESS_EVERY, int)
        or NUMBER_CLIP_PROGRESS_EVERY < 1
    ):
        raise ValueError(
            "NUMBER_CLIP_PROGRESS_EVERY must be a positive integer"
        )
    if not JAPANESE_VOICES:
        raise ValueError("JAPANESE_VOICES must contain at least one voice")
    if any(
        not isinstance(voice, str) or not voice.strip()
        for voice in JAPANESE_VOICES
    ):
        raise ValueError("Every JAPANESE_VOICES entry must be a non-empty name")
    if len(set(JAPANESE_VOICES)) != len(JAPANESE_VOICES):
        raise ValueError("JAPANESE_VOICES must not contain duplicates")


def validate_runtime_dependencies(require_edge_tts: bool = True) -> None:
    if AudioSegment is None or effects is None:
        raise ImportError("pydub is not installed. Run: pip install pydub")
    if require_edge_tts and edge_tts is None:
        raise ImportError("edge-tts is not installed. Run: pip install edge-tts")
    if not find_ffmpeg():
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and add it to PATH."
        )


def process_unfinished_units(
    units: Dict[str, Dict],
    unit_ids: Sequence[str],
    plans: Dict[str, Dict],
    progress: Dict,
    manifest: Dict,
    provider,
    unit_limit: Optional[int],
) -> Dict:
    unfinished_ids = [
        unit_id
        for unit_id in unit_ids
        if not completed_entry_is_valid(unit_id, plans[unit_id], progress)
    ]
    selected_ids = (
        unfinished_ids
        if unit_limit is None
        else unfinished_ids[:unit_limit]
    )

    total_units = len(unit_ids)
    already_complete = total_units - len(unfinished_ids)
    completed_running = already_complete
    pending_number_clips = count_pending_number_clips(total_units)
    components_per_unit = average_unit_tts_components(plans, unit_ids)

    print("\nRUN STATUS")
    print("-" * 64)
    print(f"Total units:           {total_units:,}")
    print(f"Already complete:      {already_complete:,}")
    print(f"Unfinished before run: {len(unfinished_ids):,}")
    print(f"Selected this run:     {len(selected_ids):,}")
    print(f"Number clips pending:  {pending_number_clips:,}")

    if not selected_ids:
        return {
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "elapsed_seconds": 0.0,
            "tts_generated": 0,
            "tts_reused": 0,
        }

    tts_cache = TTSCache(TTS_CACHE_DIR, provider)
    renderer = UnitRenderer(tts_cache)
    run_start = time.perf_counter()
    attempted_count = 0
    succeeded = 0
    failed_count = 0
    consecutive_failures = 0

    for run_index, unit_id in enumerate(selected_ids, 1):
        unit_start = time.perf_counter()
        plan = plans[unit_id]
        row = units[unit_id]

        print(
            f"\n[{run_index:05d}/{len(selected_ids):05d}] "
            f"Unit {format_unit_id(unit_id)} | "
            f"{completed_running:,}/{total_units:,} complete",
            flush=True,
        )

        trust_existing = trusted_partial_entry_matches(
            unit_id,
            plan,
            progress,
        )

        try:
            result = renderer.render(plan, trust_existing_parts=trust_existing)
            elapsed = time.perf_counter() - unit_start
            progress["completed_units"][unit_id] = {
                **result,
                "completed_at": utc_now(),
                "elapsed_seconds": round(elapsed, 2),
            }
            progress["failed_units"].pop(unit_id, None)
            succeeded += 1
            completed_running += 1
            consecutive_failures = 0

            update_unit_manifest(
                manifest,
                unit_id,
                row,
                plan,
                "complete",
                tts_cache,
                result=result,
            )
            now = local_now()
            print(
                f"  OK in {elapsed:.1f}s | "
                f"{format_local_time(now)}"
            )
        except Exception as error:
            elapsed = time.perf_counter() - unit_start
            failed_count += 1
            consecutive_failures += 1
            previous = progress["failed_units"].get(unit_id, {})
            try:
                previous_attempts = int(previous.get("attempts", 0))
            except (TypeError, ValueError):
                previous_attempts = 0

            error_text = f"{type(error).__name__}: {error}"
            progress["failed_units"][unit_id] = {
                "record_hash": plan["record_hash"],
                "attempts": previous_attempts + 1,
                "failed_at": utc_now(),
                "elapsed_seconds": round(elapsed, 2),
                "last_error": error_text,
            }
            update_unit_manifest(
                manifest,
                unit_id,
                row,
                plan,
                "failed",
                tts_cache,
                error=error_text,
            )
            print(f"  FAILED in {elapsed:.1f}s: {error_text}")

        save_progress_and_manifest(progress, manifest)
        attempted_count += 1

        elapsed_total = time.perf_counter() - run_start
        average_seconds = elapsed_total / attempted_count
        remaining_after = total_units - completed_running
        remaining_equivalents = estimated_remaining_unit_equivalents(
            remaining_after,
            pending_number_clips,
            components_per_unit,
        )
        now = local_now()
        print(
            f"  Progress: {completed_running:,}/{total_units:,} complete | "
            f"{remaining_after:,} left | "
            f"elapsed {format_duration(elapsed_total)} | "
            "ETA "
            f"{format_eta(average_seconds, remaining_equivalents, now)}"
        )

        if consecutive_failures >= STOP_AFTER_CONSECUTIVE_FAILURES:
            print(
                f"\nStopping after {consecutive_failures} consecutive "
                "unit failures. Rerun later to retry them."
            )
            break

        if BETWEEN_UNITS_SLEEP > 0:
            time.sleep(BETWEEN_UNITS_SLEEP)

    return {
        "attempted": attempted_count,
        "succeeded": succeeded,
        "failed": failed_count,
        "elapsed_seconds": time.perf_counter() - run_start,
        "tts_generated": tts_cache.generated_count,
        "tts_reused": tts_cache.reused_count,
    }


def main() -> bool:
    print("=" * 72)
    print("ANKI JAPANESE AUDIO-LEARNING GENERATOR V3")
    print(
        f"Version: {SCRIPT_VERSION} | Edge Japanese + Edge English | "
        "M4B + HTML"
    )
    print("=" * 72)

    print(f"Output folder:     {OUTPUT_ROOT_DIR}")
    print(f"Input:             {INPUT_JSON_PATH}")
    print(f"Progress:          {PROGRESS_PATH}")
    print(f"Manifest:          {MANIFEST_PATH}")
    print(f"Temporary work:    {WORK_DIR}")
    print(f"Audiobooks:        {AUDIOBOOK_OUTPUT_DIR}")
    print(f"Shuffle seed:      {SHUFFLE_SEED}")
    print(
        "Volume target:    "
        f"{TARGET_VOLUME_HOURS:g}h "
        f"({MIN_VOLUME_HOURS:g}-{MAX_VOLUME_HOURS:g}h normal range)"
    )
    print(
        "Run size:         "
        + (
            "FULL - every unfinished unit"
            if UNIT_LIMIT is None
            else f"LIMITED - next {UNIT_LIMIT} unfinished unit(s)"
        )
    )
    print("=" * 72)

    try:
        validate_configuration()
        OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
        AUDIOBOOK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        removed_stale_temp = cleanup_stale_temporary_files()
        if removed_stale_temp:
            print(f"Removed {removed_stale_temp:,} stale temporary file(s).")

        units, unit_ids = validate_and_load_input(INPUT_JSON_PATH)
        input_hash = sha256_file(INPUT_JSON_PATH)
        plans = {
            unit_id: build_unit_plan(unit_id, units[unit_id])
            for unit_id in unit_ids
        }
        progress = load_or_create_progress(
            PROGRESS_PATH,
            input_hash,
            len(unit_ids),
        )
        manifest = load_or_create_manifest(MANIFEST_PATH, input_hash)

        final_valid, final_error, saved_audiobooks = verify_saved_audiobooks(
            manifest,
            len(unit_ids),
        )
        if final_valid and saved_audiobooks is not None:
            cleanup_result = {"files": 0, "bytes": 0}
            if CLEAN_BUILD_ARTIFACTS_AFTER_SUCCESS and WORK_DIR.exists():
                cleanup_result = cleanup_build_artifacts()
                manifest["audiobooks"]["build_artifacts_cleaned_at"] = utc_now()
                manifest["audiobooks"]["cleaned_files"] = cleanup_result["files"]
                manifest["audiobooks"]["cleaned_bytes"] = cleanup_result["bytes"]
                save_progress_and_manifest(progress, manifest)

            print("\nExisting verified M4B and HTML output is already complete.")
            print(f"M4B volumes:        {saved_audiobooks['total_volumes']:,}")
            print(f"HTML companions:    {saved_audiobooks['total_volumes']:,}")
            print(
                "M4B duration:       "
                + format_duration(saved_audiobooks["total_duration_ms"] / 1000)
            )
            print(f"M4B folder:         {AUDIOBOOK_OUTPUT_DIR}")
            if cleanup_result["files"]:
                print(
                    f"Cleaned temporary:  {cleanup_result['files']:,} files | "
                    f"{cleanup_result['bytes'] / (1024 * 1024):.1f} MiB"
                )
            return True
        if manifest.get("audiobooks", {}).get("status") == "complete":
            print(f"\nExisting final output requires rebuilding: {final_error}")

        validate_runtime_dependencies(require_edge_tts=True)
        TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        BUILD_TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        UNIT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        UNIT_NUMBER_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        provider = EdgeTTSProvider()

        removed_obsolete = remove_obsolete_unit_files(plans)
        save_progress_and_manifest(progress, manifest)
        if removed_obsolete:
            print(
                f"Removed {removed_obsolete:,} obsolete V3 unit audio file(s)."
            )

        print(f"\nLoaded {len(unit_ids):,} validated learning units.")
        print(
            "Canonical IDs: "
            + ", ".join(format_unit_id(value) for value in unit_ids[:5])
            + (" ..." if len(unit_ids) > 5 else "")
        )
        print("\nTTS voices and delivery")
        print(
            "  Japanese engine: Edge neural TTS "
            f"({JA_RATE_JAPANESE_SLOW} slow; "
            f"{JA_RATE_JAPANESE_FINAL} final)"
        )
        print(f"  Japanese voices: {', '.join(JAPANESE_VOICES)}")
        print(
            "  Number voices:   "
            f"{JAPANESE_NUMBER_MALE_VOICE}, "
            f"{JAPANESE_NUMBER_FEMALE_VOICE} (Edge)"
        )
        print(f"  English voice:   {ENGLISH_MALE_VOICE} ({EN_RATE}, Edge)")

        run_result = process_unfinished_units(
            units=units,
            unit_ids=unit_ids,
            plans=plans,
            progress=progress,
            manifest=manifest,
            provider=provider,
            unit_limit=UNIT_LIMIT,
        )
        save_progress_and_manifest(progress, manifest)

        complete_count = sum(
            1
            for unit_id in unit_ids
            if completed_entry_is_valid(unit_id, plans[unit_id], progress)
        )
        remaining_count = len(unit_ids) - complete_count
        audiobook_result = None

        cleanup_result = {"files": 0, "bytes": 0}
        if remaining_count == 0:
            audiobook_result = assemble_m4b_volumes(
                unit_ids=unit_ids,
                units=units,
                plans=plans,
                progress=progress,
                manifest=manifest,
                provider=provider,
                input_hash=input_hash,
            )
            final_valid, final_error, _ = verify_saved_audiobooks(
                manifest,
                len(unit_ids),
            )
            if not final_valid:
                raise RuntimeError(
                    "Final M4B verification failed before cleanup: "
                    + final_error
                )
            if CLEAN_BUILD_ARTIFACTS_AFTER_SUCCESS:
                cleanup_result = cleanup_build_artifacts()
                manifest["audiobooks"]["build_artifacts_cleaned_at"] = utc_now()
                manifest["audiobooks"]["cleaned_files"] = cleanup_result["files"]
                manifest["audiobooks"]["cleaned_bytes"] = cleanup_result["bytes"]
                save_progress_and_manifest(progress, manifest)
        else:
            print(
                "\nM4B assembly is deferred until every unit is complete. "
                "This prevents partial shuffled volumes from changing between "
                "resumable runs."
            )

        print("\n" + "=" * 72)
        print("AUDIO RUN SUMMARY")
        print("=" * 72)
        print(f"Attempted this run: {run_result['attempted']:,}")
        print(f"Succeeded this run: {run_result['succeeded']:,}")
        print(f"Failed this run:    {run_result['failed']:,}")
        print(f"Total complete:     {complete_count:,}/{len(unit_ids):,}")
        print(f"Remaining:          {remaining_count:,}")
        print(
            f"Elapsed this run:   "
            f"{format_duration(run_result['elapsed_seconds'])}"
        )
        if remaining_count > 0:
            print(f"Temporary work:     {WORK_DIR}")
        print(f"Progress file:      {PROGRESS_PATH}")
        print(f"Manifest file:      {MANIFEST_PATH}")

        if audiobook_result:
            number_phase = audiobook_result.get("number_audio_phase", {})
            if number_phase:
                print(
                    "Number clip phase:  "
                    + format_duration(
                        float(number_phase.get("elapsed_seconds", 0.0))
                    )
                    + " | "
                    + f"{int(number_phase.get('built_files', 0)):,} built, "
                    + f"{int(number_phase.get('reused_files', 0)):,} reused"
                )
            print(
                f"M4B volumes:        "
                f"{audiobook_result['total_volumes']:,}"
            )
            print(
                "M4B duration:       "
                + format_duration(
                    audiobook_result["total_duration_ms"] / 1000
                )
            )
            print(
                f"HTML companions:    "
                f"{audiobook_result['total_volumes']:,}"
            )
            print(f"M4B folder:         {AUDIOBOOK_OUTPUT_DIR}")
            if cleanup_result["files"]:
                print(
                    f"Cleaned temporary:  {cleanup_result['files']:,} files | "
                    f"{cleanup_result['bytes'] / (1024 * 1024):.1f} MiB"
                )

        if remaining_count > 0:
            print(
                "\nRerun the script to retry failures or continue. "
                "Set UNIT_LIMIT = None for the full remaining run."
            )
        return True

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
