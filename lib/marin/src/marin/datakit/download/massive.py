# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AmazonScience/massive → OpenAI-style function-calling dataset.

MASSIVE is a multilingual NLU dataset with 60 intents and 55 slot types across
52 locales. Each row carries a user utterance plus an ``annot_utt`` of the form
``[slot_name : slot_value]``. We render each row as a function-calling example:
a ``tools`` list with one tool per intent and a ``messages`` pair where the
assistant emits a tool call whose arguments are the parsed slots.

The upstream is a single tar.gz on S3 (the ``AmazonScience/massive`` HF entry
is a script-only repo that points at this tarball). We download it directly
with ``requests`` — same pattern as ``nemotron_v1.py`` — and extract one
``{locale}.jsonl`` per locale into the staging dir.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import tarfile
import tempfile

import requests
from fray import ResourceConfig
from requests.adapters import HTTPAdapter
from rigging.filesystem import open_url, url_to_fs
from urllib3.util import Retry
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_jsonl
from zephyr.writers import atomic_rename

from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec
from marin.utils import fsspec_exists

logger = logging.getLogger(__name__)

HF_DATASET_ID = "AmazonScience/massive"
# The upstream v1.1 tarball — the HF script repo just points at this URL.
MASSIVE_TARBALL_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE_VERSION = "1.1"

# MASSIVE's per-row ``partition`` field uses ``dev`` for what HF calls ``validation``.
_PARTITION_TO_SPLIT = {"train": "train", "dev": "validation", "test": "test"}

# Per-intent slot vocabulary, derived from a full en-US train+dev+test scan.
# Slot names are language-invariant in MASSIVE (only slot values are localized),
# so a single English pass yields the canonical schema for every locale.
_INTENT_SLOTS: dict[str, tuple[str, ...]] = {
    "alarm_query": ("alarm_type", "date", "device_type", "event_name", "house_place", "time", "timeofday"),
    "alarm_remove": ("alarm_type", "date", "event_name", "person", "relation", "time", "timeofday"),
    "alarm_set": (
        "alarm_type",
        "date",
        "event_name",
        "general_frequency",
        "media_type",
        "order_type",
        "person",
        "relation",
        "time",
        "time_zone",
        "timeofday",
    ),
    "audio_volume_down": ("change_amount", "device_type"),
    "audio_volume_mute": ("change_amount", "date", "device_type", "event_name", "time", "timeofday"),
    "audio_volume_other": ("change_amount",),
    "audio_volume_up": ("change_amount", "device_type", "media_type", "song_name"),
    "calendar_query": (
        "business_name",
        "date",
        "event_name",
        "general_frequency",
        "list_name",
        "meal_type",
        "person",
        "place_name",
        "relation",
        "sport_type",
        "time",
        "timeofday",
    ),
    "calendar_remove": (
        "app_name",
        "business_type",
        "date",
        "event_name",
        "general_frequency",
        "list_name",
        "meal_type",
        "person",
        "place_name",
        "relation",
        "time",
        "timeofday",
        "transport_type",
    ),
    "calendar_set": (
        "artist_name",
        "business_name",
        "business_type",
        "date",
        "event_name",
        "general_frequency",
        "house_place",
        "meal_type",
        "media_type",
        "person",
        "personal_info",
        "place_name",
        "relation",
        "song_name",
        "sport_type",
        "time",
        "timeofday",
    ),
    "cooking_query": ("food_type", "meal_type"),
    "cooking_recipe": (
        "app_name",
        "business_name",
        "cooking_type",
        "date",
        "device_type",
        "drink_type",
        "event_name",
        "food_type",
        "ingredient",
        "list_name",
        "meal_type",
        "media_type",
        "time",
        "timeofday",
    ),
    "datetime_convert": ("person", "place_name", "time", "time_zone", "timeofday"),
    "datetime_query": ("date", "event_name", "food_type", "place_name", "time", "time_zone", "timeofday"),
    "email_addcontact": (
        "business_name",
        "email_address",
        "list_name",
        "person",
        "personal_info",
        "place_name",
        "relation",
    ),
    "email_query": (
        "business_name",
        "date",
        "email_folder",
        "event_name",
        "person",
        "personal_info",
        "place_name",
        "relation",
        "time",
        "timeofday",
    ),
    "email_querycontact": (
        "business_name",
        "date",
        "device_type",
        "event_name",
        "list_name",
        "media_type",
        "person",
        "personal_info",
        "place_name",
        "relation",
        "time",
    ),
    "email_sendemail": (
        "artist_name",
        "business_name",
        "date",
        "email_address",
        "email_folder",
        "event_name",
        "general_frequency",
        "meal_type",
        "person",
        "personal_info",
        "place_name",
        "relation",
        "time",
        "timeofday",
    ),
    "general_greet": ("date",),
    "general_joke": ("business_type", "date", "food_type", "joke_type", "person", "relation"),
    "general_quirky": (
        "artist_name",
        "business_type",
        "date",
        "device_type",
        "drink_type",
        "event_name",
        "food_type",
        "general_frequency",
        "meal_type",
        "media_type",
        "movie_name",
        "news_topic",
        "person",
        "place_name",
        "relation",
        "time",
        "timeofday",
    ),
    "iot_cleaning": ("date", "device_type", "general_frequency", "house_place", "place_name", "time"),
    "iot_coffee": ("business_name", "business_type", "coffee_type", "date", "device_type", "time", "timeofday"),
    "iot_hue_lightchange": ("change_amount", "color_type", "house_place", "player_setting", "time"),
    "iot_hue_lightdim": ("change_amount", "color_type", "device_type", "house_place"),
    "iot_hue_lightoff": ("color_type", "device_type", "house_place", "time"),
    "iot_hue_lighton": ("device_type", "house_place"),
    "iot_hue_lightup": ("change_amount", "device_type", "house_place"),
    "iot_wemo_off": ("device_type", "house_place"),
    "iot_wemo_on": ("device_type", "house_place"),
    "lists_createoradd": (
        "date",
        "drink_type",
        "event_name",
        "general_frequency",
        "list_name",
        "music_descriptor",
        "person",
        "place_name",
        "relation",
        "time",
    ),
    "lists_query": (
        "app_name",
        "business_name",
        "date",
        "device_type",
        "event_name",
        "list_name",
        "music_genre",
        "person",
        "timeofday",
    ),
    "lists_remove": (
        "date",
        "event_name",
        "list_name",
        "meal_type",
        "person",
        "place_name",
        "relation",
        "song_name",
        "timeofday",
    ),
    "music_dislikeness": ("music_descriptor", "music_genre"),
    "music_likeness": (
        "app_name",
        "artist_name",
        "date",
        "event_name",
        "music_descriptor",
        "music_genre",
        "place_name",
        "player_setting",
        "playlist_name",
        "song_name",
    ),
    "music_query": (
        "artist_name",
        "date",
        "media_type",
        "music_album",
        "music_genre",
        "person",
        "player_setting",
        "song_name",
        "timeofday",
    ),
    "music_settings": (
        "app_name",
        "artist_name",
        "device_type",
        "media_type",
        "music_descriptor",
        "music_genre",
        "player_setting",
        "song_name",
    ),
    "news_query": (
        "date",
        "device_type",
        "general_frequency",
        "media_type",
        "news_topic",
        "person",
        "place_name",
        "time",
        "timeofday",
        "transport_type",
    ),
    "play_audiobook": (
        "app_name",
        "artist_name",
        "audiobook_author",
        "audiobook_name",
        "date",
        "media_type",
        "player_setting",
        "song_name",
        "time",
    ),
    "play_game": ("device_type", "game_name", "game_type"),
    "play_music": (
        "app_name",
        "artist_name",
        "date",
        "device_type",
        "media_type",
        "movie_name",
        "music_album",
        "music_descriptor",
        "music_genre",
        "player_setting",
        "playlist_name",
        "song_name",
        "time",
        "timeofday",
    ),
    "play_podcasts": (
        "date",
        "media_type",
        "person",
        "place_name",
        "player_setting",
        "podcast_descriptor",
        "podcast_name",
        "radio_name",
        "time",
        "timeofday",
        "transport_type",
    ),
    "play_radio": (
        "app_name",
        "date",
        "device_type",
        "house_place",
        "media_type",
        "music_descriptor",
        "music_genre",
        "person",
        "radio_name",
        "relation",
        "time",
        "timeofday",
    ),
    "qa_currency": ("business_name", "currency_name", "date", "place_name"),
    "qa_definition": ("definition_word",),
    "qa_factoid": (
        "artist_name",
        "date",
        "event_name",
        "food_type",
        "list_name",
        "movie_name",
        "music_genre",
        "news_topic",
        "person",
        "place_name",
        "time",
    ),
    "qa_maths": ("date", "general_frequency"),
    "qa_stock": ("business_name", "currency_name", "date", "news_topic", "person", "time"),
    "recommendation_events": (
        "business_name",
        "business_type",
        "date",
        "event_name",
        "meal_type",
        "movie_type",
        "personal_info",
        "place_name",
        "time",
        "timeofday",
    ),
    "recommendation_locations": (
        "business_name",
        "business_type",
        "date",
        "drink_type",
        "food_type",
        "meal_type",
        "place_name",
        "time",
    ),
    "recommendation_movies": (
        "business_name",
        "business_type",
        "date",
        "event_name",
        "media_type",
        "movie_name",
        "movie_type",
        "place_name",
        "time",
        "timeofday",
    ),
    "social_post": (
        "business_name",
        "business_type",
        "date",
        "device_type",
        "event_name",
        "media_type",
        "person",
        "personal_info",
        "place_name",
        "relation",
        "weather_descriptor",
    ),
    "social_query": (
        "date",
        "event_name",
        "general_frequency",
        "media_type",
        "news_topic",
        "person",
        "relation",
        "time",
    ),
    "takeaway_order": (
        "app_name",
        "business_name",
        "business_type",
        "date",
        "drink_type",
        "food_type",
        "ingredient",
        "meal_type",
        "order_type",
        "time",
        "timeofday",
    ),
    "takeaway_query": (
        "business_name",
        "business_type",
        "event_name",
        "food_type",
        "meal_type",
        "order_type",
        "person",
        "place_name",
        "time",
        "timeofday",
    ),
    "transport_query": (
        "app_name",
        "business_name",
        "business_type",
        "date",
        "event_name",
        "food_type",
        "person",
        "place_name",
        "relation",
        "time",
        "timeofday",
        "transport_descriptor",
        "transport_name",
        "transport_type",
    ),
    "transport_taxi": (
        "app_name",
        "business_name",
        "business_type",
        "date",
        "event_name",
        "person",
        "place_name",
        "time",
        "timeofday",
        "transport_agency",
        "transport_name",
        "transport_type",
    ),
    "transport_ticket": (
        "app_name",
        "currency_name",
        "date",
        "place_name",
        "relation",
        "time",
        "timeofday",
        "transport_descriptor",
        "transport_name",
        "transport_type",
    ),
    "transport_traffic": ("date", "event_name", "place_name", "time", "timeofday"),
    "weather_query": (
        "business_type",
        "date",
        "event_name",
        "food_type",
        "general_frequency",
        "meal_type",
        "place_name",
        "time",
        "timeofday",
        "weather_descriptor",
    ),
}


def parse_annot_utt(annot: str) -> list[tuple[str, str]]:
    """Extract ``(slot_name, slot_value)`` pairs from a MASSIVE annotation.

    Format: ``[slot_name : slot_value]`` with arbitrary literal text between
    markers. Bracket and colon characters never occur inside slot values
    (verified across all 52 locales), so a single forward pass with
    ``str.find`` is unambiguous and faster than a regex.

    Pairs are returned in document order; a slot may repeat across an
    annotation (e.g. two ``place_name`` entries for an origin/destination).
    """
    out: list[tuple[str, str]] = []
    n = len(annot)
    i = 0
    while i < n:
        if annot[i] != "[":
            i += 1
            continue
        sep = annot.find(":", i + 1)
        end = annot.find("]", i + 1)
        if sep == -1 or end == -1 or sep > end:
            # Malformed marker — bail out rather than emitting half-parsed pairs.
            break
        out.append((annot[i + 1 : sep].strip(), annot[sep + 1 : end].strip()))
        i = end + 1
    return out


def _intent_tool_schema(intent: str) -> dict:
    """OpenAI Responses API ``tools[]`` entry for one MASSIVE intent.

    Responses API tool definitions are flat — ``name``, ``description``, and
    ``parameters`` live at the top level (no nested ``function`` wrapper, which
    is the Chat Completions convention).
    """
    slots = _INTENT_SLOTS[intent]
    return {
        "type": "function",
        "name": intent,
        "description": f"Handle a user utterance with the {intent!r} intent.",
        "parameters": {
            "type": "object",
            "properties": {
                slot: {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Values labelled as {slot!r} in the utterance.",
                }
                for slot in slots
            },
            "required": [],
        },
    }


# Full 60-tool registry; per-record subsets are sampled from this for the
# ``Tools:`` header (gold tool + distractors, see ``select_tools``).
TOOLS: list[dict] = [_intent_tool_schema(name) for name in sorted(_INTENT_SLOTS)]
_TOOLS_BY_NAME: dict[str, dict] = {tool["name"]: tool for tool in TOOLS}

# Per-record distractor count is sampled uniformly from this inclusive range.
# Capped above by ``len(TOOLS) - 1`` so we never ask for more distractors than
# exist outside the gold tool.
MIN_DISTRACTORS = 20
MAX_DISTRACTORS = 60


def render_function_call(call_id: str, intent: str, arguments: dict[str, list[str]]) -> str:
    """Serialize a Responses API ``function_call`` item as one line of JSON.

    Matches the OpenAI Responses API output shape used by mini-swe-agent's
    ``LitellmResponseModel``: ``type``, ``call_id``, ``name``, and the
    JSON-string ``arguments`` are all at the top level (no nested ``function``
    object).
    """
    return json.dumps(
        {
            "type": "function_call",
            "call_id": call_id,
            "name": intent,
            "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        },
        ensure_ascii=False,
    )


def select_tools(intent: str, seed: str) -> list[dict]:
    """Per-record tool selection: gold tool + sampled distractors, shuffled.

    Always includes the gold tool for ``intent``, plus ``randint(20, 60)``
    distractors drawn uniformly from the rest of the registry, then shuffles
    the combined list. Seeded from a stable string (locale/id/split) so reruns
    produce byte-identical output — required for StepSpec hash caching.
    ``random.Random`` SHA-512-hashes string seeds internally, so this isn't
    subject to Python's per-process hash randomization.
    """
    rng = random.Random(seed)
    others = [tool for tool in TOOLS if tool["name"] != intent]
    n_distractors = rng.randint(MIN_DISTRACTORS, min(MAX_DISTRACTORS, len(others)))
    pool = [_TOOLS_BY_NAME[intent], *rng.sample(others, n_distractors)]
    rng.shuffle(pool)
    return pool


def row_to_doc(row: dict) -> list[dict]:
    """Render one MASSIVE row as a Tools/Request/tool_call training document.

    The ``text`` field is a three-line transcript: a JSON-serialized ``Tools:``
    header (the gold tool plus 20-60 distractors sampled from the rest of the
    registry, shuffled together so the model can't latch onto positional bias),
    followed by ``Request: <utt>`` and ``tool_call: <function_call_json>``.
    Slot values are emitted as arrays of strings to handle utterances where
    the same slot appears more than once (e.g. ``[place_name : virginia] ...
    [place_name : california]``).

    Expects ``row['intent']`` to be a canonical string name (which it already
    is in the raw upstream JSONL — the HF loader script converts to ``ClassLabel``
    integers internally, but we read raw files now).
    """
    intent = row["intent"]
    if intent not in _TOOLS_BY_NAME:
        raise ValueError(
            f"Unknown intent {intent!r} (registry has {len(_TOOLS_BY_NAME)} intents). "
            "Either upstream MASSIVE schema drifted or the row is malformed."
        )
    arguments: dict[str, list[str]] = {}
    for slot, value in parse_annot_utt(row["annot_utt"]):
        arguments.setdefault(slot, []).append(value)

    split = _PARTITION_TO_SPLIT[row["partition"]]
    doc_id = f"{row['locale']}/{row['id']}/{split}"
    call_id = f"call_{row['locale']}_{row['id']}"

    tools = select_tools(intent, doc_id)
    call_json = render_function_call(call_id, intent, arguments)
    text = f"Tools: {json.dumps(tools, ensure_ascii=False)}\n" f"Request: {row['utt']}\n" f"tool_call: {call_json}"
    return [
        {
            "id": doc_id,
            "text": text,
            "source": HF_DATASET_ID,
            "locale": row["locale"],
            "split": split,
            "intent": intent,
            "source_id": row["id"],
        }
    ]


def _download_tarball(url: str, dest_path: str) -> None:
    """Stream a URL to a local file with retry on transient HTTP errors."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    headers = {"user-agent": "marin-massive-fc/1.0"}
    with session.get(url, stream=True, headers=headers) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as out:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)


def _extract_jsonl_files(tarball_path: str, output_path: str) -> None:
    """Extract every ``*.jsonl`` member of ``tarball_path`` to ``output_path``.

    Each member is written as ``{output_path}/{locale}.jsonl`` where ``locale``
    is the basename of the tar member with its extension stripped (matching
    the upstream MASSIVE loader's ``path.split('/')[-1].split('.')[0]``).
    Already-extracted targets are skipped so reruns are idempotent.
    """
    extracted = 0
    with tarfile.open(tarball_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not (member.isfile() and member.name.endswith(".jsonl")):
                continue
            locale = os.path.basename(member.name).removesuffix(".jsonl")
            target = f"{output_path}/{locale}.jsonl"
            if fsspec_exists(target):
                logger.info(f"Skipping {target} (already exists)")
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            with atomic_rename(target) as tmp, open_url(tmp, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted += 1
            logger.info(f"Extracted {target}")
    logger.info(f"Extracted {extracted} per-locale JSONL files to {output_path}")


def stage_massive_raw(output_path: str) -> None:
    """Download the MASSIVE tarball from S3 and extract per-locale JSONL files.

    Mirrors the standard datakit download pattern: fetch once to local disk,
    then materialize per-locale files into ``output_path``. The downstream
    zephyr transform reads each ``{locale}.jsonl`` in parallel.

    Each row's ``intent`` field is already a canonical string name in the raw
    upstream data, so no ClassLabel mapping is required (this is what the HF
    loader script does internally before re-emitting).
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmpf:
        logger.info(f"Downloading {MASSIVE_TARBALL_URL}")
        _download_tarball(MASSIVE_TARBALL_URL, tmpf.name)
        logger.info(f"Extracting per-locale files to {output_path}")
        _extract_jsonl_files(tmpf.name, output_path)


def _list_staged_files(input_path: str) -> list[str]:
    """Return all ``{locale}.jsonl`` files staged under ``input_path``."""
    fs, root = url_to_fs(input_path)
    protocol = input_path.split("://", 1)[0] if "://" in input_path else ""
    files: list[str] = []
    for entry in fs.ls(root, detail=False):
        if entry.endswith(".jsonl"):
            files.append(f"{protocol}://{entry}" if protocol else entry)
    return sorted(files)


def transform_staged_massive(input_path: str, output_path: str) -> None:
    """Zephyr-parallelize ``row_to_doc`` across staged per-locale JSONL files.

    Parallelism unit is one staged JSONL file (``en-US.jsonl``, ``ja-JP.jsonl``,
    …) — each contains rows for all three splits, distinguished by the
    per-row ``partition`` field. Each worker streams its file, applies
    ``row_to_doc`` per row, and writes parquet shards to ``output_path``.
    """
    files = _list_staged_files(input_path)
    if not files:
        raise FileNotFoundError(f"No staged JSONL files under {input_path}")
    logger.info(f"Transforming {len(files)} staged files via zephyr")

    pipeline = (
        Dataset.from_list(files)
        .flat_map(load_jsonl)
        .flat_map(row_to_doc)
        .write_parquet(
            f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet",
            skip_existing=True,
        )
    )
    # Each row produces ~50KB of parquet (most of it the Tools header). 2GB RAM
    # is comfortable headroom even with batched writes.
    ctx = ZephyrContext(name="massive-transform", resources=ResourceConfig(cpu=1, ram="2g"))
    ctx.execute(pipeline)


def stage_massive_step() -> StepSpec:
    """Sequential staging step: download tarball → per-locale JSONL files."""
    return StepSpec(
        name="raw/massive",
        fn=lambda output_path: stage_massive_raw(output_path=output_path),
        hash_attrs={
            "tarball_url": MASSIVE_TARBALL_URL,
            "version": MASSIVE_VERSION,
        },
    )


def transform_massive_step(staged: StepSpec) -> StepSpec:
    """Zephyr transform step: per-locale JSONL → function-calling parquet."""
    return StepSpec(
        name="processed/massive_function_calling",
        deps=[staged],
        fn=lambda output_path: transform_staged_massive(input_path=staged.output_path, output_path=output_path),
        hash_attrs={"schema_version": "v1"},
    )


def massive_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(stage, transform, normalize)`` chain for MASSIVE-FC."""
    staged = stage_massive_step()
    transformed = transform_massive_step(staged)
    return (
        staged,
        transformed,
        normalize_step(
            name="normalized/massive_function_calling",
            download=transformed,
            text_field="text",
            id_field="id",
            file_extensions=(".parquet",),
        ),
    )
