#!/usr/bin/env python3
"""Build a two-level summary dataset from the English DetectiveQA novels.

Level 1:
    Contiguous numbered paragraphs are grouped into chunks. Every chunk has at
    least ``--min-whitespace-tokens`` whitespace-delimited tokens, except when
    the entire novel is shorter than that threshold. Each chunk is summarized
    independently with Gemma 4 through the Gemini API.

Level 2:
    All level-1 summaries for a novel are merged, in narrative order, and
    summarized once more to create a whole-novel summary.

The script preserves DetectiveQA paragraph IDs and uses the exact same document
ID convention as ``prepare_detectiveqa_ranking.py``:

    dqa-en-{novel_id}-p{paragraph_id}

This makes the generated hierarchy directly mappable to the existing paragraph
corpus and qrels. If the ranking corpus exists, strict text/ID alignment is
validated before any paid API calls are made. Existing paragraph qrels are also
propagated to chunk- and novel-level qrels.

Example:
    python build_detectiveqa_summary_dataset.py

Dependencies:
    pip install -U google-genai python-dotenv tqdm

.env:
    GEMINI_API_KEY=your_google_ai_studio_key
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from google.genai import types


from dotenv import load_dotenv
from tqdm import tqdm
LOG = logging.getLogger("detectiveqa_summary_builder")

PARAGRAPH_RE = re.compile(r"(?m)^[ \t]*\[(\d+)\][ \t]*")
NOVEL_ID_RE = re.compile(r"^(\d+)-")
CORPUS_DOC_ID_RE = re.compile(r"^dqa-en-(\d+)-p(\d+)$")
PROMPT_VERSION = "detectiveqa-summary-v1.2"
DEFAULT_MODEL = "gemma-4-26b-a4b-it"

SYSTEM_INSTRUCTION = """You summarize English-language fictional narratives for retrieval.
Preserve the important characters, actions, discoveries, locations, motives, and
consequences explicitly stated in the supplied material. Resolve references only
when the supplied material supports the resolution. Do not invent facts, do not
add literary criticism, and do not speculate. Write plain English prose only.
+add literary criticism, and do not speculate. Write plain English prose only.
+Return only the final summary. Do not output reasoning, drafts, outlines,
+checklists, self-evaluation, or word-count commentary."""

CHUNK_PROMPT_TEMPLATE = """Summarize the following contiguous passage from a novel.

Requirements:
- Write a self-contained factual summary of approximately {min_words}-{max_words} words.
- Preserve names, aliases, concrete actions, discoveries, locations, and explicit motives.
- Preserve narrative order.
- Include details that could help retrieve this passage in response to a factual,
  causal, character, or event question.
- Do not mention paragraph IDs, the summarization task, or these instructions.
- Do not add information that is not supported by the passage.

Novel: {title}
Novel ID: {novel_id}
Source paragraph range: [{start_paragraph_id}] to [{end_paragraph_id}]

PASSAGE
{source_text}
"""

NOVEL_PROMPT_TEMPLATE = """Create a whole-novel summary from the ordered passage summaries below.

Requirements:
- Write approximately {min_words}-{max_words} words in plain English prose.
- Synthesize the summaries into one coherent account of the major plot progression.
- Preserve chronological order, important characters, major actions, discoveries,
  conflicts, and outcomes.
- Merge repeated information instead of listing each passage summary separately.
- Do not invent missing links or facts.
- Do not mention chunk IDs, paragraph IDs, or the summarization process.

Novel: {title}
Novel ID: {novel_id}

ORDERED PASSAGE SUMMARIES
{merged_summaries}
"""


@dataclass(frozen=True)
class Paragraph:
    paragraph_id: int
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    novel_id: int
    title: str
    source_file: str
    chunk_index: int
    paragraphs: tuple[Paragraph, ...]
    source_text: str
    whitespace_tokens: int
    character_count: int

    @property
    def start_paragraph_id(self) -> int:
        return self.paragraphs[0].paragraph_id

    @property
    def end_paragraph_id(self) -> int:
        return self.paragraphs[-1].paragraph_id

    @property
    def paragraph_ids(self) -> list[int]:
        return [paragraph.paragraph_id for paragraph in self.paragraphs]

    @property
    def paragraph_doc_ids(self) -> list[str]:
        return [doc_id(self.novel_id, paragraph.paragraph_id) for paragraph in self.paragraphs]


@dataclass(frozen=True)
class ApiSettings:
    model: str
    temperature: float
    seed: int
    max_retries: int
    initial_backoff_seconds: float
    max_backoff_seconds: float
    request_pause_seconds: float
    diagnostics_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a two-level Gemma summary hierarchy for English DetectiveQA novels."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/detectiveqa/novel_data_en"),
        help="Directory containing numbered English DetectiveQA .txt novels.",
    )
    parser.add_argument(
        "--ranking-dir",
        type=Path,
        default=Path("data/detectiveqa-ranking-en"),
        help="Existing paragraph ranking dataset used for strict alignment and qrel propagation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/detectiveqa-summary-en"),
        help="Output directory for the hierarchical summary dataset.",
    )
    parser.add_argument(
        "--min-whitespace-tokens",
        type=int,
        default=1500,
        help="Minimum whitespace-delimited tokens per level-1 chunk.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-summary-min-words", type=int, default=180)
    parser.add_argument("--chunk-summary-max-words", type=int, default=300)
    parser.add_argument("--novel-summary-min-words", type=int, default=500)
    parser.add_argument("--novel-summary-max-words", type=int, default=900)
    parser.add_argument("--chunk-max-output-tokens", type=int, default=700)
    parser.add_argument("--novel-max-output-tokens", type=int, default=1800)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--initial-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=60.0)
    parser.add_argument(
        "--request-pause-seconds",
        type=float,
        default=0.5,
        help="Pause between successful API calls to reduce rate-limit pressure.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Process only the first N novels for a smoke test; 0 means all.",
    )
    parser.add_argument(
        "--only-novel-id",
        type=int,
        action="append",
        default=[],
        help="Process only specific novel IDs. Repeat the option to select multiple novels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output dataset files and checkpoints before rebuilding.",
    )
    parser.add_argument(
        "--skip-baseline-validation",
        action="store_true",
        help="Allow running without strict comparison to ranking-dir/corpus.jsonl.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Parse, validate, chunk, and report the plan without calling the API.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def whitespace_token_count(text: str) -> int:
    return len(text.split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonl_write(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield payload


def doc_id(novel_id: int, paragraph_id: int) -> str:
    """Must stay identical to prepare_detectiveqa_ranking.py."""
    return f"dqa-en-{novel_id}-p{paragraph_id}"


def chunk_id(novel_id: int, chunk_index: int) -> str:
    return f"dqa-en-{novel_id}-c{chunk_index:04d}"


def novel_summary_id(novel_id: int) -> str:
    return f"dqa-en-{novel_id}-novel-summary"


def parse_novel(path: Path) -> list[Paragraph]:
    """Parse `[paragraph_id] text`, including multiline paragraphs.

    This intentionally mirrors the parser in prepare_detectiveqa_ranking.py.
    """
    raw = path.read_text(encoding="utf-8-sig")
    markers = list(PARAGRAPH_RE.finditer(raw))
    if not markers:
        raise ValueError(f"No numbered paragraphs found in {path}")

    paragraphs: list[Paragraph] = []
    seen_ids: set[int] = set()
    for index, marker in enumerate(markers):
        paragraph_id = int(marker.group(1))
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(raw)
        if paragraph_id in seen_ids:
            raise ValueError(f"Duplicate paragraph [{paragraph_id}] in {path}")
        seen_ids.add(paragraph_id)
        paragraphs.append(Paragraph(paragraph_id=paragraph_id, text=clean_text(raw[start:end])))

    # IDs do not need to be consecutive, but order must be strictly increasing.
    for left, right in zip(paragraphs, paragraphs[1:]):
        if right.paragraph_id <= left.paragraph_id:
            raise ValueError(f"Paragraph IDs are not strictly increasing in {path}")
    return paragraphs


def extract_novel_id(path: Path) -> int:
    match = NOVEL_ID_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot extract novel ID from filename: {path.name}")
    return int(match.group(1))


def extract_title(path: Path) -> str:
    # Keep the same convention used by prepare_detectiveqa_ranking.py.
    return path.stem.split("-", maxsplit=2)[1] if "-" in path.stem else path.stem


def render_paragraphs(paragraphs: Sequence[Paragraph]) -> str:
    return "\n".join(f"[{paragraph.paragraph_id}] {paragraph.text}" for paragraph in paragraphs)


def build_chunks(
    *,
    novel_id: int,
    title: str,
    source_file: str,
    paragraphs: Sequence[Paragraph],
    min_tokens: int,
) -> list[Chunk]:
    """Greedily group complete paragraphs and merge an undersized tail backward."""
    if min_tokens <= 0:
        raise ValueError("min_tokens must be positive")
    if not paragraphs:
        raise ValueError(f"Novel {novel_id} has no paragraphs")

    groups: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    current_tokens = 0

    for paragraph in paragraphs:
        current.append(paragraph)
        # Count exactly the representation passed to the model, including [id].
        current_tokens = whitespace_token_count(render_paragraphs(current))
        if current_tokens >= min_tokens:
            groups.append(current)
            current = []
            current_tokens = 0

    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)

    chunks: list[Chunk] = []
    for index, group in enumerate(groups):
        source_text = render_paragraphs(group)
        chunks.append(
            Chunk(
                chunk_id=chunk_id(novel_id, index),
                novel_id=novel_id,
                title=title,
                source_file=source_file,
                chunk_index=index,
                paragraphs=tuple(group),
                source_text=source_text,
                whitespace_tokens=whitespace_token_count(source_text),
                character_count=len(source_text),
            )
        )

    # Every chunk should meet the threshold unless the entire novel cannot.
    total_tokens = whitespace_token_count(render_paragraphs(paragraphs))
    for chunk in chunks:
        if chunk.whitespace_tokens < min_tokens and total_tokens >= min_tokens:
            raise AssertionError(
                f"Chunk {chunk.chunk_id} has {chunk.whitespace_tokens} tokens, below {min_tokens}"
            )

    # Exact one-to-one and order-preserving coverage.
    flattened_ids = [pid for chunk in chunks for pid in chunk.paragraph_ids]
    original_ids = [paragraph.paragraph_id for paragraph in paragraphs]
    if flattened_ids != original_ids:
        raise AssertionError(f"Chunk coverage/order mismatch for novel {novel_id}")
    if len(flattened_ids) != len(set(flattened_ids)):
        raise AssertionError(f"Paragraph assigned to multiple chunks in novel {novel_id}")

    return chunks


def load_ranking_corpus(path: Path) -> dict[str, dict[str, Any]]:
    corpus: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        identifier = str(row.get("_id", ""))
        if not identifier:
            raise ValueError(f"Corpus row without _id in {path}")
        if identifier in corpus:
            raise ValueError(f"Duplicate corpus ID {identifier} in {path}")
        corpus[identifier] = row
    return corpus


def validate_alignment(
    novel_id: int,
    paragraphs: Sequence[Paragraph],
    ranking_corpus: dict[str, dict[str, Any]],
) -> None:
    """Fail fast if raw paragraphs do not exactly match the prior baseline corpus."""
    expected_ids = {doc_id(novel_id, paragraph.paragraph_id) for paragraph in paragraphs}
    actual_ids = {
        identifier
        for identifier in ranking_corpus
        if (match := CORPUS_DOC_ID_RE.fullmatch(identifier))
        and int(match.group(1)) == novel_id
    }

    if not actual_ids:
        raise ValueError(
            f"Novel {novel_id} has no documents in the ranking corpus. "
            "This usually means the ranking dataset was built from a narrower "
            "annotation subset than the raw novel directory."
        )

    def id_sort_key(identifier: str) -> int:
        match = CORPUS_DOC_ID_RE.fullmatch(identifier)
        return int(match.group(2)) if match else -1

    missing = sorted(expected_ids - actual_ids, key=id_sort_key)
    extra = sorted(actual_ids - expected_ids, key=id_sort_key)
    if missing or extra:
        raise ValueError(
            f"Novel {novel_id} ID mismatch with ranking corpus: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    for paragraph in paragraphs:
        identifier = doc_id(novel_id, paragraph.paragraph_id)
        baseline_text = clean_text(str(ranking_corpus[identifier].get("text", "")))
        if paragraph.text != baseline_text:
            raise ValueError(
                f"Text mismatch for {identifier}. Stop before API calls to avoid shifted mapping."
            )



class ModelResponseError(RuntimeError):
    """A model response was returned, but it did not contain usable answer text."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def enum_name(value: Any) -> str | None:
    """Return a stable readable name for SDK enum-like values."""
    if value is None:
        return None
    for attribute in ("name", "value"):
        candidate = getattr(value, attribute, None)
        if candidate is not None:
            return str(candidate)
    return str(value)


def response_to_jsonable(response: Any) -> dict[str, Any]:
    """Convert a google-genai response to JSON-compatible data for diagnostics."""
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            payload = model_dump(mode="json", exclude_none=True)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    to_json_dict = getattr(response, "to_json_dict", None)
    if callable(to_json_dict):
        try:
            payload = to_json_dict()
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    return {"repr": repr(response)}


def extract_visible_response_text(response: Any) -> str:
    """Extract visible answer text without relying only on response.text.

    The SDK's response.text property is a convenience accessor. It can be empty
    or raise when a response has no valid text part. Iterating candidate parts
    gives better diagnostics and also works around accessor edge cases.
    """
    fragments: list[str] = []

    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            # Never use hidden thought content as a dataset summary.
            if bool(getattr(part, "thought", False)):
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                fragments.append(text.strip())

    if fragments:
        return "\n".join(fragments).strip()

    # Fallback for SDK versions that expose text only through the accessor.
    try:
        text = getattr(response, "text", None)
    except Exception:
        text = None
    return text.strip() if isinstance(text, str) else ""


def response_diagnostics(response: Any) -> dict[str, Any]:
    candidates = getattr(response, "candidates", None) or []
    finish_reasons = [
        enum_name(getattr(candidate, "finish_reason", None))
        for candidate in candidates
    ]
    safety_ratings = []
    for candidate in candidates:
        ratings = getattr(candidate, "safety_ratings", None) or []
        safety_ratings.append(
            [
                {
                    "category": enum_name(getattr(rating, "category", None)),
                    "probability": enum_name(getattr(rating, "probability", None)),
                    "blocked": getattr(rating, "blocked", None),
                }
                for rating in ratings
            ]
        )

    prompt_feedback = getattr(response, "prompt_feedback", None)
    usage = getattr(response, "usage_metadata", None)

    return {
        "finish_reasons": finish_reasons,
        "prompt_block_reason": enum_name(
            getattr(prompt_feedback, "block_reason", None)
        ),
        "prompt_block_reason_message": getattr(
            prompt_feedback, "block_reason_message", None
        ),
        "safety_ratings": safety_ratings,
        "usage": {
            "prompt_token_count": getattr(usage, "prompt_token_count", None),
            "candidates_token_count": getattr(usage, "candidates_token_count", None),
            "thoughts_token_count": getattr(usage, "thoughts_token_count", None),
            "total_token_count": getattr(usage, "total_token_count", None),
        },
    }


def save_failed_response_diagnostic(
    *,
    settings: ApiSettings,
    prompt: str,
    response: Any,
    attempt: int,
    max_output_tokens: int,
) -> Path:
    settings.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    prompt_hash = sha256_text(prompt)
    path = settings.diagnostics_dir / (
        f"empty-{prompt_hash[:16]}-attempt-{attempt + 1}.json"
    )
    payload = {
        "model": settings.model,
        "attempt": attempt + 1,
        "prompt_sha256": prompt_hash,
        "prompt_character_count": len(prompt),
        "prompt_whitespace_tokens": whitespace_token_count(prompt),
        "max_output_tokens": max_output_tokens,
        "diagnostics": response_diagnostics(response),
        "response": response_to_jsonable(response),
    }
    atomic_write_json(path, payload)
    return path


def empty_response_retryability(diagnostics: dict[str, Any]) -> tuple[bool, str]:
    reasons = {
        str(reason).upper()
        for reason in diagnostics.get("finish_reasons", [])
        if reason
    }
    block_reason = str(diagnostics.get("prompt_block_reason") or "").upper()

    if block_reason and block_reason not in {"NONE", "BLOCK_REASON_UNSPECIFIED"}:
        return False, f"prompt blocked: {block_reason}"

    non_retryable_markers = {
        "SAFETY",
        "PROHIBITED_CONTENT",
        "BLOCKLIST",
        "RECITATION",
        "SPII",
    }
    if any(any(marker in reason for marker in non_retryable_markers) for reason in reasons):
        return False, f"candidate blocked: {sorted(reasons)}"

    if any("MAX_TOKENS" in reason for reason in reasons):
        return False, "generation reached MAX_TOKENS; increase --chunk-max-output-tokens or --novel-max-output-tokens"

    # Empty STOP/unspecified responses can be transient backend/SDK behavior.
    return True, f"no visible text; finish reasons={sorted(reasons) or ['UNKNOWN']}"


def load_api_key() -> str:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Add GEMINI_API_KEY=... (or GOOGLE_API_KEY=...) to .env"
        )
    return key


def exception_status_code(exc: Exception) -> int | None:
    for attribute in ("code", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"\b(429|500|502|503|504)\b", str(exc))
    return int(match.group(1)) if match else None


def generate_text_with_retry(
    *,
    client: Any,
    settings: ApiSettings,
    prompt: str,
    max_output_tokens: int,
) -> str:
    from google.genai import types

    retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(settings.max_retries + 1):
        try:
            response = client.models.generate_content(
                model=settings.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=settings.temperature,
                    seed=settings.seed,
                    max_output_tokens=max_output_tokens,
                    candidate_count=1,
                    response_mime_type="text/plain",
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="NONE")
                    ),
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL,
                        include_thoughts=False,
                    ),
                )
            )

            text = extract_visible_response_text(response)
            if text:
                if settings.request_pause_seconds > 0:
                    time.sleep(settings.request_pause_seconds)
                return text

            diagnostic_path = save_failed_response_diagnostic(
                settings=settings,
                prompt=prompt,
                response=response,
                attempt=attempt,
                max_output_tokens=max_output_tokens,
            )
            diagnostics = response_diagnostics(response)
            retryable, reason = empty_response_retryability(diagnostics)
            raise ModelResponseError(
                f"The API returned no visible text ({reason}). "
                f"Diagnostic saved to {diagnostic_path}",
                retryable=retryable,
            )

        except Exception as exc:  # SDK exception classes can vary by version.
            status = exception_status_code(exc)
            explicit_retryable = getattr(exc, "retryable", None)
            retryable = (
                bool(explicit_retryable)
                if explicit_retryable is not None
                else status in retryable_statuses or status is None
            )

            if attempt >= settings.max_retries or not retryable:
                raise

            backoff = min(
                settings.max_backoff_seconds,
                settings.initial_backoff_seconds * (2**attempt),
            )
            jitter = random.uniform(0.0, min(1.0, backoff * 0.1))
            wait_seconds = backoff + jitter
            LOG.warning(
                "API call failed (status=%s, attempt=%d/%d): %s. Retrying in %.1fs",
                status,
                attempt + 1,
                settings.max_retries + 1,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise AssertionError("Unreachable retry loop")

def generation_request_sha256(
    *,
    prompt: str,
    settings: ApiSettings,
    max_output_tokens: int,
) -> str:
    """Fingerprint every input that can change a cached generation."""
    request = {
        "prompt_version": PROMPT_VERSION,
        "system_instruction": SYSTEM_INSTRUCTION,
        "prompt": prompt,
        "model": settings.model,
        "temperature": settings.temperature,
        "seed": settings.seed,
        "max_output_tokens": max_output_tokens,
    }
    return sha256_text(json.dumps(request, ensure_ascii=False, sort_keys=True))


def checkpoint_is_reusable(
    payload: dict[str, Any],
    *,
    request_sha256: str,
) -> bool:
    return (
        payload.get("request_sha256") == request_sha256
        and bool(str(payload.get("summary", "")).strip())
    )


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def summarize_chunk(
    *,
    client: Any,
    settings: ApiSettings,
    chunk: Chunk,
    checkpoint_dir: Path,
    min_words: int,
    max_words: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    prompt = CHUNK_PROMPT_TEMPLATE.format(
        min_words=min_words,
        max_words=max_words,
        title=chunk.title,
        novel_id=chunk.novel_id,
        start_paragraph_id=chunk.start_paragraph_id,
        end_paragraph_id=chunk.end_paragraph_id,
        source_text=chunk.source_text,
    )
    source_hash = sha256_text(chunk.source_text)
    request_hash = generation_request_sha256(
        prompt=prompt,
        settings=settings,
        max_output_tokens=max_output_tokens,
    )
    checkpoint_path = checkpoint_dir / f"{chunk.chunk_id}.json"
    cached = load_json_if_exists(checkpoint_path)
    if cached and checkpoint_is_reusable(
        cached,
        request_sha256=request_hash,
    ):
        LOG.info("Reusing chunk checkpoint %s", chunk.chunk_id)
        return cached

    LOG.info(
        "Calling API for %s: source_tokens=%d, chars=%d, paragraphs=[%d]-[%d], max_output_tokens=%d",
        chunk.chunk_id,
        chunk.whitespace_tokens,
        chunk.character_count,
        chunk.start_paragraph_id,
        chunk.end_paragraph_id,
        max_output_tokens,
    )

    summary = generate_text_with_retry(
        client=client,
        settings=settings,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
    )
    payload = {
        "_id": chunk.chunk_id,
        "level": 1,
        "novel_id": chunk.novel_id,
        "title": chunk.title,
        "source_file": chunk.source_file,
        "chunk_index": chunk.chunk_index,
        "summary": summary,
        "summary_whitespace_tokens": whitespace_token_count(summary),
        "start_paragraph_id": chunk.start_paragraph_id,
        "end_paragraph_id": chunk.end_paragraph_id,
        "paragraph_ids": chunk.paragraph_ids,
        "paragraph_doc_ids": chunk.paragraph_doc_ids,
        "source_whitespace_tokens": chunk.whitespace_tokens,
        "source_character_count": chunk.character_count,
        "source_sha256": source_hash,
        "request_sha256": request_hash,
        "model": settings.model,
        "temperature": settings.temperature,
        "seed": settings.seed,
        "max_output_tokens": max_output_tokens,
        "prompt_version": PROMPT_VERSION,
    }
    atomic_write_json(checkpoint_path, payload)
    return payload


def summarize_novel(
    *,
    client: Any,
    settings: ApiSettings,
    novel_id: int,
    title: str,
    source_file: str,
    chunk_summaries: Sequence[dict[str, Any]],
    checkpoint_dir: Path,
    min_words: int,
    max_words: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    merged_summaries = "\n\n".join(
        (
            f"Passage {row['chunk_index'] + 1} "
            f"(source paragraphs [{row['start_paragraph_id']}]"
            f"-[{row['end_paragraph_id']}]):\n{row['summary']}"
        )
        for row in sorted(chunk_summaries, key=lambda item: int(item["chunk_index"]))
    )
    prompt = NOVEL_PROMPT_TEMPLATE.format(
        min_words=min_words,
        max_words=max_words,
        title=title,
        novel_id=novel_id,
        merged_summaries=merged_summaries,
    )
    source_hash = sha256_text(merged_summaries)
    request_hash = generation_request_sha256(
        prompt=prompt,
        settings=settings,
        max_output_tokens=max_output_tokens,
    )
    identifier = novel_summary_id(novel_id)
    checkpoint_path = checkpoint_dir / f"{identifier}.json"
    cached = load_json_if_exists(checkpoint_path)
    if cached and checkpoint_is_reusable(
        cached,
        request_sha256=request_hash,
    ):
        LOG.info("Reusing novel checkpoint %s", identifier)
        return cached

    summary = generate_text_with_retry(
        client=client,
        settings=settings,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
    )
    ordered_children = sorted(chunk_summaries, key=lambda item: int(item["chunk_index"]))
    paragraph_doc_ids = [
        doc_identifier
        for row in ordered_children
        for doc_identifier in row["paragraph_doc_ids"]
    ]
    payload = {
        "_id": identifier,
        "level": 2,
        "novel_id": novel_id,
        "title": title,
        "source_file": source_file,
        "summary": summary,
        "summary_whitespace_tokens": whitespace_token_count(summary),
        "child_chunk_ids": [str(row["_id"]) for row in ordered_children],
        "paragraph_doc_ids": paragraph_doc_ids,
        "start_paragraph_id": int(ordered_children[0]["start_paragraph_id"]),
        "end_paragraph_id": int(ordered_children[-1]["end_paragraph_id"]),
        "merged_chunk_summary_whitespace_tokens": whitespace_token_count(merged_summaries),
        "source_sha256": source_hash,
        "request_sha256": request_hash,
        "model": settings.model,
        "temperature": settings.temperature,
        "seed": settings.seed,
        "max_output_tokens": max_output_tokens,
        "prompt_version": PROMPT_VERSION,
    }
    atomic_write_json(checkpoint_path, payload)
    return payload


def read_paragraph_qrels(path: Path) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        expected = ["query-id", "corpus-id", "score"]
        if header != expected:
            raise ValueError(f"Unexpected qrels header in {path}: {header}")
        for line_number, line in enumerate(handle, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"Malformed qrel at {path}:{line_number}")
            query_id_value, corpus_id_value, score_text = parts
            rows.append((query_id_value, corpus_id_value, int(score_text)))
    return rows


def propagate_qrels(
    *,
    paragraph_qrels_path: Path,
    paragraph_to_chunk: dict[str, str],
    chunk_to_novel_summary: dict[str, str],
    processed_novel_ids: set[int],
    output_qrels_dir: Path,
) -> dict[str, int]:
    paragraph_qrels = read_paragraph_qrels(paragraph_qrels_path)
    chunk_rows: set[tuple[str, str, int]] = set()
    novel_rows: set[tuple[str, str, int]] = set()

    for query_id_value, paragraph_doc_id, score in paragraph_qrels:
        if score <= 0:
            continue
        mapped_chunk = paragraph_to_chunk.get(paragraph_doc_id)
        if mapped_chunk is None:
            match = re.match(r"^dqa-en-(\d+)-p\d+$", paragraph_doc_id)
            qrel_novel_id = int(match.group(1)) if match else None
            if qrel_novel_id in processed_novel_ids:
                raise ValueError(f"No chunk mapping for gold paragraph {paragraph_doc_id}")
            # Smoke tests may process only a subset of novels. Ignore qrels
            # belonging to novels outside that selected subset.
            continue
        chunk_rows.add((query_id_value, mapped_chunk, 1))
        novel_rows.add((query_id_value, chunk_to_novel_summary[mapped_chunk], 1))

    output_qrels_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("chunks.tsv", chunk_rows),
        ("novels.tsv", novel_rows),
    ):
        with (output_qrels_dir / filename).open("w", encoding="utf-8") as handle:
            handle.write("query-id\tcorpus-id\tscore\n")
            for query_id_value, corpus_id_value, score in sorted(rows):
                handle.write(f"{query_id_value}\t{corpus_id_value}\t{score}\n")

    return {
        "source_paragraph_qrels": len(paragraph_qrels),
        "chunk_qrels": len(chunk_rows),
        "novel_qrels": len(novel_rows),
    }


def prepare_output(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "novels").mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / "chunks", checkpoint_dir / "novels"


def discover_novel_files(input_dir: Path) -> dict[int, Path]:
    """Index every raw English novel by its numeric DetectiveQA ID."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    indexed: dict[int, Path] = {}
    for path in input_dir.glob("*.txt"):
        match = NOVEL_ID_RE.match(path.name)
        if not match:
            LOG.warning("Skipping file without numeric DetectiveQA prefix: %s", path.name)
            continue
        novel_id_value = int(match.group(1))
        if novel_id_value in indexed:
            raise ValueError(f"Duplicate novel ID {novel_id_value} in {input_dir}")
        indexed[novel_id_value] = path

    if not indexed:
        raise FileNotFoundError(f"No matching English novel files found in {input_dir}")
    return indexed


def ranking_novel_ids(ranking_corpus: dict[str, dict[str, Any]]) -> set[int]:
    """Return novel IDs represented by the baseline paragraph corpus."""
    result: set[int] = set()
    malformed: list[str] = []
    for identifier in ranking_corpus:
        match = CORPUS_DOC_ID_RE.fullmatch(identifier)
        if not match:
            malformed.append(identifier)
            continue
        result.add(int(match.group(1)))
    if malformed:
        raise ValueError(
            "Ranking corpus contains document IDs outside the expected "
            f"dqa-en-{{novel_id}}-p{{paragraph_id}} format: {malformed[:5]}"
        )
    if not result:
        raise ValueError("Ranking corpus contains no DetectiveQA paragraph documents")
    return result


def select_novel_files(
    *,
    raw_index: dict[int, Path],
    baseline_ids: set[int] | None,
    only_ids: set[int],
    max_files: int,
) -> list[Path]:
    """Select files after applying baseline scope, explicit IDs, and smoke-test limit.

    When a ranking corpus is available, the default scope is exactly the novels
    represented in that corpus. This is required for apples-to-apples comparison
    with the existing retrieval baseline. Raw novels without baseline documents
    are skipped rather than treated as alignment failures.
    """
    raw_ids = set(raw_index)

    if only_ids:
        missing_raw = sorted(only_ids - raw_ids)
        if missing_raw:
            raise FileNotFoundError(
                f"Requested novel IDs are missing from the raw directory: {missing_raw}"
            )
        if baseline_ids is not None:
            outside_baseline = sorted(only_ids - baseline_ids)
            if outside_baseline:
                raise ValueError(
                    "Requested novel IDs are not present in the ranking corpus and "
                    "therefore cannot be compared with the current baseline: "
                    f"{outside_baseline}. Rebuild the ranking dataset with matching "
                    "annotation scope, or use --skip-baseline-validation to build an "
                    "unscored summary dataset."
                )
        selected_ids = sorted(only_ids)
    elif baseline_ids is not None:
        missing_raw = sorted(baseline_ids - raw_ids)
        if missing_raw:
            raise FileNotFoundError(
                "The ranking corpus references novels missing from the raw English "
                f"directory: {missing_raw}"
            )
        skipped_raw = sorted(raw_ids - baseline_ids)
        if skipped_raw:
            LOG.info(
                "Skipping %d raw novels not represented in the ranking corpus "
                "(examples: %s)",
                len(skipped_raw),
                skipped_raw[:10],
            )
        selected_ids = sorted(baseline_ids)
    else:
        selected_ids = sorted(raw_ids)

    if max_files > 0:
        selected_ids = selected_ids[:max_files]
    if not selected_ids:
        raise FileNotFoundError("No novels remain after applying selection filters")
    return [raw_index[novel_id_value] for novel_id_value in selected_ids]



def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.min_whitespace_tokens <= 0:
        raise ValueError("--min-whitespace-tokens must be positive")
    if args.chunk_summary_min_words > args.chunk_summary_max_words:
        raise ValueError("Chunk summary min words cannot exceed max words")
    if args.novel_summary_min_words > args.novel_summary_max_words:
        raise ValueError("Novel summary min words cannot exceed max words")

    ranking_corpus_path = args.ranking_dir / "corpus.jsonl"
    paragraph_qrels_path = args.ranking_dir / "qrels" / "test.tsv"
    ranking_corpus: dict[str, dict[str, Any]] | None = None

    if ranking_corpus_path.exists():
        LOG.info("Loading ranking corpus for strict alignment: %s", ranking_corpus_path)
        ranking_corpus = load_ranking_corpus(ranking_corpus_path)
    elif not args.skip_baseline_validation:
        raise FileNotFoundError(
            f"Missing {ranking_corpus_path}. Run prepare_detectiveqa_ranking.py first, "
            "or pass --skip-baseline-validation explicitly."
        )
    else:
        LOG.warning("Baseline validation skipped; mapping cannot be independently verified.")

    baseline_ids = ranking_novel_ids(ranking_corpus) if ranking_corpus is not None else None
    if baseline_ids is not None:
        LOG.info(
            "Ranking corpus covers %d novels; summary construction will use the same scope.",
            len(baseline_ids),
        )

    raw_index = discover_novel_files(args.input_dir)
    novel_files = select_novel_files(
        raw_index=raw_index,
        baseline_ids=baseline_ids,
        only_ids=set(args.only_novel_id),
        max_files=args.max_files,
    )

    prepared: list[tuple[Path, int, str, list[Paragraph], list[Chunk]]] = []
    total_paragraphs = 0
    total_chunks = 0

    # Parse, align, and chunk every selected file before making any paid API call.
    for path in novel_files:
        novel_id_value = extract_novel_id(path)
        title = extract_title(path)
        paragraphs = parse_novel(path)
        if ranking_corpus is not None:
            validate_alignment(novel_id_value, paragraphs, ranking_corpus)

        chunks = build_chunks(
            novel_id=novel_id_value,
            title=title,
            source_file=display_path(path),
            paragraphs=paragraphs,
            min_tokens=args.min_whitespace_tokens,
        )
        prepared.append((path, novel_id_value, title, paragraphs, chunks))
        total_paragraphs += len(paragraphs)
        total_chunks += len(chunks)

    # Only create/delete output files after all raw/baseline alignment checks pass.
    chunk_checkpoint_dir, novel_checkpoint_dir = prepare_output(
        args.output_dir,
        args.overwrite,
    )

    LOG.info(
        "Preflight passed: %d novels, %d paragraphs, %d level-1 chunks",
        len(prepared),
        total_paragraphs,
        total_chunks,
    )
    for _, novel_id_value, title, paragraphs, chunks in prepared:
        counts = [chunk.whitespace_tokens for chunk in chunks]
        LOG.info(
            "Novel %s (%s): %d paragraphs -> %d chunks; token range %d-%d",
            novel_id_value,
            title,
            len(paragraphs),
            len(chunks),
            min(counts),
            max(counts),
        )

    if args.plan_only:
        plan = {
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "min_whitespace_tokens": args.min_whitespace_tokens,
            "novels": [
                {
                    "novel_id": novel_id_value,
                    "title": title,
                    "source_file": str(path),
                    "paragraph_count": len(paragraphs),
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "start_paragraph_id": chunk.start_paragraph_id,
                            "end_paragraph_id": chunk.end_paragraph_id,
                            "paragraph_count": len(chunk.paragraphs),
                            "whitespace_tokens": chunk.whitespace_tokens,
                        }
                        for chunk in chunks
                    ],
                }
                for path, novel_id_value, title, paragraphs, chunks in prepared
            ],
        }
        atomic_write_json(args.output_dir / "plan.json", plan)
        LOG.info("Plan written to %s; no API calls were made.", args.output_dir / "plan.json")
        return

    api_key = load_api_key()
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Missing google-genai. Install dependencies with: "
            "pip install -U google-genai python-dotenv tqdm"
        ) from exc
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=180_000),
    )

    settings = ApiSettings(
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_retries=args.max_retries,
        initial_backoff_seconds=args.initial_backoff_seconds,
        max_backoff_seconds=args.max_backoff_seconds,
        request_pause_seconds=args.request_pause_seconds,
        diagnostics_dir=args.output_dir / "diagnostics" / "api_responses",
    )

    chunk_summary_rows: list[dict[str, Any]] = []
    novel_summary_rows: list[dict[str, Any]] = []
    raw_chunk_rows: list[dict[str, Any]] = []
    paragraph_mapping_rows: list[dict[str, Any]] = []
    paragraph_to_chunk: dict[str, str] = {}
    chunk_to_novel_summary: dict[str, str] = {}

    for path, novel_id_value, title, paragraphs, chunks in prepared:
        per_novel_chunk_summaries: list[dict[str, Any]] = []
        progress = tqdm(chunks, desc=f"Novel {novel_id_value}: level-1 summaries", unit="chunk")
        for chunk in progress:
            raw_chunk_rows.append(
                {
                    "_id": chunk.chunk_id,
                    "level": 1,
                    "novel_id": chunk.novel_id,
                    "title": chunk.title,
                    "source_file": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "start_paragraph_id": chunk.start_paragraph_id,
                    "end_paragraph_id": chunk.end_paragraph_id,
                    "paragraph_ids": chunk.paragraph_ids,
                    "paragraph_doc_ids": chunk.paragraph_doc_ids,
                    "source_text": chunk.source_text,
                    "source_whitespace_tokens": chunk.whitespace_tokens,
                    "source_character_count": chunk.character_count,
                    "source_sha256": sha256_text(chunk.source_text),
                }
            )
            summary_row = summarize_chunk(
                client=client,
                settings=settings,
                chunk=chunk,
                checkpoint_dir=chunk_checkpoint_dir,
                min_words=args.chunk_summary_min_words,
                max_words=args.chunk_summary_max_words,
                max_output_tokens=args.chunk_max_output_tokens,
            )
            per_novel_chunk_summaries.append(summary_row)
            chunk_summary_rows.append(summary_row)

            for paragraph in chunk.paragraphs:
                paragraph_document_id = doc_id(novel_id_value, paragraph.paragraph_id)
                if paragraph_document_id in paragraph_to_chunk:
                    raise AssertionError(f"Duplicate mapping for {paragraph_document_id}")
                paragraph_to_chunk[paragraph_document_id] = chunk.chunk_id
                paragraph_mapping_rows.append(
                    {
                        "paragraph_doc_id": paragraph_document_id,
                        "novel_id": novel_id_value,
                        "paragraph_id": paragraph.paragraph_id,
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "chunk_start_paragraph_id": chunk.start_paragraph_id,
                        "chunk_end_paragraph_id": chunk.end_paragraph_id,
                    }
                )

        novel_summary = summarize_novel(
            client=client,
            settings=settings,
            novel_id=novel_id_value,
            title=title,
            source_file=str(path),
            chunk_summaries=per_novel_chunk_summaries,
            checkpoint_dir=novel_checkpoint_dir,
            min_words=args.novel_summary_min_words,
            max_words=args.novel_summary_max_words,
            max_output_tokens=args.novel_max_output_tokens,
        )
        novel_summary_rows.append(novel_summary)
        for summary_row in per_novel_chunk_summaries:
            chunk_to_novel_summary[str(summary_row["_id"])] = str(novel_summary["_id"])

    # Final global coverage checks before writing dataset files.
    expected_doc_ids = {
        doc_id(novel_id_value, paragraph.paragraph_id)
        for _, novel_id_value, _, paragraphs, _ in prepared
        for paragraph in paragraphs
    }
    if set(paragraph_to_chunk) != expected_doc_ids:
        missing = sorted(expected_doc_ids - set(paragraph_to_chunk))
        extra = sorted(set(paragraph_to_chunk) - expected_doc_ids)
        raise AssertionError(f"Global mapping mismatch: missing={missing[:5]}, extra={extra[:5]}")

    dataset_files = {
        "chunks.jsonl": raw_chunk_rows,
        "chunk_summaries.jsonl": sorted(
            chunk_summary_rows, key=lambda row: (int(row["novel_id"]), int(row["chunk_index"]))
        ),
        "novel_summaries.jsonl": sorted(
            novel_summary_rows, key=lambda row: int(row["novel_id"])
        ),
        "paragraph_to_chunk.jsonl": sorted(
            paragraph_mapping_rows,
            key=lambda row: (int(row["novel_id"]), int(row["paragraph_id"])),
        ),
    }
    for filename, rows in dataset_files.items():
        with (args.output_dir / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                jsonl_write(handle, row)

    qrel_counts: dict[str, int] = {}
    if paragraph_qrels_path.exists():
        qrel_counts = propagate_qrels(
            paragraph_qrels_path=paragraph_qrels_path,
            paragraph_to_chunk=paragraph_to_chunk,
            chunk_to_novel_summary=chunk_to_novel_summary,
            processed_novel_ids={novel_id_value for _, novel_id_value, _, _, _ in prepared},
            output_qrels_dir=args.output_dir / "qrels",
        )
    else:
        LOG.warning("Paragraph qrels not found at %s; no propagated qrels written.", paragraph_qrels_path)

    manifest = {
        "dataset": "DetectiveQA English two-level summary hierarchy",
        "created_at_unix": int(time.time()),
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "source_novel_dir": str(args.input_dir),
        "ranking_dir": str(args.ranking_dir),
        "baseline_alignment_validated": ranking_corpus is not None,
        "novel_scope": "ranking_corpus" if ranking_corpus is not None else "all_selected_raw",
        "baseline_novel_ids": sorted(baseline_ids) if baseline_ids is not None else None,
        "processed_novel_ids": sorted(
            novel_id_value for _, novel_id_value, _, _, _ in prepared
        ),
        "paragraph_doc_id_format": "dqa-en-{novel_id}-p{paragraph_id}",
        "chunk_id_format": "dqa-en-{novel_id}-c{chunk_index:04d}",
        "chunking": {
            "method": "contiguous complete paragraphs, whitespace-token threshold",
            "min_whitespace_tokens": args.min_whitespace_tokens,
            "tail_policy": "merge undersized final tail into previous chunk",
            "overlap": 0,
        },
        "summarization": {
            "levels": 2,
            "level_1": "independent chunk summary from raw numbered paragraphs",
            "level_2": "whole-novel summary from ordered level-1 summaries",
            "temperature": args.temperature,
            "seed": args.seed,
            "chunk_summary_words": [
                args.chunk_summary_min_words,
                args.chunk_summary_max_words,
            ],
            "novel_summary_words": [
                args.novel_summary_min_words,
                args.novel_summary_max_words,
            ],
        },
        "counts": {
            "novels": len(prepared),
            "paragraphs": len(paragraph_mapping_rows),
            "chunks": len(raw_chunk_rows),
            "chunk_summaries": len(chunk_summary_rows),
            "novel_summaries": len(novel_summary_rows),
            **qrel_counts,
        },
        "files": sorted(dataset_files) + (["qrels/chunks.tsv", "qrels/novels.tsv"] if qrel_counts else []),
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)

    LOG.info("Done. Output dataset: %s", args.output_dir.resolve())
    LOG.info(
        "Generated %d chunk summaries and %d novel summaries.",
        len(chunk_summary_rows),
        len(novel_summary_rows),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.error("Interrupted. Checkpoints are preserved; rerun without --overwrite to resume.")
        sys.exit(130)
