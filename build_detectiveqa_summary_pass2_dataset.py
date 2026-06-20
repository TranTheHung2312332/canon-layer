

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
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

from dotenv import load_dotenv
from tqdm import tqdm

LOG = logging.getLogger("detectiveqa_summary_pass2_builder")

# Fallback only. By default, main() replaces args.model with the pass-1
# manifest model when available.
DEFAULT_MODEL = "gemma-4-26b-a4b-it"
PROMPT_VERSION = "detectiveqa-summary-pass2-v1.1-fair"

SYSTEM_INSTRUCTION = """You revise English-language fictional narrative summaries for retrieval.
Preserve the important characters, actions, discoveries, locations, motives, and
consequences explicitly supported by the supplied summaries. Resolve references
only when the supplied material supports the resolution. Do not invent facts, do
not add literary criticism, and do not speculate.

For chunk summaries, do not add future events unless those events are already
supported by the current chunk summary or the previous contextual summary
supplied to you.

Write plain English prose only. Return only the final summary. Do not output
reasoning, drafts, outlines, checklists, self-evaluation, or word-count commentary."""

CHUNK_PASS2_PROMPT = """Create a pass-2 contextual summary for the CURRENT chunk of a novel.

Inputs:
1. PREVIOUS PASS-2 CHUNK SUMMARY: context from the immediately preceding chunk.
2. CURRENT PASS-1 CHUNK SUMMARY: the primary source for the current chunk.
3. GLOBAL PASS-1 NOVEL SUMMARY: broad background for names, roles, and setting.

Requirements:
- Rewrite the CURRENT chunk summary into a clearer, self-contained retrieval summary.
- The output must describe the CURRENT chunk only.
- Preserve names, aliases, concrete actions, discoveries, locations, explicit motives, and narrative order when supported.
- Include details that could help retrieve this chunk in response to a factual, causal, character, or event question.
- You may use the previous pass-2 summary to resolve references, continuity, and local context.
- You may use the global pass-1 novel summary only as background context.
- Do NOT import later events, final outcomes, hidden identities, causes, or consequences from the global summary unless they are also supported by the current chunk summary or the previous pass-2 summary.
- Do NOT predict what happens next.
- Do NOT mention chunk IDs, paragraph IDs, pass 1, pass 2, or the instructions.
- Write approximately {min_words}-{max_words} words.

Novel: {title}
Novel ID: {novel_id}
Current source paragraph range: [{start_paragraph_id}] to [{end_paragraph_id}]

PREVIOUS PASS-2 CHUNK SUMMARY
{previous_pass2_summary}

CURRENT PASS-1 CHUNK SUMMARY
{current_pass1_summary}

GLOBAL PASS-1 NOVEL SUMMARY
{global_pass1_summary}
"""

NOVEL_PASS2_PROMPT = """Create a whole-novel summary from ordered contextual chunk summaries.

Inputs:
1. ORDERED CONTEXTUAL CHUNK SUMMARIES: the main source.
2. GLOBAL PASS-1 NOVEL SUMMARY: auxiliary context.

Requirements:
- Write approximately {min_words}-{max_words} words in plain English prose.
- Synthesize the summaries into one coherent account of the major plot progression.
- Preserve chronological order, important characters, major actions, discoveries, conflicts, and outcomes.
- Merge repeated information instead of listing each chunk separately.
- Do not invent missing links or facts beyond the supplied summaries.
- Do not mention chunk IDs, paragraph IDs, pass 1, pass 2, or the summarization process.

Novel: {title}
Novel ID: {novel_id}

GLOBAL PASS-1 NOVEL SUMMARY
{global_pass1_summary}

ORDERED CONTEXTUAL CHUNK SUMMARIES
{ordered_pass2_summaries}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pass-2 contextual summaries from DetectiveQA pass-1 summaries."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/detectiveqa-summary-en"),
        help="Pass-1 summary dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/detectiveqa-summary-pass2-en"),
        help="Output pass-2 summary dataset directory.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Generation model for pass-2. Defaults to the model recorded in the "
            "pass-1 manifest; falls back to gemma-4-26b-a4b-it."
        ),
    )
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
    parser.add_argument("--request-pause-seconds", type=float, default=0.5)
    parser.add_argument("--only-novel-id", type=int, action="append", default=[])
    parser.add_argument("--max-novels", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield row


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Atomic JSONL writer to avoid partial final dataset files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def whitespace_token_count(text: str) -> int:
    return len(text.split())


def load_api_key() -> str:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Add GEMINI_API_KEY=... or GOOGLE_API_KEY=... to .env")
    return key


def exception_status_code(exc: Exception) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"\b(429|500|502|503|504)\b", str(exc))
    return int(match.group(1)) if match else None


def generation_request_sha256(
    *,
    prompt: str,
    model: str,
    temperature: float,
    seed: int,
    max_output_tokens: int,
) -> str:
    """Fingerprint every input that can change a cached generation."""
    request = {
        "prompt_version": PROMPT_VERSION,
        "system_instruction": SYSTEM_INSTRUCTION,
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "max_output_tokens": max_output_tokens,
        "generation_config_policy": {
            "candidate_count": 1,
            "response_mime_type": "text/plain",
            "automatic_function_calling": "disabled",
            "function_calling_mode": "NONE",
            "thinking_level": "MINIMAL",
            "include_thoughts": False,
        },
    }
    return sha256_text(json.dumps(request, ensure_ascii=False, sort_keys=True))


def checkpoint_ok(payload: dict[str, Any], *, request_sha256: str) -> bool:
    return (
        payload.get("request_sha256") == request_sha256
        and bool(str(payload.get("summary", "")).strip())
    )


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def extract_visible_response_text(response: Any) -> str:
    """Extract visible text; never use hidden thought parts as summaries."""
    fragments: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if bool(getattr(part, "thought", False)):
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                fragments.append(text.strip())
    if fragments:
        return "\n".join(fragments).strip()

    try:
        text = getattr(response, "text", None)
    except Exception:
        text = None
    return text.strip() if isinstance(text, str) else ""


def generate_text_with_retry(
    *,
    client: Any,
    model: str,
    temperature: float,
    seed: int,
    max_retries: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
    request_pause_seconds: float,
    prompt: str,
    max_output_tokens: int,
) -> str:
    from google.genai import types

    retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=temperature,
                    seed=seed,
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
                ),
            )
            text = extract_visible_response_text(response)
            if not text:
                raise RuntimeError("The API returned no visible text response")
            if request_pause_seconds > 0:
                time.sleep(request_pause_seconds)
            return text
        except Exception as exc:
            status = exception_status_code(exc)
            explicit_retryable = getattr(exc, "retryable", None)
            retryable = (
                bool(explicit_retryable)
                if explicit_retryable is not None
                else status in retryable_statuses or status is None
            )
            if attempt >= max_retries or not retryable:
                raise
            backoff = min(max_backoff_seconds, initial_backoff_seconds * (2**attempt))
            wait_seconds = backoff + random.uniform(0.0, min(1.0, backoff * 0.1))
            LOG.warning(
                "API call failed (status=%s, attempt=%d/%d): %s. Retrying in %.1fs",
                status,
                attempt + 1,
                max_retries + 1,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise AssertionError("Unreachable retry loop")


def load_input_dataset(input_dir: Path) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    required = [
        "chunks.jsonl",
        "chunk_summaries.jsonl",
        "novel_summaries.jsonl",
        "paragraph_to_chunk.jsonl",
        "manifest.json",
    ]
    for name in required:
        path = input_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required input file: {path}")

    chunks = list(iter_jsonl(input_dir / "chunks.jsonl"))
    chunk_summaries = list(iter_jsonl(input_dir / "chunk_summaries.jsonl"))
    novel_summaries = list(iter_jsonl(input_dir / "novel_summaries.jsonl"))
    paragraph_to_chunk = list(iter_jsonl(input_dir / "paragraph_to_chunk.jsonl"))
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest: {input_dir / 'manifest.json'}")
    return chunks, chunk_summaries, novel_summaries, paragraph_to_chunk, manifest


def select_novels(
    chunk_summaries: Sequence[dict[str, Any]],
    only_ids: set[int],
    max_novels: int,
) -> list[int]:
    novel_ids = sorted({int(row["novel_id"]) for row in chunk_summaries})
    if only_ids:
        missing = sorted(only_ids - set(novel_ids))
        if missing:
            raise ValueError(f"Requested novel IDs are not in input summaries: {missing}")
        novel_ids = [novel_id for novel_id in novel_ids if novel_id in only_ids]
    if max_novels > 0:
        novel_ids = novel_ids[:max_novels]
    if not novel_ids:
        raise ValueError("No novels selected")
    return novel_ids


def filter_qrels_file(input_path: Path, output_path: Path, allowed_ids: set[str]) -> int:
    if not input_path.exists():
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    written = 0
    with input_path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        header = src.readline()
        dst.write(header)
        for line in src:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1] in allowed_ids:
                dst.write(line)
                written += 1
    tmp.replace(output_path)
    return written


def copy_or_filter_static_files(
    *,
    input_dir: Path,
    output_dir: Path,
    selected_novel_ids: set[int],
    chunks: Sequence[dict[str, Any]],
    paragraph_to_chunk: Sequence[dict[str, Any]],
    pass2_chunk_summaries: Sequence[dict[str, Any]],
    pass2_novel_summaries: Sequence[dict[str, Any]],
) -> dict[str, int]:
    selected_chunk_ids = {str(row["_id"]) for row in pass2_chunk_summaries}
    selected_novel_summary_ids = {str(row["_id"]) for row in pass2_novel_summaries}

    selected_chunks = [row for row in chunks if int(row["novel_id"]) in selected_novel_ids]
    selected_mapping = [row for row in paragraph_to_chunk if int(row["novel_id"]) in selected_novel_ids]

    write_jsonl(
        output_dir / "chunks.jsonl",
        sorted(selected_chunks, key=lambda r: (int(r["novel_id"]), int(r["chunk_index"]))),
    )
    write_jsonl(
        output_dir / "chunk_summaries.jsonl",
        sorted(pass2_chunk_summaries, key=lambda r: (int(r["novel_id"]), int(r["chunk_index"]))),
    )
    write_jsonl(
        output_dir / "novel_summaries.jsonl",
        sorted(pass2_novel_summaries, key=lambda r: int(r["novel_id"])),
    )
    write_jsonl(
        output_dir / "paragraph_to_chunk.jsonl",
        sorted(selected_mapping, key=lambda r: (int(r["novel_id"]), int(r["paragraph_id"]))),
    )

    qrel_chunk_count = filter_qrels_file(
        input_dir / "qrels" / "chunks.tsv",
        output_dir / "qrels" / "chunks.tsv",
        selected_chunk_ids,
    )
    qrel_novel_count = filter_qrels_file(
        input_dir / "qrels" / "novels.tsv",
        output_dir / "qrels" / "novels.tsv",
        selected_novel_summary_ids,
    )

    return {
        "chunks": len(selected_chunks),
        "chunk_summaries": len(pass2_chunk_summaries),
        "novel_summaries": len(pass2_novel_summaries),
        "paragraph_mappings": len(selected_mapping),
        "chunk_qrels": qrel_chunk_count,
        "novel_qrels": qrel_novel_count,
    }


def make_first_chunk_identity(
    row: dict[str, Any],
    *,
    source_pass1_model: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = str(row.get("summary", "")).strip()
    output = dict(row)
    output.update(
        {
            "summary": summary,
            "summary_whitespace_tokens": whitespace_token_count(summary),
            "model": source_pass1_model or str(row.get("model", "")),
            "temperature": row.get("temperature", args.temperature),
            "seed": row.get("seed", args.seed),
            "max_output_tokens": row.get("max_output_tokens", args.chunk_max_output_tokens),
            "prompt_version": PROMPT_VERSION,
            "pass2_method": "first_chunk_identity_from_pass1",
            "pass1_prompt_version": row.get("prompt_version"),
            "pass1_model": row.get("model"),
            "pass1_summary_sha256": sha256_text(summary),
            "source_sha256": sha256_text(summary),
            "request_sha256": sha256_text(
                json.dumps(
                    {
                        "prompt_version": PROMPT_VERSION,
                        "method": "first_chunk_identity_from_pass1",
                        "pass1_summary": summary,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
        }
    )
    return output


def summarize_chunk_pass2(
    *,
    client: Any,
    args: argparse.Namespace,
    row: dict[str, Any],
    previous_pass2_summary: str,
    global_pass1_summary: str,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    current_pass1_summary = str(row.get("summary", "")).strip()
    prompt = CHUNK_PASS2_PROMPT.format(
        min_words=args.chunk_summary_min_words,
        max_words=args.chunk_summary_max_words,
        title=row.get("title", ""),
        novel_id=int(row["novel_id"]),
        start_paragraph_id=int(row["start_paragraph_id"]),
        end_paragraph_id=int(row["end_paragraph_id"]),
        previous_pass2_summary=previous_pass2_summary,
        current_pass1_summary=current_pass1_summary,
        global_pass1_summary=global_pass1_summary,
    )
    source_material = json.dumps(
        {
            "previous_pass2_summary": previous_pass2_summary,
            "current_pass1_summary": current_pass1_summary,
            "global_pass1_summary": global_pass1_summary,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    source_hash = sha256_text(source_material)
    request_hash = generation_request_sha256(
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_output_tokens=args.chunk_max_output_tokens,
    )
    checkpoint_path = checkpoint_dir / f"{row['_id']}.json"
    cached = load_json_if_exists(checkpoint_path)
    if cached and checkpoint_ok(cached, request_sha256=request_hash):
        LOG.info("Reusing pass-2 chunk checkpoint %s", row["_id"])
        return cached

    summary = generate_text_with_retry(
        client=client,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_retries=args.max_retries,
        initial_backoff_seconds=args.initial_backoff_seconds,
        max_backoff_seconds=args.max_backoff_seconds,
        request_pause_seconds=args.request_pause_seconds,
        prompt=prompt,
        max_output_tokens=args.chunk_max_output_tokens,
    )
    output = dict(row)
    output.update(
        {
            "summary": summary,
            "summary_whitespace_tokens": whitespace_token_count(summary),
            "model": args.model,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_output_tokens": args.chunk_max_output_tokens,
            "prompt_version": PROMPT_VERSION,
            "pass2_method": "previous_pass2_current_pass1_global_pass1",
            "pass1_prompt_version": row.get("prompt_version"),
            "pass1_model": row.get("model"),
            "pass1_summary_sha256": sha256_text(current_pass1_summary),
            "previous_pass2_summary_sha256": sha256_text(previous_pass2_summary),
            "global_pass1_summary_sha256": sha256_text(global_pass1_summary),
            "source_sha256": source_hash,
            "request_sha256": request_hash,
        }
    )
    atomic_write_json(checkpoint_path, output)
    return output


def summarize_novel_pass2(
    *,
    client: Any,
    args: argparse.Namespace,
    pass1_novel_summary: dict[str, Any],
    pass2_chunk_summaries: Sequence[dict[str, Any]],
    checkpoint_dir: Path,
) -> dict[str, Any]:
    ordered = sorted(pass2_chunk_summaries, key=lambda r: int(r["chunk_index"]))
    ordered_text = "\n\n".join(
        f"Passage {int(row['chunk_index']) + 1} "
        f"(source paragraphs [{row['start_paragraph_id']}]-[{row['end_paragraph_id']}]):\n{row['summary']}"
        for row in ordered
    )
    global_pass1_summary = str(pass1_novel_summary.get("summary", "")).strip()
    prompt = NOVEL_PASS2_PROMPT.format(
        min_words=args.novel_summary_min_words,
        max_words=args.novel_summary_max_words,
        title=pass1_novel_summary.get("title", ""),
        novel_id=int(pass1_novel_summary["novel_id"]),
        global_pass1_summary=global_pass1_summary,
        ordered_pass2_summaries=ordered_text,
    )
    source_material = json.dumps(
        {
            "global_pass1_summary": global_pass1_summary,
            "ordered_pass2_summaries": ordered_text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    source_hash = sha256_text(source_material)
    request_hash = generation_request_sha256(
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_output_tokens=args.novel_max_output_tokens,
    )
    checkpoint_path = checkpoint_dir / f"{pass1_novel_summary['_id']}.json"
    cached = load_json_if_exists(checkpoint_path)
    if cached and checkpoint_ok(cached, request_sha256=request_hash):
        LOG.info("Reusing pass-2 novel checkpoint %s", pass1_novel_summary["_id"])
        return cached

    summary = generate_text_with_retry(
        client=client,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_retries=args.max_retries,
        initial_backoff_seconds=args.initial_backoff_seconds,
        max_backoff_seconds=args.max_backoff_seconds,
        request_pause_seconds=args.request_pause_seconds,
        prompt=prompt,
        max_output_tokens=args.novel_max_output_tokens,
    )
    output = dict(pass1_novel_summary)
    output.update(
        {
            "summary": summary,
            "summary_whitespace_tokens": whitespace_token_count(summary),
            "model": args.model,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_output_tokens": args.novel_max_output_tokens,
            "prompt_version": PROMPT_VERSION,
            "pass2_method": "whole_novel_from_ordered_pass2_chunks_plus_global_pass1",
            "pass1_prompt_version": pass1_novel_summary.get("prompt_version"),
            "pass1_model": pass1_novel_summary.get("model"),
            "pass1_summary_sha256": sha256_text(global_pass1_summary),
            "source_sha256": source_hash,
            "request_sha256": request_hash,
        }
    )
    atomic_write_json(checkpoint_path, output)
    return output


def prepare_output(output_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_checkpoint_dir = output_dir / "checkpoints" / "chunks"
    novel_checkpoint_dir = output_dir / "checkpoints" / "novels"
    chunk_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    novel_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return chunk_checkpoint_dir, novel_checkpoint_dir


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.chunk_summary_min_words > args.chunk_summary_max_words:
        raise ValueError("Chunk summary min words cannot exceed max words")
    if args.novel_summary_min_words > args.novel_summary_max_words:
        raise ValueError("Novel summary min words cannot exceed max words")
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("Refusing to write output into the input dataset directory")

    (
        chunks,
        pass1_chunk_summaries,
        pass1_novel_summaries,
        paragraph_to_chunk,
        pass1_manifest,
    ) = load_input_dataset(args.input_dir)

    pass1_model = str(pass1_manifest.get("model") or "").strip()
    if args.model is None:
        args.model = pass1_model or DEFAULT_MODEL
    elif pass1_model and args.model != pass1_model:
        LOG.warning(
            "Pass-2 model %s differs from pass-1 manifest model %s; benchmark may include a model confound.",
            args.model,
            pass1_model,
        )

    selected_novel_ids = select_novels(
        pass1_chunk_summaries,
        set(args.only_novel_id),
        args.max_novels,
    )
    selected_novel_id_set = set(selected_novel_ids)

    by_novel_chunks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pass1_chunk_summaries:
        novel_id = int(row["novel_id"])
        if novel_id in selected_novel_id_set:
            by_novel_chunks[novel_id].append(row)
    for rows in by_novel_chunks.values():
        rows.sort(key=lambda r: int(r["chunk_index"]))

    pass1_novel_by_id = {int(row["novel_id"]): row for row in pass1_novel_summaries}
    missing_novel_summaries = sorted(selected_novel_id_set - set(pass1_novel_by_id))
    if missing_novel_summaries:
        raise ValueError(f"Missing pass-1 novel summaries for IDs: {missing_novel_summaries}")

    total_chunks = sum(len(rows) for rows in by_novel_chunks.values())
    LOG.info(
        "Plan: %d novels, %d chunks -> pass-2 contextual summaries; model=%s",
        len(selected_novel_ids),
        total_chunks,
        args.model,
    )
    for novel_id in selected_novel_ids:
        rows = by_novel_chunks[novel_id]
        LOG.info("Novel %s: %d chunks", novel_id, len(rows))

    if args.plan_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            args.output_dir / "pass2_plan.json",
            {
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "pass1_manifest_model": pass1_model or None,
                "novels": [
                    {
                        "novel_id": novel_id,
                        "chunk_count": len(by_novel_chunks[novel_id]),
                        "first_chunk_id": by_novel_chunks[novel_id][0]["_id"],
                        "last_chunk_id": by_novel_chunks[novel_id][-1]["_id"],
                    }
                    for novel_id in selected_novel_ids
                ],
            },
        )
        LOG.info("Plan written to %s; no API calls were made.", args.output_dir / "pass2_plan.json")
        return

    chunk_checkpoint_dir, novel_checkpoint_dir = prepare_output(args.output_dir, args.overwrite)

    api_key = load_api_key()
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install dependencies: pip install -U google-genai python-dotenv tqdm") from exc

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=180_000),
    )

    all_pass2_chunk_summaries: list[dict[str, Any]] = []
    all_pass2_novel_summaries: list[dict[str, Any]] = []
    source_pass1_model = pass1_model or None

    for novel_id in selected_novel_ids:
        rows = by_novel_chunks[novel_id]
        global_pass1_summary = str(pass1_novel_by_id[novel_id].get("summary", "")).strip()
        per_novel_pass2: list[dict[str, Any]] = []
        previous_pass2_summary = ""

        progress = tqdm(rows, desc=f"Novel {novel_id}: pass-2 chunks", unit="chunk")
        for idx, row in enumerate(progress):
            if idx == 0:
                output = make_first_chunk_identity(
                    row,
                    source_pass1_model=source_pass1_model,
                    args=args,
                )
                atomic_write_json(chunk_checkpoint_dir / f"{row['_id']}.json", output)
            else:
                output = summarize_chunk_pass2(
                    client=client,
                    args=args,
                    row=row,
                    previous_pass2_summary=previous_pass2_summary,
                    global_pass1_summary=global_pass1_summary,
                    checkpoint_dir=chunk_checkpoint_dir,
                )
            per_novel_pass2.append(output)
            all_pass2_chunk_summaries.append(output)
            previous_pass2_summary = str(output.get("summary", "")).strip()

        novel_output = summarize_novel_pass2(
            client=client,
            args=args,
            pass1_novel_summary=pass1_novel_by_id[novel_id],
            pass2_chunk_summaries=per_novel_pass2,
            checkpoint_dir=novel_checkpoint_dir,
        )
        all_pass2_novel_summaries.append(novel_output)

    file_counts = copy_or_filter_static_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        selected_novel_ids=selected_novel_id_set,
        chunks=chunks,
        paragraph_to_chunk=paragraph_to_chunk,
        pass2_chunk_summaries=all_pass2_chunk_summaries,
        pass2_novel_summaries=all_pass2_novel_summaries,
    )

    manifest = {
        "dataset": "DetectiveQA English pass-2 contextual summary hierarchy",
        "created_at_unix": int(time.time()),
        "input_summary_dir": str(args.input_dir),
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "source_pass1_manifest": {
            "dataset": pass1_manifest.get("dataset"),
            "prompt_version": pass1_manifest.get("prompt_version"),
            "model": pass1_manifest.get("model"),
            "chunk_id_format": pass1_manifest.get("chunk_id_format"),
            "paragraph_doc_id_format": pass1_manifest.get("paragraph_doc_id_format"),
            "chunking": pass1_manifest.get("chunking"),
            "summarization": pass1_manifest.get("summarization"),
        },
        "structure": {
            "same_as_pass1": True,
            "files_compatible_with_pass1_summary_baseline": True,
            "chunk_ids_preserved": True,
            "paragraph_to_chunk_preserved": True,
            "qrels_preserved_or_filtered": True,
        },
        "generation_config": {
            "aligned_with_pass1": True,
            "candidate_count": 1,
            "response_mime_type": "text/plain",
            "automatic_function_calling": "disabled",
            "function_calling_mode": "NONE",
            "thinking_level": "MINIMAL",
            "include_thoughts": False,
            "client_timeout_ms": 180_000,
        },
        "pass2_policy": {
            "chunk_0": "identity: pass-2 summary equals pass-1 summary",
            "chunk_i": "previous pass-2 chunk summary + current pass-1 chunk summary + global pass-1 summary",
            "global_pass2": "ordered pass-2 chunk summaries + global pass-1 summary",
            "future_leakage_guard": "global pass-1 is auxiliary context only; prompts forbid importing future facts into chunk summaries unless supported by current or previous summaries",
        },
        "summarization": {
            "levels": 2,
            "level_1": "contextual chunk summary from previous pass-2 chunk summary, current pass-1 chunk summary, and global pass-1 summary",
            "level_2": "whole-novel summary from ordered pass-2 chunk summaries plus global pass-1 summary",
            "temperature": args.temperature,
            "seed": args.seed,
            "chunk_summary_words": [args.chunk_summary_min_words, args.chunk_summary_max_words],
            "novel_summary_words": [args.novel_summary_min_words, args.novel_summary_max_words],
        },
        "counts": {
            "novels": len(selected_novel_ids),
            **file_counts,
        },
        "files": [
            "chunks.jsonl",
            "chunk_summaries.jsonl",
            "novel_summaries.jsonl",
            "paragraph_to_chunk.jsonl",
            "qrels/chunks.tsv",
            "qrels/novels.tsv",
        ],
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)

    LOG.info("Done. Output dataset: %s", args.output_dir.resolve())
    LOG.info(
        "Generated %d pass-2 chunk summaries and %d pass-2 novel summaries.",
        len(all_pass2_chunk_summaries),
        len(all_pass2_novel_summaries),
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.error("Interrupted. Checkpoints are preserved; rerun without --overwrite to resume.")
        sys.exit(130)
