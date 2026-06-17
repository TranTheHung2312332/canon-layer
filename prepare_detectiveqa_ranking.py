#!/usr/bin/env python3
"""Download English DetectiveQA and convert it to a ranking benchmark.

Output is BEIR-like:
  corpus.jsonl          {"_id", "title", "text"}
  queries.jsonl         {"_id", "text"}
  qrels/test.tsv        query-id, corpus-id, score
  query_metadata.jsonl  DetectiveQA-specific metadata and candidate scope
  manifest.json         conversion statistics

Recommended evaluation:
  * query = question only
  * candidates = paragraphs from the same novel
  * positives = unique clue_position values greater than zero
  * clue_position == -1 is implicit reasoning, not a retrievable passage
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "Phospheneser/DetectiveQA"
PARAGRAPH_RE = re.compile(r"(?m)^[ \t]*\[(\d+)\][ \t]*")
NOVEL_ID_RE = re.compile(r"^(\d+)-")
LOG = logging.getLogger("detectiveqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/detectiveqa-ranking-en"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/detectiveqa"))
    parser.add_argument(
        "--annotation-part",
        choices=["human_anno", "AIsup_anno", "both"],
        default="human_anno",
        help="Use human annotations by default; 'both' also includes AI-assisted data.",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def jsonl_write(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def to_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def doc_id(novel_id: int, paragraph_id: int) -> str:
    return f"dqa-en-{novel_id}-p{paragraph_id}"


def query_id(annotation_part: str, novel_id: int, question_index: int) -> str:
    return f"dqa-en-{annotation_part}-{novel_id}-q{question_index:03d}"


def parse_novel(path: Path) -> dict[int, str]:
    """Parse `[paragraph_id] text` records, including multiline paragraphs."""
    raw = path.read_text(encoding="utf-8-sig")
    markers = list(PARAGRAPH_RE.finditer(raw))
    if not markers:
        raise ValueError(f"No numbered paragraphs found in {path}")

    paragraphs: dict[int, str] = {}
    for index, marker in enumerate(markers):
        paragraph_id = int(marker.group(1))
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(raw)
        if paragraph_id in paragraphs:
            raise ValueError(f"Duplicate paragraph [{paragraph_id}] in {path}")
        paragraphs[paragraph_id] = clean_text(raw[start:end])
    return paragraphs


def load_annotation_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise ValueError(f"Unexpected JSON structure in {path}")


def novel_file_index(novel_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in novel_dir.glob("*.txt"):
        match = NOVEL_ID_RE.match(path.name)
        if match:
            novel_id = int(match.group(1))
            if novel_id in result:
                raise ValueError(f"Duplicate novel ID {novel_id}")
            result[novel_id] = path
    if not result:
        raise FileNotFoundError(f"No English novel files found in {novel_dir}")
    return result


def download_snapshot(args: argparse.Namespace, parts: list[str]) -> tuple[Path, str | None]:
    patterns = ["novel_data_en/*.txt", "README.md"]
    patterns += [f"anno_data_en/{part}/*.json" for part in parts]

    snapshot = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=patterns,
        local_dir=str(args.cache_dir),
    )

    try:
        commit_sha = HfApi().dataset_info(REPO_ID, revision=args.revision).sha
    except Exception:
        commit_sha = None
    return Path(snapshot), commit_sha


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty; use --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    (path / "qrels").mkdir(exist_ok=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parts = ["human_anno", "AIsup_anno"] if args.annotation_part == "both" else [args.annotation_part]
    prepare_output_dir(args.output_dir, args.overwrite)
    snapshot_root, commit_sha = download_snapshot(args, parts)

    novel_paths = novel_file_index(snapshot_root / "novel_data_en")

    # Collect annotation records first so the corpus only contains novels used
    # by the selected annotation source(s).
    jobs: list[tuple[str, Path, dict[str, Any]]] = []
    required_novel_ids: set[int] = set()
    for part in parts:
        annotation_dir = snapshot_root / "anno_data_en" / part
        for path in sorted(annotation_dir.glob("*.json")):
            fallback_id = int(path.stem) if path.stem.isdigit() else None
            for record in load_annotation_records(path):
                novel_id = to_int(record.get("novel_id"), fallback_id)
                if novel_id is None:
                    LOG.warning("Skipping record without novel_id: %s", path)
                    continue
                record = dict(record)
                record["novel_id"] = novel_id
                jobs.append((part, path, record))
                required_novel_ids.add(novel_id)

    missing_novels = sorted(required_novel_ids - novel_paths.keys())
    if missing_novels:
        raise FileNotFoundError(f"Missing English novel files for IDs: {missing_novels}")

    corpus_path = args.output_dir / "corpus.jsonl"
    queries_path = args.output_dir / "queries.jsonl"
    qrels_path = args.output_dir / "qrels" / "test.tsv"
    metadata_path = args.output_dir / "query_metadata.jsonl"

    parsed_novels: dict[int, dict[int, str]] = {}
    counts = Counter()

    # Build paragraph corpus.
    with corpus_path.open("w", encoding="utf-8") as corpus_file:
        for novel_id in sorted(required_novel_ids):
            path = novel_paths[novel_id]
            paragraphs = parse_novel(path)
            parsed_novels[novel_id] = paragraphs

            # The release keeps Chinese names in filenames even in novel_data_en;
            # the folder, not the filename, determines the text language.
            title = path.stem.split("-", maxsplit=2)[1] if "-" in path.stem else path.stem
            for paragraph_id, text in sorted(paragraphs.items()):
                jsonl_write(
                    corpus_file,
                    {
                        "_id": doc_id(novel_id, paragraph_id),
                        "title": title,
                        "text": text,
                    },
                )
                counts["documents"] += 1
            counts["novels"] += 1

    # Build question-only queries and paragraph-level qrels.
    with (
        queries_path.open("w", encoding="utf-8") as queries_file,
        qrels_path.open("w", encoding="utf-8") as qrels_file,
        metadata_path.open("w", encoding="utf-8") as metadata_file,
    ):
        qrels_file.write("query-id\tcorpus-id\tscore\n")

        for part, annotation_path, record in jobs:
            novel_id = int(record["novel_id"])
            paragraphs = parsed_novels[novel_id]
            questions = record.get("questions", [])
            if not isinstance(questions, list):
                LOG.warning("Invalid questions field: %s", annotation_path)
                continue

            for question_index, item in enumerate(questions):
                counts["questions_seen"] += 1
                if not isinstance(item, dict):
                    continue

                qid = query_id(part, novel_id, question_index)
                question = clean_text(str(item.get("question", "")))
                clue_positions = item.get("clue_position", [])
                reasoning = item.get("reasoning", [])
                answer_position = to_int(item.get("answer_position"))

                if not question or not isinstance(clue_positions, list):
                    LOG.warning("Skipping malformed question %s", qid)
                    continue

                raw_positive_positions: list[int] = []
                missing_positions: list[int] = []
                implicit_count = 0

                for value in clue_positions:
                    position = to_int(value)
                    if position == -1:
                        implicit_count += 1
                    elif position is not None and position > 0:
                        if position in paragraphs:
                            raw_positive_positions.append(position)
                        else:
                            missing_positions.append(position)

                position_counts = Counter(raw_positive_positions)
                gold_positions = sorted(position_counts)
                counts["duplicate_gold_entries_removed"] += len(raw_positive_positions) - len(gold_positions)
                counts["missing_gold_positions"] += len(set(missing_positions))

                options = item.get("options") if isinstance(item.get("options"), dict) else {}
                answer_label = item.get("answer")
                prefix_eligible = bool(
                    answer_position is not None
                    and gold_positions
                    and all(position <= answer_position for position in gold_positions)
                )

                # Keep answer/options/reasoning in metadata for analysis only.
                # They must not be fed to the retriever in the primary setting.
                jsonl_write(
                    metadata_file,
                    {
                        "query_id": qid,
                        "language": "en",
                        "annotation_part": part,
                        "annotation_file": str(annotation_path.relative_to(snapshot_root)),
                        "novel_id": novel_id,
                        "candidate_filter": {"novel_id": novel_id},
                        "question": question,
                        "options": options,
                        "answer_label": answer_label,
                        "correct_answer_text": options.get(answer_label),
                        "reasoning": reasoning,
                        "clue_position": clue_positions,
                        "answer_position": answer_position,
                        "gold_paragraph_ids": gold_positions,
                        "gold_doc_ids": [doc_id(novel_id, pos) for pos in gold_positions],
                        "gold_step_multiplicity": {
                            str(pos): position_counts[pos] for pos in gold_positions
                        },
                        "implicit_reasoning_count": implicit_count,
                        "missing_gold_positions": sorted(set(missing_positions)),
                        "prefix_eligible": prefix_eligible,
                    },
                )

                # A ranking query needs at least one retrievable explicit clue.
                if not gold_positions:
                    counts["queries_without_gold"] += 1
                    continue

                jsonl_write(queries_file, {"_id": qid, "text": question})
                counts["queries"] += 1

                for paragraph_id in gold_positions:
                    qrels_file.write(f"{qid}\t{doc_id(novel_id, paragraph_id)}\t1\n")
                    counts["qrels"] += 1

                if prefix_eligible:
                    counts["prefix_eligible_queries"] += 1

    manifest = {
        "source": REPO_ID,
        "language": "en",
        "revision": args.revision,
        "commit_sha": commit_sha,
        "annotation_parts": parts,
        "query": "question only",
        "document_unit": "numbered novel paragraph",
        "relevance": "binary unique explicit clue paragraph",
        "recommended_candidate_scope": "same novel",
        "counts": dict(counts),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    LOG.info(
        "Done: %d novels, %d documents, %d queries, %d qrels",
        counts["novels"],
        counts["documents"],
        counts["queries"],
        counts["qrels"],
    )
    LOG.info("Output: %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
