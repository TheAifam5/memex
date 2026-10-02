#!/usr/bin/env python3
"""Prepare public retrieval benchmarks for the reranker experiment.

Writes, per dataset, under ``<out>/<name>/``:

- ``claude/corpus/<docid>.jsonl``: one Claude-transcript-shaped line per document.
- ``queries.json``: ``[{"id", "text"}]``, a seeded sample of queries with relevant docs.
- ``qrels.json``: ``{qid: {docid: grade}}`` for the sampled queries, grade >= 1 only.
- ``docid_map.json``: sanitized docid -> original docid, only when sanitizing changed one.

Downloads are cached in ``<out>/downloads/`` and skipped when already present.
Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OUT = Path(
    "/tmp/claude-1000/-home-theaifam5-Projects-memex/"
    "155e6480-e6f8-4ad2-920f-0fe4295da95d/scratchpad/rerank-exp"
)
SEED = 7
MAX_QUERIES = 150
MAX_DOCS = 5000
MAX_CONTENT_CHARS = 4000
TIMESTAMP = "2026-01-01T00:00:00Z"
DOWNLOAD_TIMEOUT_SECS = 120

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
# Pinned revision of https://huggingface.co/datasets/mteb/cqadupstack-programmers
CQA_REVISION = "339629ee116f204b54b1cfd50b9084199a6d415c"
CQA_BASE = f"https://huggingface.co/datasets/mteb/cqadupstack-programmers/resolve/{CQA_REVISION}"

_UNSAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


@dataclass(frozen=True, slots=True)
class Doc:
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class Raw:
    """A BEIR-format dataset loaded into memory, keyed by original ids."""

    corpus: dict[str, Doc]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"cached   {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"download {url}")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECS) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)
    return dest


def parse_jsonl(lines: Iterator[str]) -> Iterator[dict]:
    for line in lines:
        if line.strip():
            yield json.loads(line)


def parse_qrels(lines: Iterator[str]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    reader = csv.reader(lines, delimiter="\t")
    header = next(reader)
    if header[:3] != ["query-id", "corpus-id", "score"]:
        raise ValueError(f"unexpected qrels header: {header!r}")
    for row in reader:
        if not row:
            continue
        qid, did, score = row[0], row[1], int(row[2])
        if score >= 1:
            qrels.setdefault(qid, {})[did] = score
    return qrels


def load_beir(corpus: Iterator[str], queries: Iterator[str], qrels: Iterator[str]) -> Raw:
    docs = {
        str(r["_id"]): Doc(r.get("title") or "", r.get("text") or "") for r in parse_jsonl(corpus)
    }
    qs = {str(r["_id"]): r["text"] for r in parse_jsonl(queries)}
    return Raw(docs, qs, parse_qrels(qrels))


def load_scifact(downloads: Path) -> Raw:
    path = download(SCIFACT_URL, downloads / "scifact.zip")
    with zipfile.ZipFile(path) as zf:

        def lines(name: str) -> Iterator[str]:
            return io.TextIOWrapper(zf.open(f"scifact/{name}"), encoding="utf-8")

        return load_beir(lines("corpus.jsonl"), lines("queries.jsonl"), lines("qrels/test.tsv"))


def load_cqa_programmers(downloads: Path) -> Raw:
    d = downloads / f"cqadupstack-programmers-{CQA_REVISION[:12]}"
    paths = {
        name: download(f"{CQA_BASE}/{name}", d / name)
        for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
    }
    with (
        paths["corpus.jsonl"].open(encoding="utf-8") as c,
        paths["queries.jsonl"].open(encoding="utf-8") as q,
        paths["qrels/test.tsv"].open(encoding="utf-8", newline="") as r,
    ):
        return load_beir(c, q, r)


def sanitize_ids(ids: list[str]) -> dict[str, str]:
    """Map original docids to unique ``[A-Za-z0-9_-]`` ids."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for orig in ids:
        base = _UNSAFE_ID.sub("_", orig) or "doc"
        safe, n = base, 1
        while safe in used:
            safe, n = f"{base}_{n}", n + 1
        used.add(safe)
        mapping[orig] = safe
    return mapping


def build(name: str, raw: Raw, out: Path, max_docs: int | None) -> None:
    rng = random.Random(SEED)
    eligible = sorted(
        qid for qid, rel in raw.qrels.items() if qid in raw.queries and any(d in raw.corpus for d in rel)
    )
    sampled = sorted(rng.sample(eligible, min(MAX_QUERIES, len(eligible))))

    relevant = sorted({d for q in sampled for d in raw.qrels[q] if d in raw.corpus})
    if max_docs is None:
        doc_ids = sorted(raw.corpus)
    else:
        rel_set = set(relevant)
        pool = sorted(d for d in raw.corpus if d not in rel_set)
        n_distractors = max(0, max_docs - len(relevant))
        doc_ids = sorted(relevant + rng.sample(pool, min(n_distractors, len(pool))))

    mapping = sanitize_ids(doc_ids)
    ds = out / name
    corpus_dir = ds / "claude" / "corpus"
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True)

    total_chars = 0
    for orig in doc_ids:
        doc = raw.corpus[orig]
        content = f"{doc.title}\n\n{doc.text}" if doc.title else doc.text
        content = content[:MAX_CONTENT_CHARS]
        total_chars += len(content)
        safe = mapping[orig]
        record = {
            "type": "assistant",
            "uuid": safe,
            "sessionId": safe,
            "timestamp": TIMESTAMP,
            "message": {"role": "assistant", "content": content},
        }
        (corpus_dir / f"{safe}.jsonl").write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    queries = [{"id": q, "text": raw.queries[q]} for q in sampled]
    qrels = {
        q: {mapping[d]: g for d, g in sorted(raw.qrels[q].items()) if d in raw.corpus} for q in sampled
    }
    (ds / "queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=1), encoding="utf-8")
    (ds / "qrels.json").write_text(json.dumps(qrels, indent=1), encoding="utf-8")
    changed = {safe: orig for orig, safe in mapping.items() if safe != orig}
    map_path = ds / "docid_map.json"
    if changed:
        map_path.write_text(json.dumps(changed, indent=1), encoding="utf-8")
    elif map_path.exists():
        map_path.unlink()

    n_rel = sum(len(v) for v in qrels.values())
    print(
        f"{name}: docs={len(doc_ids)} (relevant={len(relevant)}) queries={len(queries)} "
        f"eligible_queries={len(eligible)} avg_relevant_per_query={n_rel / len(queries):.2f} "
        f"avg_doc_chars={total_chars / len(doc_ids):.0f} renamed_docids={len(changed)} -> {ds}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out: Path = args.out
    downloads = out / "downloads"

    build("scifact", load_scifact(downloads), out, max_docs=None)
    build("cqadupstack-programmers", load_cqa_programmers(downloads), out, max_docs=MAX_DOCS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
