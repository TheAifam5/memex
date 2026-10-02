# /// script
# requires-python = ">=3.10"
# dependencies = ["hev-rerank>=0.1.0", "typesafe-sdk>=0.6", "pyyaml>=6"]
# ///
"""Score reranker candidates with TypeSafe Jev (System One) for the reranker experiment.

Mirrors ``hev_rerank.rerank``: the query and up to 30 candidates go into one state as
``{"query", "documents": {"D00": text, ...}}`` with one Noul question per candidate, using the
package's ``generic-1`` prompt. The calls are made here rather than through ``hev_rerank.rerank``
because that function discards token usage, which the budget guard needs.

Input (``--candidates``): one object or a list of objects
``{"dataset", "source", "queries": [{"id", "text", "candidates": [{"docid", "text"}]}]}``.
Output (``--out``) mirrors the input shape, one object per input object:
``{"arm": "jev", "dataset", "source", "model", "prompt_version", "queries": {qid: {docid: p}},
"usage": {...}, "truncated": bool, "dropped_queries": n, "budget_stop": bool, "error": {...} | null}``.
A query appears in ``queries`` only when every candidate was scored.

The API key is read by the SDK from ``TYPESAFE_API_KEY``; this script never reads or prints it.
Exit status: 0 on completion or a budget stop, 1 on an API failure, 2 on invalid input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hev_rerank import load_prompt

USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
CHARS_PER_TOKEN = 4
# Same limits as hev_rerank.rerank: ~32k-token request budget, with headroom.
MAX_DOCS_PER_CALL = 30
MAX_STATE_CHARS = 100_000
MAX_CONCURRENCY = 24
HTTP_TIMEOUT_SECS = 60.0


class InputError(ValueError):
    """The candidates file does not match the expected shape."""


class MalformedResponseError(Exception):
    """A response lacks a valid Noul answer for a question that was asked."""


class Client(Protocol):
    def system_one(self, state: Any, questions: Mapping[str, Any], *, model: str | None = None) -> Any: ...


@dataclass(frozen=True, slots=True)
class Query:
    id: str
    text: str
    candidates: tuple[tuple[str, str], ...]  # (docid, text)


@dataclass(frozen=True, slots=True)
class Dataset:
    dataset: str
    source: str
    queries: tuple[Query, ...]


@dataclass(frozen=True, slots=True)
class Batch:
    dataset: int  # index into the input datasets
    qid: str
    keys: dict[str, str]  # question id -> docid
    state: dict[str, Any]
    questions: dict[str, Any]
    est_tokens: int


@dataclass(slots=True)
class Outcome:
    """Mutable per-dataset result; written out on every exit path."""

    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    partial: dict[str, dict[str, float]] = field(default_factory=dict)
    pending: dict[str, int] = field(default_factory=dict)  # qid -> batches not yet scored
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_calls: int = 0
    calls: int = 0
    latencies_ms: list[float] = field(default_factory=list)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise InputError(msg)


def parse_dataset(raw: Any, where: str) -> Dataset:
    _require(isinstance(raw, dict), f"{where}: expected an object")
    for key in ("dataset", "source"):
        _require(isinstance(raw.get(key), str), f"{where}.{key}: expected a string")
    _require(isinstance(raw.get("queries"), list), f"{where}.queries: expected a list")
    queries: list[Query] = []
    seen_q: set[str] = set()
    for qi, q in enumerate(raw["queries"]):
        qw = f"{where}.queries[{qi}]"
        _require(isinstance(q, dict), f"{qw}: expected an object")
        _require(isinstance(q.get("id"), str) and q["id"] != "", f"{qw}.id: expected a nonempty string")
        _require(q["id"] not in seen_q, f"{qw}.id: duplicate query id {q['id']!r}")
        seen_q.add(q["id"])
        _require(isinstance(q.get("text"), str), f"{qw}.text: expected a string")
        _require(isinstance(q.get("candidates"), list), f"{qw}.candidates: expected a list")
        cands: list[tuple[str, str]] = []
        seen_d: set[str] = set()
        for ci, c in enumerate(q["candidates"]):
            cw = f"{qw}.candidates[{ci}]"
            _require(isinstance(c, dict), f"{cw}: expected an object")
            _require(isinstance(c.get("docid"), str) and c["docid"] != "", f"{cw}.docid: expected a nonempty string")
            _require(c["docid"] not in seen_d, f"{cw}.docid: duplicate docid {c['docid']!r}")
            seen_d.add(c["docid"])
            _require(isinstance(c.get("text"), str), f"{cw}.text: expected a string")
            cands.append((c["docid"], c["text"]))
        queries.append(Query(q["id"], q["text"], tuple(cands)))
    return Dataset(raw["dataset"], raw["source"], tuple(queries))


def load_candidates(path: Path, max_queries: int | None) -> tuple[list[Dataset], bool]:
    """Return the datasets and whether the file held a list (the output mirrors it)."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise InputError(f"{path}: {err}") from err
    is_list = isinstance(raw, list)
    items = raw if is_list else [raw]
    _require(len(items) > 0, f"{path}: empty list")
    datasets = [parse_dataset(item, f"[{i}]" if is_list else "$") for i, item in enumerate(items)]
    if max_queries is not None:
        datasets = [Dataset(d.dataset, d.source, d.queries[:max_queries]) for d in datasets]
    return datasets, is_list


def _chunk(texts: Sequence[str]) -> list[list[int]]:
    batches: list[list[int]] = []
    batch: list[int] = []
    chars = 0
    for i, text in enumerate(texts):
        if batch and (len(batch) >= MAX_DOCS_PER_CALL or chars + len(text) > MAX_STATE_CHARS):
            batches.append(batch)
            batch, chars = [], 0
        batch.append(i)
        chars += len(text)
    if batch:
        batches.append(batch)
    return batches


def estimate_tokens(state: Any, questions: Any) -> int:
    """Approximate input tokens as serialized request characters / 4."""
    chars = len(json.dumps(state, ensure_ascii=False)) + len(json.dumps(questions, ensure_ascii=False))
    return math.ceil(chars / CHARS_PER_TOKEN)


def build_batches(datasets: Sequence[Dataset], prompt: Mapping[str, Any]) -> list[Batch]:
    batches: list[Batch] = []
    for di, ds in enumerate(datasets):
        for q in ds.queries:
            texts = [text for _, text in q.candidates]
            for idx in _chunk(texts):
                keys = {f"D{j:02d}": q.candidates[i][0] for j, i in enumerate(idx)}
                state = {
                    "query": q.text,
                    "documents": {k: q.candidates[i][1] for k, i in zip(keys, idx)},
                }
                questions = {
                    k: {"type": "noul", "instructions": prompt["question"].format(id=k), "criteria": prompt["criteria"]}
                    for k in keys
                }
                batches.append(Batch(di, q.id, keys, state, questions, estimate_tokens(state, questions)))
    return batches


def _api_error_info(err: BaseException) -> dict[str, Any]:
    # Only the class, status, and request id: str(err) and err.args carry the server body.
    return {
        "type": type(err).__name__,
        "status": getattr(err, "status", None),
        "request_id": getattr(err, "request_id", None),
    }


@dataclass(frozen=True, slots=True)
class CallResult:
    scores: dict[str, float]
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    model: str | None


def score_batch(client: Client, model: str, batch: Batch) -> CallResult:
    started = time.monotonic()
    r = client.system_one(state=batch.state, questions=batch.questions, model=model)
    latency_ms = (time.monotonic() - started) * 1000
    scores: dict[str, float] = {}
    for key, docid in batch.keys.items():
        answer = r.answers.get(key)
        p = getattr(answer, "noul", None)
        if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise MalformedResponseError(f"no valid noul answer for {key}")
        scores[docid] = float(p)
    usage = getattr(r, "usage", None)
    return CallResult(
        scores, getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None), latency_ms, getattr(r, "model", None)
    )


@dataclass(slots=True)
class Run:
    outcomes: list[Outcome]
    spent_usd: float = 0.0
    error: dict[str, Any] | None = None
    budget_stop: bool = False
    models_served: set[str] = field(default_factory=set)


def run(
    client: Client,
    model: str,
    batches: Sequence[Batch],
    budget_usd: float,
    concurrency: int,
    fatal: tuple[type[BaseException], ...],
    result: Run,
) -> None:
    """Score batches into `result` with at most `concurrency` calls in flight.

    A batch is admitted only if spend so far plus the estimates of in-flight batches plus its own
    estimate stays within `budget_usd`. Exceptions of a `fatal` type stop admission; in-flight calls
    finish and are recorded. Any other exception propagates after queued calls are cancelled and
    in-flight calls are awaited; `result` stays consistent for a partial write.
    """
    queue = list(reversed(batches))
    inflight: dict[Future[CallResult], Batch] = {}
    stopped = False

    def record(fut: Future[CallResult], b: Batch) -> None:
        nonlocal stopped
        try:
            res = fut.result()
        except fatal as err:
            if result.error is None:
                result.error = _api_error_info(err)
            stopped = True
            return
        out = result.outcomes[b.dataset]
        out.calls += 1
        if res.model is not None:
            result.models_served.add(res.model)
        out.latencies_ms.append(res.latency_ms)
        if res.input_tokens is None:
            out.estimated_calls += 1
            tokens = b.est_tokens
        else:
            tokens = res.input_tokens
        out.input_tokens += tokens
        out.output_tokens += res.output_tokens or 0
        result.spent_usd += tokens * USD_PER_INPUT_TOKEN
        out.partial.setdefault(b.qid, {}).update(res.scores)
        out.pending[b.qid] -= 1
        if out.pending[b.qid] == 0:
            out.scores[b.qid] = out.partial.pop(b.qid)

    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        while queue or inflight:
            while queue and not stopped and len(inflight) < concurrency:
                nxt = queue[-1]
                reserved = sum(b.est_tokens for b in inflight.values()) + nxt.est_tokens
                if result.spent_usd + reserved * USD_PER_INPUT_TOKEN > budget_usd:
                    stopped = result.budget_stop = True
                    break
                queue.pop()
                inflight[pool.submit(score_batch, client, model, nxt)] = nxt
            if stopped:
                queue.clear()
            if not inflight:
                break
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                record(fut, inflight.pop(fut))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _percentile(values: Sequence[float], q: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    return round(statistics.quantiles(values, n=100, method="inclusive")[q - 1], 1)


def render(ds: Dataset, out: Outcome, model: str, prompt_version: str, result: Run) -> dict[str, Any]:
    return {
        "arm": "jev",
        "dataset": ds.dataset,
        "source": ds.source,
        "model": model,
        "models_served": sorted(result.models_served),
        "prompt_version": prompt_version,
        "queries": out.scores,
        "usage": {
            "input_tokens": out.input_tokens,
            "output_tokens": out.output_tokens,
            "est_cost_usd": round(out.input_tokens * USD_PER_INPUT_TOKEN, 6),
            "calls": out.calls,
            "estimated_calls": out.estimated_calls,
            "latency_ms_p50": _percentile(out.latencies_ms, 50),
            "latency_ms_p95": _percentile(out.latencies_ms, 95),
        },
        "truncated": len(out.scores) < len(ds.queries),
        "dropped_queries": len(ds.queries) - len(out.scores),
        "budget_stop": result.budget_stop,
        "error": result.error,
    }


def write_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def execute(
    client: Client,
    model: str,
    datasets: Sequence[Dataset],
    is_list: bool,
    out_path: Path,
    budget_usd: float,
    concurrency: int,
    fatal: tuple[type[BaseException], ...],
    prompt: Mapping[str, Any],
) -> Run:
    """Run scoring and write the output, including on budget stop, API failure, or interrupt."""
    batches = build_batches(datasets, prompt)
    result = Run([Outcome() for _ in datasets])
    for b in batches:
        pend = result.outcomes[b.dataset].pending
        pend[b.qid] = pend.get(b.qid, 0) + 1
    try:
        run(client, model, batches, budget_usd, concurrency, fatal, result)
    except BaseException as err:
        if result.error is None:
            result.error = _api_error_info(err)
        raise
    finally:
        docs = [render(d, o, model, str(prompt.get("version")), result) for d, o in zip(datasets, result.outcomes)]
        write_atomic(out_path, docs if is_list else docs[0])
    return result


def dry_run(datasets: Sequence[Dataset], prompt: Mapping[str, Any], budget_usd: float) -> None:
    batches = build_batches(datasets, prompt)
    total_tokens = 0
    for di, ds in enumerate(datasets):
        own = [b for b in batches if b.dataset == di]
        tokens = sum(b.est_tokens for b in own)
        total_tokens += tokens
        print(
            f"{ds.dataset} ({ds.source}): queries={len(ds.queries)} "
            f"candidates={sum(len(q.candidates) for q in ds.queries)} calls={len(own)} "
            f"est_input_tokens={tokens} est_cost_usd={tokens * USD_PER_INPUT_TOKEN:.6f}"
        )
    cost = total_tokens * USD_PER_INPUT_TOKEN
    verdict = "within" if cost <= budget_usd else "EXCEEDS"
    print(f"total: calls={len(batches)} est_input_tokens={total_tokens} est_cost_usd={cost:.6f} ({verdict} budget {budget_usd})")


def self_test() -> None:
    """Exercise chunking, scoring, usage fallback, budget stop, and error stop with a fake client."""
    from types import SimpleNamespace

    import httpx2
    from typesafe_sdk import TypeSafeBadRequestError, TypeSafeError

    prompt = load_prompt()
    fatal = (TypeSafeError, MalformedResponseError)

    class Fake:
        def __init__(self, fail_on: str | None = None, usage_none: bool = False) -> None:
            self.calls: list[dict[str, Any]] = []
            self.fail_on = fail_on
            self.usage_none = usage_none

        def system_one(self, state: Any, questions: Mapping[str, Any], *, model: str | None = None) -> Any:
            self.calls.append(state)
            if self.fail_on is not None and state["query"] == self.fail_on:
                raise TypeSafeBadRequestError(400, {"error": "secret body"}, httpx2.Headers({"x-typesafe-request-id": "r1"}))
            answers = {k: SimpleNamespace(noul=(len(state["documents"][k]) % 100) / 100) for k in questions}
            usage = SimpleNamespace(input_tokens=None if self.usage_none else 1000, output_tokens=len(questions))
            return SimpleNamespace(answers=answers, usage=usage, model="jev-fake")

    def ds(name: str, queries: list[tuple[str, list[str]]]) -> dict[str, Any]:
        return {
            "dataset": name,
            "source": "test",
            "queries": [
                {"id": qid, "text": qid, "candidates": [{"docid": f"{qid}-d{i}", "text": t} for i, t in enumerate(texts)]}
                for qid, texts in queries
            ],
        }

    tmp = Path(tempfile.mkdtemp(prefix="jev-selftest-"))
    try:
        many = ["x" * (i + 1) for i in range(65)]  # 65 docs -> 3 calls of <=30
        big = ["y" * 40_000 for _ in range(4)]  # 160k chars -> 2 calls
        cand = tmp / "c.json"
        cand.write_text(json.dumps([ds("a", [("q1", many), ("q2", big)]), ds("b", [("q3", ["z"])])]), encoding="utf-8")

        # Full run: list in, list out; every candidate scored; multi-batch merge.
        datasets, is_list = load_candidates(cand, None)
        out = tmp / "o.json"
        fake = Fake()
        execute(fake, "jev-latest", datasets, is_list, out, 10.0, 3, fatal, prompt)
        got = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(got, list) and len(got) == 2, got
        a, b = got
        assert len(fake.calls) == 3 + 2 + 1, len(fake.calls)
        assert set(a["queries"]) == {"q1", "q2"} and len(a["queries"]["q1"]) == 65 and len(a["queries"]["q2"]) == 4
        assert a["queries"]["q1"]["q1-d4"] == 0.05 and b["queries"]["q3"]["q3-d0"] == 0.01
        assert a["usage"]["calls"] == 5 and a["usage"]["input_tokens"] == 5000 and a["usage"]["estimated_calls"] == 0
        assert a["usage"]["latency_ms_p95"] is not None and b["usage"]["latency_ms_p50"] is not None
        assert a["truncated"] is False and a["error"] is None and a["prompt_version"] == "generic-1"
        assert a["models_served"] == ["jev-fake"] and a["budget_stop"] is False
        assert all(set(c["documents"]) <= {f"D{j:02d}" for j in range(30)} for c in fake.calls)
        assert not any(p.name.endswith(".tmp") for p in tmp.iterdir())

        # Usage missing: charge the chars/4 estimate and count it.
        fake = Fake(usage_none=True)
        one = tmp / "one.json"
        one.write_text(json.dumps(ds("c", [("q1", ["abc", "de"])])), encoding="utf-8")
        datasets, is_list = load_candidates(one, None)
        execute(fake, "jev-latest", datasets, is_list, out, 10.0, 2, fatal, prompt)
        got = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(got, dict) and got["usage"]["estimated_calls"] == 1
        assert got["usage"]["input_tokens"] == build_batches(datasets, prompt)[0].est_tokens

        # Budget stop: the 160k-char query cannot be admitted after q1; it is dropped, not half-written.
        datasets, is_list = load_candidates(cand, None)
        # Serial admission of the k-th q1 call needs 1000 * k spent tokens plus its own estimate.
        q1_est = [b.est_tokens for b in build_batches(datasets, prompt) if b.qid == "q1"]
        budget = (max(1000 * k + est for k, est in enumerate(q1_est)) + 1) * USD_PER_INPUT_TOKEN
        fake = Fake()
        execute(fake, "jev-latest", datasets, is_list, out, budget, 1, fatal, prompt)
        a, b = json.loads(out.read_text(encoding="utf-8"))
        assert set(a["queries"]) == {"q1"} and a["truncated"] is True and a["dropped_queries"] == 1
        assert a["budget_stop"] is True and b["queries"] == {} and b["dropped_queries"] == 1 and a["error"] is None

        # Non-retryable API error: stop, keep finished queries, report no body text.
        fake = Fake(fail_on="q2")
        run_result = execute(fake, "jev-latest", datasets, is_list, out, 10.0, 1, fatal, prompt)
        text = out.read_text(encoding="utf-8")
        a, b = json.loads(text)
        assert run_result.error == {"type": "TypeSafeBadRequestError", "status": 400, "request_id": "r1"}
        assert "secret body" not in text and set(a["queries"]) == {"q1"} and a["truncated"] is True

        # Malformed answer is fatal too.
        class Bad(Fake):
            def system_one(self, state: Any, questions: Mapping[str, Any], *, model: str | None = None) -> Any:
                return SimpleNamespace(answers={}, usage=SimpleNamespace(input_tokens=1, output_tokens=0))

        run_result = execute(Bad(), "jev-latest", datasets[:1], False, out, 10.0, 2, fatal, prompt)
        assert run_result.error is not None and run_result.error["type"] == "MalformedResponseError"

        # Input validation.
        for bad in ({"dataset": "x", "source": "y"}, [], {"dataset": "x", "source": "y", "queries": [{"id": "q", "text": "t", "candidates": [{"docid": "d", "text": "a"}, {"docid": "d", "text": "b"}]}]}):
            cand.write_text(json.dumps(bad), encoding="utf-8")
            try:
                load_candidates(cand, None)
            except InputError:
                pass
            else:
                raise AssertionError(f"accepted invalid input {bad!r}")
    finally:
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
    print("self-test: PASS")


def make_live_client() -> Client:
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    # 408 and other 4xx are not retried; 429 honors Retry-After.
    retry = RetryPolicy(max_retries=6, backoff_max=30.0, timeout=90.0, http_statuses={429, *range(500, 600)})
    return TypeSafeClient(retry=retry, timeout=HTTP_TIMEOUT_SECS)


def main(argv: Sequence[str] | None = None, client_factory: Callable[[], Client] = make_live_client) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", type=Path, help="candidates JSON written by the harness")
    ap.add_argument("--out", type=Path, help="scores JSON to write (atomically)")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--max-queries", type=int, default=None, help="score only the first N queries of each dataset")
    ap.add_argument("--budget-usd", type=float, default=0.25, help="stop before estimated spend exceeds this")
    ap.add_argument("--concurrency", type=int, default=4, help=f"calls in flight, 1..{MAX_CONCURRENCY}")
    ap.add_argument("--dry-run", action="store_true", help="validate input and print a cost estimate; no network")
    ap.add_argument("--self-test", action="store_true", help="run offline checks with a fake client")
    args = ap.parse_args(argv)

    if args.self_test:
        self_test()
        return 0
    if args.candidates is None:
        ap.error("--candidates is required")
    if not args.dry_run and args.out is None:
        ap.error("--out is required unless --dry-run")
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        ap.error(f"--concurrency must be in 1..{MAX_CONCURRENCY}")
    if not (math.isfinite(args.budget_usd) and args.budget_usd > 0):
        ap.error("--budget-usd must be a positive number")
    if args.max_queries is not None and args.max_queries < 1:
        ap.error("--max-queries must be at least 1")

    try:
        datasets, is_list = load_candidates(args.candidates, args.max_queries)
    except InputError as err:
        print(f"invalid candidates: {err}", file=sys.stderr)
        return 2
    prompt = load_prompt()
    if args.dry_run:
        dry_run(datasets, prompt, args.budget_usd)
        return 0

    from typesafe_sdk import TypeSafeError

    try:
        client = client_factory()
    except TypeSafeError as err:
        # Raised for a missing or malformed key; the message contains no key material.
        print(f"cannot create TypeSafe client: {err}", file=sys.stderr)
        return 1
    try:
        result = execute(
            client, args.model, datasets, is_list, args.out, args.budget_usd, args.concurrency,
            (TypeSafeError, MalformedResponseError), prompt,
        )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
    scored = sum(len(o.scores) for o in result.outcomes)
    total = sum(len(d.queries) for d in datasets)
    print(f"jev: scored {scored}/{total} queries, est_cost_usd={result.spent_usd:.6f}, wrote {args.out}", file=sys.stderr)
    if result.error is not None:
        print(f"jev: stopped on {result.error['type']} status={result.error['status']} request_id={result.error['request_id']}", file=sys.stderr)
        return 1
    if result.budget_stop:
        print("jev: budget reached; output truncated", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
