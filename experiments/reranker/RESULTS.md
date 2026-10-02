# Reranker experiment: results

Question: does reranking memex's retrieval candidates improve ranking quality, and which kind of reranker is worth supporting?

Date: 2026-10-02. Branch: `experiment/reranker` (based on upstream `main` at `1b577fe`, memex 0.25.0).

## Setup

- **Candidates:** the real memex pipeline. Each dataset is converted to one-record Claude-style transcripts, indexed with `memex index --embeddings --model bge` (local `Xenova/bge-small-en-v1.5`), and queried with `memex search --mode lexical|hybrid --limit 30`. Every arm reorders the same 30 candidates per query. Hybrid is memex's BM25 + vector reciprocal-rank fusion with its default recency weighting.
- **Datasets (public, binary relevance):**
  - SciFact (BEIR): 5183 documents, 150 sampled test queries, first 100 evaluated, 1.07 relevant documents per query.
  - CQADupStack programmers (BEIR, Hugging Face `mteb/cqadupstack-programmers` at revision `339629ee116f204b54b1cfd50b9084199a6d415c`): 5000 documents (255 relevant plus seeded distractors), first 100 of 150 sampled queries evaluated, 1.70 relevant documents per query. Stack Exchange posts, a proxy for developer conversations.
  - Sampling is seeded (`random.Random(7)`); `prepare.py` is deterministic.
- **Metrics:** nDCG@10, MRR@10, P@1, Recall@30 (the ceiling of the candidate pool). Query subsets differ per arm (see below); each arm is only compared with the baseline on its own subset.
- **Arms:**
  - `none`: memex order.
  - Local cross-encoders via fastembed `TextRerank`, CPU only: `bge-reranker-base`, `jina-reranker-v1-turbo-en`.
  - Jev (TypeSafe System One, `jev-latest`, one yes/no question per candidate, prompt `generic-1` from `hev-rerank`), via TypeSafe's API.
  - Hosted rerankers through OpenRouter's `/api/v1/rerank`: `cohere/rerank-v3.5`, `voyageai/rerank-2.5-lite`, `qwen/qwen3-reranker-8b`.
  - LLM judge through OpenRouter chat completions: `openai/gpt-6-luna`, one call per query returning a probability per candidate.
- **Hardware:** AMD Threadripper 2950X (32 threads), no GPU used for reranking.
- **Documents:** truncated to 1500 characters for rerankers (1200 for the LLM judge).

## Results

### Local arms and Jev, 100 queries, all candidates

| Dataset | Source | Arm | nDCG@10 | MRR@10 | P@1 | R@30 | p50 / p95 latency |
|---|---|---|---|---|---|---|---|
| SciFact | lexical | none | 0.682 | 0.643 | 0.540 | 0.853 | |
| SciFact | lexical | bge-reranker-base | 0.689 | 0.654 | 0.580 | 0.853 | 6.34 s / 7.16 s |
| SciFact | lexical | jina-turbo | 0.710 | 0.685 | 0.600 | 0.853 | 1.42 s / 1.74 s |
| SciFact | lexical | Jev | 0.786 | 0.771 | 0.730 | 0.853 | ~0.36 s / ~0.6 s |
| SciFact | hybrid | none | 0.728 | 0.692 | 0.610 | 0.935 | |
| SciFact | hybrid | bge-reranker-base | 0.708 | 0.671 | 0.590 | 0.935 | 6.26 s / 7.36 s |
| SciFact | hybrid | jina-turbo | 0.727 | 0.696 | 0.600 | 0.935 | 1.48 s / 1.83 s |
| SciFact | hybrid | Jev | 0.843 | 0.817 | 0.770 | 0.935 | ~0.36 s / ~0.46 s |
| Programming | lexical | none | 0.391 | 0.433 | 0.380 | 0.555 | |
| Programming | lexical | bge-reranker-base | 0.411 | 0.437 | 0.350 | 0.555 | 5.37 s / 6.41 s |
| Programming | lexical | jina-turbo | 0.434 | 0.471 | 0.390 | 0.555 | 1.27 s / 1.46 s |
| Programming | lexical | Jev | 0.456 | 0.480 | 0.390 | 0.555 | ~0.32 s / ~0.43 s |
| Programming | hybrid | none | 0.473 | 0.485 | 0.400 | 0.743 | |
| Programming | hybrid | bge-reranker-base | 0.466 | 0.475 | 0.380 | 0.743 | 5.22 s / 6.52 s |
| Programming | hybrid | jina-turbo | 0.479 | 0.501 | 0.420 | 0.743 | 1.35 s / 1.58 s |
| Programming | hybrid | Jev | 0.590 | 0.596 | 0.510 | 0.743 | ~0.3 s |

Five of the 100 programming queries were sanitized before searching (see Findings); no query was skipped.

### Hosted arms, hybrid candidates

Each hosted arm ran on the first 50 queries (LLM judge: first 30). Baseline columns are `none` on the same subset.

| Dataset | Arm | n | Baseline nDCG@10 | nDCG@10 | MRR@10 | P@1 | p50 / p95 latency | Cost per query |
|---|---|---|---|---|---|---|---|---|
| SciFact | Cohere rerank-v3.5 | 50 | 0.654 | 0.775 | 0.753 | 0.700 | 0.33 s / 0.53 s | $0.0010 |
| SciFact | Voyage rerank-2.5-lite | 50 | 0.654 | 0.766 | 0.736 | 0.680 | 0.36 s / 0.43 s | $0.0002 |
| SciFact | Qwen3-Reranker-8B | 50 | 0.654 | 0.747 | 0.721 | 0.660 | 0.73 s / 2.67 s | $0.0024 |
| SciFact | LLM judge (gpt-6-luna) | 30 | 0.642 | 0.715 | 0.683 | 0.600 | 5.5 s / 11.0 s | $0.0012 |
| Programming | Cohere rerank-v3.5 | 50 | 0.521 | 0.596 | 0.594 | 0.480 | 0.30 s / 0.44 s | $0.0010 |
| Programming | Voyage rerank-2.5-lite | 50 | 0.521 | 0.656 | 0.658 | 0.560 | 0.32 s / 0.54 s | $0.0001 |
| Programming | Qwen3-Reranker-8B | 50 | 0.521 | 0.682 | 0.672 | 0.560 | 0.57 s / 1.49 s | $0.0017 |
| Programming | LLM judge (gpt-6-luna) | 30 | 0.440 | 0.650 | 0.651 | 0.533 | 6.0 s / 9.0 s | $0.0010 |

On the 30 queries judged by every OpenRouter arm (same subset, hybrid): SciFact baseline 0.642, Cohere 0.730, Voyage 0.734, Qwen 0.684, LLM judge 0.715; programming baseline 0.440, Cohere 0.529, Voyage 0.621, Qwen 0.683, LLM judge 0.650.

### Gain over each arm's own baseline (nDCG@10, hybrid)

| Arm | SciFact | Programming |
|---|---|---|
| Jev (100 q) | +0.114 | +0.117 |
| Cohere v3.5 (50 q) | +0.121 | +0.075 |
| Voyage 2.5-lite (50 q) | +0.112 | +0.135 |
| Qwen3-Reranker-8B (50 q) | +0.093 | +0.161 |
| LLM judge (30 q) | +0.073 | +0.210 |
| jina-turbo local (100 q) | -0.001 | +0.006 |
| bge-reranker-base local (100 q) | -0.021 | -0.007 |

### Calibration (hybrid, Cohere v3.5)

Dropping candidates below a probability threshold: on SciFact, a 0.2 threshold keeps 3.6 of 30 candidates on average and retains 87% of the relevant documents in the pool; on the programming set, a 0.2 threshold keeps 4.4 and retains 71%. Thresholds of 0.5 and above lose most relevant documents on both sets.

### Cost

Jev: $0.104 (SciFact, 200 scored queries) and $0.080 (programming, 200 queries). OpenRouter: $0.21 (SciFact) and $0.17 (programming) for 180 calls each.

## Findings

1. Reranking helps. Every hosted reranker and Jev improved nDCG@10 by about 0.07 to 0.16 on both datasets at 0.3 to 0.7 s per query.
2. Local cross-encoders did not pay off on hybrid candidates. Jina turbo helped a little on lexical candidates (+0.03 and +0.04) and was neutral on hybrid; `bge-reranker-base` was neutral to negative and took 5 to 6 s per query on CPU.
3. No clear winner among the hosted options. Jev's raw numbers are highest on SciFact partly because it ran on more queries with a higher baseline; against each arm's own baseline the gains are comparable (SciFact) or smaller (programming) than Voyage and Qwen. Voyage lite is the cheapest by an order of magnitude.
4. The LLM judge gave the largest gain on the programming set but at 5 to 6 s per query is not suited to interactive search.
5. Reranking cannot recover what the pool lacks: Recall@30 is 0.94 / 0.74 (SciFact / programming, hybrid) and 0.85 / 0.56 (lexical).
6. Reranking helps lexical-only candidates (Jev: +0.104 and +0.065), which matters because embeddings are off by default.
7. memex lexical search fails hard on queries containing Tantivy query-syntax characters: `memex search --mode lexical -- "C++ Renaissance - marketing slogan?"`, `foo: bar` and `say "hello` all exit with `Syntax Error`. 5 of 100 programming questions hit it. The harness retried once with punctuation replaced by spaces and used the same text for every arm. This is a separate bug from reranking.

## Limitations

- Small samples (30 to 100 queries per arm), a single run each; differences of a few points are within noise. Re-running the baseline gave nDCG@10 between 0.725 and 0.731 on SciFact hybrid (retrieval is not exactly deterministic), so only gains larger than about 0.01 are meaningful, and with 50 queries much more.
- Public benchmark text, not agent transcripts; binary relevance only.
- Arms ran on different query subsets (100 / 50 / 30); only compare an arm with the baseline on its own subset or use the shared-subset rows.
- Hosted reranker costs are the `usage.cost` OpenRouter reported; Jev cost is the estimate from reported input tokens at $0.042 per million.
- Local rerankers ran on CPU only. A GPU or CoreML build would change their latency but not their quality.
- The OpenRouter hosted rerankers saw the first 30 candidates truncated to 1500 characters; Cohere v3.5 has a 4096-token context.
- Reranker providers receive the query and candidate text; this experiment used public data only.

## Reproducing

All commands run from the repository root; datasets and indexes live outside the repository.

```
python3 experiments/reranker/prepare.py                       # public datasets -> $D
cargo build --release --bin memex --example rerank_experiment
./target/release/examples/rerank_experiment $D/scifact --max-queries 100 \
    --dump-candidates $D/scifact/candidates.json
TYPESAFE_API_KEY=... uv run --script experiments/reranker/jev_rerank.py \
    --candidates $D/scifact/candidates.json --out $D/scifact/jev-scores.json \
    --max-queries 100 --budget-usd 0.15
./target/release/examples/rerank_experiment $D/scifact --max-queries 100 \
    --rerankers bge-base,jina-turbo --external-scores jev=$D/scifact/jev-scores.json
OPENROUTER_API_KEY=... ./target/release/examples/rerank_experiment $D/scifact \
    --max-queries 100 --sources hybrid --rerankers none \
    --remote-rerankers cohere/rerank-v3.5,voyageai/rerank-2.5-lite,qwen/qwen3-reranker-8b \
    --remote-rerank-queries 50 --llm-queries 30 --llm-sources hybrid --budget-usd 0.30
```

Keys are read only from `TYPESAFE_API_KEY` and `OPENROUTER_API_KEY`; the harness caps OpenRouter spend with `--budget-usd` and `jev_rerank.py` caps Jev spend with its own `--budget-usd`.
