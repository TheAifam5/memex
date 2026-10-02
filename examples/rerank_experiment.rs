//! Measures whether reranking memex search candidates improves retrieval.
//!
//! For one dataset directory (`claude/corpus/<docid>.jsonl`, `queries.json`,
//! `qrels.json`) the harness indexes the corpus with the memex binary, collects
//! lexical and hybrid candidate lists, reorders them with local cross-encoders
//! and optionally an OpenRouter LLM judge, and reports nDCG@10, MRR@10, P@1,
//! and Recall@N per candidate source and arm.
//!
//! ```sh
//! cargo run --release --example rerank_experiment -- <dataset-dir> [--max-queries N]
//! ```
//!
//! The LLM arm runs only when `OPENROUTER_API_KEY` is set and stops once key
//! usage since the start exceeds `--budget-usd`.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, ValueEnum};
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use ureq::Agent;
use ureq::tls::{RootCerts, TlsConfig, TlsProvider};

const OPENROUTER_CHAT_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
const OPENROUTER_RERANK_URL: &str = "https://openrouter.ai/api/v1/rerank";
/// Document length for the single retry after a hosted reranker rejects the input as too long.
const SHORT_DOC_CHARS: usize = 800;
const OPENROUTER_KEY_URL: &str = "https://openrouter.ai/api/v1/key";
const API_KEY_ENV: &str = "OPENROUTER_API_KEY";
/// Written to the memex root only after a successful index, so a failed run is retried.
const INDEX_MARKER: &str = "rerank-experiment-indexed";
const EMBED_MODEL: &str = "bge";
const NDCG_K: usize = 10;
/// LLM calls between two key-usage checks of the budget guard.
const BUDGET_CHECK_EVERY: usize = 10;
const LLM_ATTEMPTS: u32 = 3;
const LLM_TIMEOUT: Duration = Duration::from_secs(180);
const LLM_BACKOFF: Duration = Duration::from_secs(2);
const MAX_CONSECUTIVE_LLM_FAILURES: usize = 3;
const MAX_RESPONSE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_ERROR_BODY_BYTES: u64 = 4 * 1024;
const MAX_ERROR_EXCERPT_CHARS: usize = 300;
const MAX_STDERR_EXCERPT_CHARS: usize = 2000;
const THRESHOLDS: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

const JUDGE_SYSTEM_PROMPT: &str = "You are a careful search relevance judge. For each candidate \
document, estimate the calibrated probability (between 0 and 1) that it is relevant to the query. \
Relevant means the document contains information that answers or directly addresses the query; a \
document that is only loosely related or on the same topic is not relevant. Respond with strict \
JSON only: one object mapping every document id to its probability, for example \
{\"D00\": 0.91, \"D01\": 0.03}.";

#[derive(Parser, Debug)]
#[command(about = "Measure reranking of memex search candidates")]
struct Args {
    /// Dataset directory containing claude/corpus, queries.json, and qrels.json.
    dataset: PathBuf,
    /// memex binary; defaults to target/release/memex, then target/debug/memex.
    #[arg(long)]
    memex_bin: Option<PathBuf>,
    /// memex data directory; defaults to <dataset>/memex-root.
    #[arg(long)]
    root: Option<PathBuf>,
    /// fastembed model cache shared by memex and the rerankers; defaults to <dataset>/../models.
    #[arg(long)]
    model_cache: Option<PathBuf>,
    /// Candidate sources to evaluate.
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = [CandidateSource::Lexical, CandidateSource::Hybrid])]
    sources: Vec<CandidateSource>,
    /// Local rerankers: bge-base, bge-v2-m3, jina-turbo, jina-v2, a fastembed model code, or `none`.
    #[arg(long, value_delimiter = ',', default_value = "bge-base,jina-turbo")]
    rerankers: Vec<String>,
    /// Candidates requested from memex per query.
    #[arg(long, default_value_t = 30)]
    candidates: usize,
    /// Evaluate only the first N queries that have relevance judgments.
    #[arg(long)]
    max_queries: Option<usize>,
    /// Run the LLM arm on the first N evaluated queries only.
    #[arg(long)]
    llm_queries: Option<usize>,
    /// OpenRouter model id for the LLM judge.
    #[arg(long, default_value = "openai/gpt-6-luna")]
    llm_model: String,
    /// Candidate sources the LLM arm judges.
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = [CandidateSource::Lexical, CandidateSource::Hybrid])]
    llm_sources: Vec<CandidateSource>,
    /// Skip the LLM arm even when an API key is set.
    #[arg(long)]
    no_llm: bool,
    /// OpenRouter rerank models (e.g. cohere/rerank-v3.5), evaluated as `or:<model>` arms.
    #[arg(long, value_delimiter = ',')]
    remote_rerankers: Vec<String>,
    /// Run the hosted rerankers on the first N evaluated queries only.
    #[arg(long)]
    remote_rerank_queries: Option<usize>,
    /// Hard ceiling on OpenRouter key usage (USD) across the LLM judge and hosted rerankers.
    #[arg(long, default_value_t = 0.30)]
    budget_usd: f64,
    /// Characters of document text given to local rerankers.
    #[arg(long, default_value_t = 1500)]
    rerank_doc_chars: usize,
    /// Characters of document text given to the LLM judge.
    #[arg(long, default_value_t = 1200)]
    llm_doc_chars: usize,
    /// Passed to `memex search --recency-weight` when set; memex's default applies otherwise.
    #[arg(long)]
    recency_weight: Option<f64>,
    /// Results JSON path; defaults to <dataset>/rerank-results.json.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Write candidate lists with document text to this path after retrieval and exit.
    #[arg(long)]
    dump_candidates: Option<PathBuf>,
    /// Evaluate an externally scored arm, as `<arm>=<path>` (repeatable).
    #[arg(long, value_parser = parse_external_arg)]
    external_scores: Vec<(String, PathBuf)>,
}

fn parse_external_arg(raw: &str) -> Result<(String, PathBuf), String> {
    match raw.split_once('=') {
        Some((arm, path)) if !arm.trim().is_empty() && !path.is_empty() => {
            Ok((arm.trim().to_string(), PathBuf::from(path)))
        }
        _ => Err(format!("expected <arm>=<path>, got {raw:?}")),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ValueEnum)]
enum CandidateSource {
    Lexical,
    Hybrid,
}

impl CandidateSource {
    fn name(self) -> &'static str {
        match self {
            Self::Lexical => "lexical",
            Self::Hybrid => "hybrid",
        }
    }
}

impl fmt::Display for CandidateSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Deserialize)]
struct QueryRow {
    id: String,
    text: String,
}

struct Query {
    id: String,
    text: String,
    /// Relevance grades above zero, keyed by document id.
    rels: HashMap<String, f64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let dataset = fs::canonicalize(&args.dataset)
        .with_context(|| format!("dataset dir {} not found", args.dataset.display()))?;
    let claude_dir = dataset.join("claude");
    let mut queries = load_queries(&dataset, args.max_queries)?;
    if queries.is_empty() {
        bail!(
            "no queries with relevance judgments in {}",
            dataset.display()
        );
    }
    let rerankers = parse_rerankers(&args.rerankers)?;
    let name = dataset_name(&dataset);
    let mut external = Vec::new();
    for (arm, path) in &args.external_scores {
        let reserved = arm == "none"
            || arm.starts_with("or:")
            || arm.starts_with("llm:")
            || rerankers.iter().any(|(label, _)| label == arm)
            || args
                .external_scores
                .iter()
                .filter(|(a, _)| a == arm)
                .count()
                > 1;
        if reserved {
            bail!("external arm name {arm:?} is already used");
        }
        external.push((arm.clone(), path.clone(), load_external_scores(path)?));
    }
    let model_cache = args.model_cache.clone().unwrap_or_else(|| {
        dataset
            .parent()
            .map_or_else(|| dataset.join("models"), |p| p.join("models"))
    });
    fs::create_dir_all(&model_cache)?;
    let model_cache = fs::canonicalize(&model_cache)?;
    let root = args
        .root
        .clone()
        .unwrap_or_else(|| dataset.join("memex-root"));
    fs::create_dir_all(&root)?;
    let root = fs::canonicalize(&root)?;
    let memex = Memex {
        bin: resolve_memex_bin(args.memex_bin.as_deref())?,
        home: root.join("home"),
        root,
        model_cache: model_cache.clone(),
        recency_weight: args.recency_weight,
    };
    eprintln!(
        "dataset {} | {} queries | memex {} | root {}",
        dataset.display(),
        queries.len(),
        memex.bin.display(),
        memex.root.display()
    );

    memex.ensure_index(&claude_dir)?;

    let mut corpus = Corpus::new(claude_dir.join("corpus"));
    let candidates = collect_candidates(&memex, &mut queries, &args.sources, args.candidates)?;
    if queries.is_empty() {
        bail!("no searchable queries left in {}", dataset.display());
    }
    let fixes = QueryFixes {
        sanitized: candidates.sanitized,
        skipped: candidates.skipped,
    };
    let mut results: Vec<SourceRun> = args
        .sources
        .iter()
        .zip(candidates.pools)
        .map(|(&source, pools)| SourceRun {
            source,
            arms: vec![Arm::new("none", pools.clone())],
            pools,
        })
        .collect();

    if let Some(path) = &args.dump_candidates {
        let dump = candidates_dump(
            &name,
            &queries,
            &results,
            &mut corpus,
            args.rerank_doc_chars,
        );
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&dump)?)
            .with_context(|| format!("failed to write {}", path.display()))?;
        eprintln!("candidates written to {}", path.display());
        return Ok(());
    }

    let mut judged = Vec::new();
    for (arm, path, entries) in &external {
        let judged_arm = external_arm(arm, path, entries, &name, &queries, &results)?;
        eprintln!("external arm {arm}: {}", judged_arm.summary);
        judged.push(judged_arm);
    }

    for (label, model) in &rerankers {
        run_local_reranker(
            label,
            model,
            &model_cache,
            &queries,
            &mut results,
            &mut corpus,
            args.rerank_doc_chars,
        )?;
    }

    let mut remote_models: Vec<&str> = Vec::new();
    for model in args.remote_rerankers.iter().map(|m| m.trim()) {
        if !model.is_empty() && !remote_models.contains(&model) {
            remote_models.push(model);
        }
    }
    let mut openrouter = None;
    if args.no_llm && remote_models.is_empty() {
        eprintln!("LLM arm skipped: --no-llm");
    } else {
        match std::env::var(API_KEY_ENV) {
            Ok(key) if !key.trim().is_empty() => {
                let client = OpenRouter::new(key.trim());
                match Budget::start(&client, args.budget_usd) {
                    Ok(mut budget) => {
                        for model in &remote_models {
                            let run = run_remote_reranker(
                                &args,
                                model,
                                &client,
                                &mut budget,
                                &queries,
                                &results,
                                &mut corpus,
                            );
                            judged.push(run.into_judged());
                        }
                        if args.no_llm {
                            eprintln!("LLM arm skipped: --no-llm");
                        } else {
                            let run = run_llm_arm(
                                &args,
                                &client,
                                &mut budget,
                                &queries,
                                &results,
                                &mut corpus,
                            );
                            judged.push(run.into_judged());
                        }
                        openrouter = Some(budget.finish(&client));
                    }
                    Err(err) => {
                        eprintln!(
                            "OpenRouter arms skipped: initial key usage check failed: {err:#}"
                        );
                        openrouter = Some(json!({
                            "budget_usd": args.budget_usd,
                            "stopped": format!("initial key usage check failed: {err:#}"),
                        }));
                    }
                }
            }
            _ => eprintln!(
                "OpenRouter arms (LLM judge, hosted rerankers) skipped: {API_KEY_ENV} is not set"
            ),
        }
    }

    if corpus.missing > 0 {
        eprintln!(
            "warning: {} candidate documents had no corpus file",
            corpus.missing
        );
    }

    let mut report = build_report(&args, &dataset, &memex, &queries, &results, &judged, &fixes);
    if let Some(map) = report.as_object_mut() {
        map.insert("openrouter".into(), openrouter.unwrap_or(Value::Null));
    }
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| dataset.join("rerank-results.json"));
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&out, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("failed to write {}", out.display()))?;
    print_tables(
        &dataset,
        &queries,
        &results,
        &judged,
        &fixes,
        args.candidates,
    );
    eprintln!("results written to {}", out.display());
    Ok(())
}

fn load_queries(dataset: &Path, max_queries: Option<usize>) -> Result<Vec<Query>> {
    let rows: Vec<QueryRow> = read_json(&dataset.join("queries.json"))?;
    let qrels: HashMap<String, HashMap<String, f64>> = read_json(&dataset.join("qrels.json"))?;
    let mut queries: Vec<Query> = rows
        .into_iter()
        .filter_map(|row| {
            let rels: HashMap<String, f64> = qrels
                .get(&row.id)?
                .iter()
                .filter(|(_, grade)| **grade > 0.0)
                .map(|(doc, grade)| (doc.clone(), *grade))
                .collect();
            (!rels.is_empty()).then_some(Query {
                id: row.id,
                text: row.text,
                rels,
            })
        })
        .collect();
    if let Some(max) = max_queries {
        queries.truncate(max);
    }
    Ok(queries)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let raw = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

fn parse_rerankers(names: &[String]) -> Result<Vec<(String, RerankerModel)>> {
    let mut out = Vec::new();
    for name in names.iter().map(|n| n.trim()).filter(|n| !n.is_empty()) {
        if name.eq_ignore_ascii_case("none") {
            continue;
        }
        let model = match name.to_ascii_lowercase().as_str() {
            "bge-base" | "bgererankerbase" => RerankerModel::BGERerankerBase,
            "bge-v2-m3" | "bgererankerv2m3" => RerankerModel::BGERerankerV2M3,
            "jina-turbo" | "jinarerankerv1turboen" => RerankerModel::JINARerankerV1TurboEn,
            "jina-v2" | "jinarerankerv2basemultiligual" => {
                RerankerModel::JINARerankerV2BaseMultiligual
            }
            _ => name.parse::<RerankerModel>().map_err(|e| anyhow!(e))?,
        };
        out.push((reranker_label(&model).to_string(), model));
    }
    Ok(out)
}

fn reranker_label(model: &RerankerModel) -> &'static str {
    match model {
        RerankerModel::BGERerankerBase => "bge-reranker-base",
        RerankerModel::BGERerankerV2M3 => "bge-reranker-v2-m3",
        RerankerModel::JINARerankerV1TurboEn => "jina-reranker-v1-turbo-en",
        RerankerModel::JINARerankerV2BaseMultiligual => "jina-reranker-v2-base-multilingual",
    }
}

fn resolve_memex_bin(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return fs::canonicalize(path)
            .with_context(|| format!("memex binary {} not found", path.display()));
    }
    let target = Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
    ["release", "debug"]
        .iter()
        .map(|profile| target.join(profile).join("memex"))
        .find(|path| path.is_file())
        .ok_or_else(|| {
            anyhow!(
                "no memex binary under {}; run cargo build --release",
                target.display()
            )
        })
}

/// Runs the memex binary with an isolated home and only the environment it needs,
/// so no API key from this process reaches it and no remote embedding is configured.
struct Memex {
    bin: PathBuf,
    root: PathBuf,
    home: PathBuf,
    model_cache: PathBuf,
    recency_weight: Option<f64>,
}

impl Memex {
    fn command(&self) -> Command {
        let mut cmd = Command::new(&self.bin);
        cmd.env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", &self.home)
            .env("MEMEX_MODEL", EMBED_MODEL)
            .env("FASTEMBED_CACHE_DIR", &self.model_cache)
            .env("HF_HOME", &self.model_cache)
            .current_dir(&self.root);
        for key in ["ORT_DYLIB_PATH", "LD_LIBRARY_PATH"] {
            if let Some(value) = std::env::var_os(key) {
                cmd.env(key, value);
            }
        }
        cmd
    }

    /// Indexes unless the marker records the same corpus path and file count;
    /// a changed corpus is re-indexed incrementally.
    fn ensure_index(&self, claude_dir: &Path) -> Result<()> {
        let marker = self.root.join(INDEX_MARKER);
        let corpus_files = fs::read_dir(claude_dir.join("corpus"))
            .with_context(|| format!("failed to list {}/corpus", claude_dir.display()))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "jsonl"))
            .count();
        let stamp = format!(
            "claude_path={}\ncorpus_files={corpus_files}\n",
            claude_dir.display()
        );
        if fs::read_to_string(&marker).is_ok_and(|existing| existing == stamp) {
            eprintln!("index exists ({}), skipping memex index", marker.display());
            return Ok(());
        }
        fs::create_dir_all(&self.home)?;
        eprintln!(
            "indexing {} with {EMBED_MODEL} embeddings",
            claude_dir.display()
        );
        let started = Instant::now();
        let mut cmd = self.command();
        cmd.arg("index")
            .arg("--root")
            .arg(&self.root)
            .arg("--claude-path")
            .arg(claude_dir)
            .args([
                "--only-source",
                "claude",
                "--embeddings",
                "--model",
                EMBED_MODEL,
            ])
            .args(["--no-update-check", "--non-interactive"]);
        let output = run_checked(cmd, "memex index")?;
        let summary = String::from_utf8_lossy(&output.stdout);
        eprintln!(
            "memex index finished in {:.1}s: {}",
            started.elapsed().as_secs_f64(),
            summary.trim()
        );
        fs::write(&marker, stamp)?;
        Ok(())
    }

    /// Returns distinct session ids in memex rank order, at most `limit`.
    fn search(
        &self,
        query: &str,
        source: CandidateSource,
        limit: usize,
    ) -> Result<Vec<String>, SearchFailure> {
        let mut cmd = self.command();
        cmd.arg("search")
            .args(["--mode", source.name()])
            .args(["--limit", &limit.to_string()])
            .args(["--unique-session", "--source", "claude"])
            .args(["--fields", "session_id,score", "--format", "jsonl"])
            .arg("--root")
            .arg(&self.root)
            .args(["--no-update-check", "--non-interactive"]);
        if let Some(weight) = self.recency_weight {
            cmd.args(["--recency-weight", &weight.to_string()]);
        }
        // `--` keeps a query that starts with `-` from being parsed as a flag.
        cmd.arg("--").arg(query);
        let output = cmd.output().map_err(|e| {
            SearchFailure::Other(anyhow!(e).context("failed to start memex search"))
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let err = anyhow!(
                "memex search exited with {}: {}",
                output.status,
                tail_chars(stderr.trim(), MAX_STDERR_EXCERPT_CHARS)
            );
            return Err(if is_query_syntax_error(&stderr) {
                SearchFailure::Syntax(err)
            } else {
                SearchFailure::Other(err)
            });
        }
        Ok(parse_search_output(
            &String::from_utf8_lossy(&output.stdout),
            limit,
        ))
    }
}

fn run_checked(mut cmd: Command, what: &str) -> Result<Output> {
    let output = cmd
        .output()
        .with_context(|| format!("failed to start {what}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "{what} exited with {}: {}",
            output.status,
            tail_chars(stderr.trim(), MAX_STDERR_EXCERPT_CHARS)
        );
    }
    Ok(output)
}

fn parse_search_output(stdout: &str, limit: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
        .filter_map(|hit| hit.get("session_id")?.as_str().map(str::to_string))
        .filter(|id| seen.insert(id.clone()))
        .take(limit)
        .collect()
}

enum SearchFailure {
    /// memex rejected the query text itself (Tantivy query syntax).
    Syntax(anyhow::Error),
    Other(anyhow::Error),
}

impl SearchFailure {
    fn into_error(self) -> anyhow::Error {
        match self {
            Self::Syntax(err) | Self::Other(err) => err,
        }
    }
}

fn is_query_syntax_error(stderr: &str) -> bool {
    stderr.contains("Syntax Error")
}

/// Whether a failed search is retried with the sanitized query: only a syntax error on
/// the original text, and only when sanitizing leaves something to search.
fn retry_sanitized(failure: &SearchFailure, already_sanitized: bool, sanitized: &str) -> bool {
    matches!(failure, SearchFailure::Syntax(_)) && !already_sanitized && !sanitized.is_empty()
}

/// Replaces every character that is not alphanumeric or whitespace with a space and
/// collapses whitespace runs.
fn sanitize_query(text: &str) -> String {
    let replaced: String = text
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    replaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn search_sources(
    memex: &Memex,
    text: &str,
    sources: &[CandidateSource],
    limit: usize,
) -> Result<Vec<Vec<String>>, SearchFailure> {
    sources
        .iter()
        .map(|&source| memex.search(text, source, limit))
        .collect()
}

/// Query ids changed by query-syntax sanitizing; the same set applies to every source.
struct QueryFixes {
    sanitized: Vec<String>,
    skipped: Vec<String>,
}

/// Candidate pools per source, plus the queries that needed sanitizing.
struct Candidates {
    /// One list of pools per source, aligned with the kept queries.
    pools: Vec<Vec<Vec<String>>>,
    sanitized: Vec<String>,
    /// Queries dropped because sanitizing left no text.
    skipped: Vec<String>,
}

/// Searches every source per query. A query memex rejects as query syntax is searched
/// again, for every source, with [`sanitize_query`] text, which then replaces the query
/// text for all arms; a query that sanitizes to nothing is dropped from `queries`.
/// Any other failure is fatal.
fn collect_candidates(
    memex: &Memex,
    queries: &mut Vec<Query>,
    sources: &[CandidateSource],
    limit: usize,
) -> Result<Candidates> {
    let started = Instant::now();
    let mut out = Candidates {
        pools: vec![Vec::with_capacity(queries.len()); sources.len()],
        sanitized: Vec::new(),
        skipped: Vec::new(),
    };
    let total = queries.len();
    let mut kept = Vec::with_capacity(total);
    for (n, mut query) in std::mem::take(queries).into_iter().enumerate() {
        let per_source = match search_sources(memex, &query.text, sources, limit) {
            Ok(pools) => pools,
            Err(failure) => {
                let clean = sanitize_query(&query.text);
                if matches!(failure, SearchFailure::Syntax(_)) && clean.is_empty() {
                    eprintln!(
                        "query {} has only query-syntax characters; skipped",
                        query.id
                    );
                    out.skipped.push(query.id);
                    continue;
                }
                if !retry_sanitized(&failure, false, &clean) {
                    return Err(failure
                        .into_error()
                        .context(format!("search failed for query {}", query.id)));
                }
                eprintln!(
                    "query {} rejected as query syntax; retrying sanitized",
                    query.id
                );
                let pools = search_sources(memex, &clean, sources, limit).map_err(|f| {
                    f.into_error()
                        .context(format!("sanitized search failed for query {}", query.id))
                })?;
                query.text = clean;
                out.sanitized.push(query.id.clone());
                pools
            }
        };
        for (slot, pool) in out.pools.iter_mut().zip(per_source) {
            slot.push(pool);
        }
        kept.push(query);
        if (n + 1) % 50 == 0 {
            eprintln!("  candidates: {}/{total} queries", n + 1);
        }
    }
    *queries = kept;
    eprintln!(
        "candidates for {} queries x {} sources in {:.1}s ({} sanitized, {} skipped)",
        queries.len(),
        sources.len(),
        started.elapsed().as_secs_f64(),
        out.sanitized.len(),
        out.skipped.len()
    );
    Ok(out)
}

/// Loads document text from corpus transcripts, caching by document id.
struct Corpus {
    dir: PathBuf,
    cache: HashMap<String, String>,
    missing: usize,
}

impl Corpus {
    fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            cache: HashMap::new(),
            missing: 0,
        }
    }

    /// Returns at most `max_chars` characters; an unknown or unsafe id yields empty text.
    fn text(&mut self, doc_id: &str, max_chars: usize) -> String {
        if !self.cache.contains_key(doc_id) {
            let text = if is_safe_doc_id(doc_id) {
                fs::read_to_string(self.dir.join(format!("{doc_id}.jsonl")))
                    .ok()
                    .map(|raw| transcript_text(&raw))
            } else {
                None
            };
            if text.is_none() {
                self.missing += 1;
            }
            self.cache
                .insert(doc_id.to_string(), text.unwrap_or_default());
        }
        self.cache
            .get(doc_id)
            .map(|text| truncate_chars(text, max_chars))
            .unwrap_or_default()
    }
}

fn is_safe_doc_id(id: &str) -> bool {
    !id.is_empty()
        && !id.starts_with('.')
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Extracts message text from Claude transcript lines, keeping raw lines that are not JSON.
fn transcript_text(raw: &str) -> String {
    let mut parts = Vec::new();
    for line in raw.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            parts.push(line.to_string());
            continue;
        };
        match value.pointer("/message/content") {
            Some(Value::String(text)) => parts.push(text.clone()),
            Some(Value::Array(blocks)) => parts.extend(
                blocks
                    .iter()
                    .filter_map(|b| b.get("text")?.as_str().map(str::to_string)),
            ),
            _ => {}
        }
    }
    parts.join("\n")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn tail_chars(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    text.chars().skip(count.saturating_sub(max_chars)).collect()
}

struct SourceRun {
    source: CandidateSource,
    pools: Vec<Vec<String>>,
    arms: Vec<Arm>,
}

struct Arm {
    name: String,
    /// One ranking per evaluated query, in query order.
    rankings: Vec<Vec<String>>,
    latencies_ms: Vec<f64>,
    load_seconds: Option<f64>,
}

impl Arm {
    fn new(name: &str, rankings: Vec<Vec<String>>) -> Self {
        Self {
            name: name.to_string(),
            rankings,
            latencies_ms: Vec::new(),
            load_seconds: None,
        }
    }
}

fn run_local_reranker(
    label: &str,
    model: &RerankerModel,
    model_cache: &Path,
    queries: &[Query],
    results: &mut [SourceRun],
    corpus: &mut Corpus,
    doc_chars: usize,
) -> Result<()> {
    if std::env::var_os("HF_HOME").is_some() {
        eprintln!("note: HF_HOME is set and overrides the reranker cache dir");
    }
    eprintln!("loading reranker {label}");
    let started = Instant::now();
    let mut reranker = TextRerank::try_new(
        RerankInitOptions::new(model.clone())
            .with_cache_dir(model_cache.to_path_buf())
            .with_show_download_progress(true),
    )
    .with_context(|| format!("failed to load reranker {label}"))?;
    let load_seconds = started.elapsed().as_secs_f64();

    for run in results.iter_mut() {
        let mut arm = Arm::new(label, Vec::with_capacity(queries.len()));
        arm.load_seconds = Some(load_seconds);
        for (query, pool) in queries.iter().zip(&run.pools) {
            if pool.is_empty() {
                arm.rankings.push(Vec::new());
                continue;
            }
            let texts: Vec<String> = pool.iter().map(|id| corpus.text(id, doc_chars)).collect();
            let docs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let started = Instant::now();
            let ranked = reranker
                .rerank(query.text.as_str(), &docs, false, None)
                .with_context(|| format!("{label} failed on query {}", query.id))?;
            arm.latencies_ms
                .push(started.elapsed().as_secs_f64() * 1000.0);
            arm.rankings.push(
                ranked
                    .iter()
                    .filter_map(|r| pool.get(r.index).cloned())
                    .collect(),
            );
        }
        eprintln!(
            "{label} on {}: p50 {:.1} ms, p95 {:.1} ms",
            run.source,
            percentile(&arm.latencies_ms, 0.50),
            percentile(&arm.latencies_ms, 0.95)
        );
        run.arms.push(arm);
    }
    Ok(())
}

/// One object per candidate source:
/// `{"dataset", "source", "queries": [{"id", "text", "candidates": [{"docid", "text"}]}]}`.
fn candidates_dump(
    dataset: &str,
    queries: &[Query],
    results: &[SourceRun],
    corpus: &mut Corpus,
    doc_chars: usize,
) -> Value {
    let mut out = Vec::new();
    for run in results {
        let mut rows = Vec::new();
        for (query, pool) in queries.iter().zip(&run.pools) {
            let candidates: Vec<Value> = pool
                .iter()
                .map(|id| json!({"docid": id, "text": corpus.text(id, doc_chars)}))
                .collect();
            rows.push(json!({"id": query.id, "text": query.text, "candidates": candidates}));
        }
        out.push(json!({"dataset": dataset, "source": run.source.name(), "queries": rows}));
    }
    Value::Array(out)
}

/// One entry of an external scores file. Fields other than `queries` are optional and
/// unknown fields are ignored.
#[derive(Debug, Deserialize)]
struct ExternalScores {
    arm: Option<String>,
    dataset: Option<String>,
    source: Option<String>,
    model: Option<String>,
    /// Scores by query id, then document id; a null score counts as 0.
    queries: HashMap<String, HashMap<String, Option<f64>>>,
    usage: Option<Value>,
    truncated: Option<bool>,
    dropped_queries: Option<u64>,
    budget_stop: Option<bool>,
    error: Option<Value>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ExternalFile {
    One(Box<ExternalScores>),
    Many(Vec<ExternalScores>),
}

fn load_external_scores(path: &Path) -> Result<Vec<ExternalScores>> {
    Ok(match read_json::<ExternalFile>(path)? {
        ExternalFile::One(entry) => vec![*entry],
        ExternalFile::Many(entries) => entries,
    })
}

/// Builds an arm from the file entry matching each source; an entry without `dataset` or
/// `source` matches any. Only queries present in the entry are judged, pool documents
/// without a score get 0, and equal scores keep the memex order.
fn external_arm(
    arm: &str,
    path: &Path,
    entries: &[ExternalScores],
    dataset: &str,
    queries: &[Query],
    results: &[SourceRun],
) -> Result<JudgedArm> {
    let mut per_source = Vec::new();
    let mut source_meta = Map::new();
    for run in results {
        let mut matching = entries.iter().filter(|e| {
            e.dataset.as_deref().is_none_or(|d| d == dataset)
                && e.source.as_deref().is_none_or(|s| s == run.source.name())
        });
        let Some(entry) = matching.next() else {
            continue;
        };
        if matching.next().is_some() {
            bail!(
                "{} has several entries for {dataset}/{}",
                path.display(),
                run.source
            );
        }
        let mut judged = JudgedSource::new(run.source);
        let mut unscored_docs = 0usize;
        for (qi, (query, pool)) in queries.iter().zip(&run.pools).enumerate() {
            let Some(scores) = entry.queries.get(&query.id) else {
                continue;
            };
            let aligned: Vec<f64> = pool
                .iter()
                .map(|doc| match scores.get(doc) {
                    Some(score) => score.filter(|s| s.is_finite()).unwrap_or(0.0),
                    None => {
                        unscored_docs += 1;
                        0.0
                    }
                })
                .collect();
            judged.query_indexes.push(qi);
            judged.rankings.push(
                order_by_scores(&aligned)
                    .into_iter()
                    .filter_map(|i| pool.get(i).cloned())
                    .collect(),
            );
            judged.probabilities.push(aligned);
        }
        let evaluated: HashSet<&str> = queries.iter().map(|q| q.id.as_str()).collect();
        source_meta.insert(
            run.source.name().into(),
            json!({
                "queries_in_file": entry.queries.len(),
                "queries_judged": judged.query_indexes.len(),
                "queries_not_evaluated": entry.queries.keys().filter(|q| !evaluated.contains(q.as_str())).count(),
                "pool_docs_without_score": unscored_docs,
                "file_arm": entry.arm,
                "model": entry.model,
                "usage": entry.usage,
                "truncated": entry.truncated,
                "dropped_queries": entry.dropped_queries,
                "budget_stop": entry.budget_stop,
                "error": entry.error,
            }),
        );
        per_source.push(judged);
    }
    if per_source.is_empty() {
        bail!(
            "{} has no entry for dataset {dataset} and the evaluated sources",
            path.display()
        );
    }
    let summary = per_source
        .iter()
        .filter_map(|j| {
            let meta = source_meta.get(j.source.name())?;
            let usage = |key: &str, decimals: usize| {
                meta.pointer(&format!("/usage/{key}"))
                    .and_then(Value::as_f64)
                    .map_or_else(|| "-".to_string(), |v| format!("{v:.decimals$}"))
            };
            let cost = match usage("est_cost_usd", 4) {
                missing if missing == "-" => usage("cost", 4),
                found => found,
            };
            Some(format!(
                "{}: judged {}, truncated {}, dropped_queries {}, budget_stop {}, error {}, unscored docs {}, cost ${cost}, latency p50/p95 {}/{} ms",
                j.source,
                j.query_indexes.len(),
                meta.get("truncated").unwrap_or(&Value::Null),
                meta.get("dropped_queries").unwrap_or(&Value::Null),
                meta.get("budget_stop").unwrap_or(&Value::Null),
                meta.get("error").is_some_and(|e| !e.is_null()),
                meta.get("pool_docs_without_score").unwrap_or(&Value::Null),
                usage("latency_ms_p50", 0),
                usage("latency_ms_p95", 0),
            ))
        })
        .collect::<Vec<_>>()
        .join("; ");
    Ok(JudgedArm {
        name: arm.to_string(),
        summary,
        meta: json!({
            "kind": "external",
            "path": path.display().to_string(),
            "sources": source_meta,
        }),
        per_source,
    })
}

/// Outcome of one OpenRouter arm (LLM judge or hosted reranker), shared across sources.
struct HostedRun {
    name: String,
    endpoint: &'static str,
    model: String,
    status: String,
    /// Sum of `usage.cost` reported by this arm's responses.
    reported_cost: f64,
    calls: usize,
    failures: Vec<String>,
    /// `<source> query <id>` entries retried with shorter documents after a length rejection.
    short_doc_retries: Vec<String>,
    per_source: Vec<JudgedSource>,
}

impl HostedRun {
    fn new(
        name: String,
        endpoint: &'static str,
        model: &str,
        results: &[SourceRun],
        sources: &[CandidateSource],
    ) -> Self {
        Self {
            name,
            endpoint,
            model: model.to_string(),
            status: "complete".to_string(),
            reported_cost: 0.0,
            calls: 0,
            failures: Vec::new(),
            short_doc_retries: Vec::new(),
            per_source: results
                .iter()
                .filter(|r| sources.contains(&r.source))
                .map(|r| JudgedSource::new(r.source))
                .collect(),
        }
    }

    fn into_judged(self) -> JudgedArm {
        let mut summary = format!(
            "status: {}, calls {}, failures {}, reported cost ${:.4}",
            self.status,
            self.calls,
            self.failures.len(),
            self.reported_cost
        );
        if !self.short_doc_retries.is_empty() {
            summary.push_str(&format!(
                ", short-doc retries {}",
                self.short_doc_retries.len()
            ));
        }
        JudgedArm {
            name: self.name,
            summary,
            meta: json!({
                "kind": "openrouter",
                "endpoint": self.endpoint,
                "model": self.model,
                "status": self.status,
                "calls": self.calls,
                "reported_cost_usd": self.reported_cost,
                "failures": self.failures,
                "short_doc_retries": self.short_doc_retries,
                "short_doc_chars": SHORT_DOC_CHARS,
            }),
            per_source: self.per_source,
        }
    }
}

/// A score-based arm evaluated only on the queries it judged: an OpenRouter arm or an
/// external scores file.
struct JudgedArm {
    name: String,
    /// One-line status printed above the arm's subset table.
    summary: String,
    /// Run metadata copied into the results JSON.
    meta: Value,
    per_source: Vec<JudgedSource>,
}

struct JudgedSource {
    source: CandidateSource,
    /// Indexes into the evaluated query list, in judging order.
    query_indexes: Vec<usize>,
    rankings: Vec<Vec<String>>,
    /// Scores aligned with the candidate pool of each judged query.
    probabilities: Vec<Vec<f64>>,
    /// Wall time per successful call, including retries.
    latencies_ms: Vec<f64>,
}

impl JudgedSource {
    fn new(source: CandidateSource) -> Self {
        Self {
            source,
            query_indexes: Vec::new(),
            rankings: Vec::new(),
            probabilities: Vec::new(),
            latencies_ms: Vec::new(),
        }
    }

    fn push(&mut self, qi: usize, pool: &[String], scores: Vec<f64>) {
        self.query_indexes.push(qi);
        self.rankings.push(
            order_by_scores(&scores)
                .into_iter()
                .filter_map(|i| pool.get(i).cloned())
                .collect(),
        );
        self.probabilities.push(scores);
    }
}

/// Spending guard shared by every OpenRouter arm in one run.
///
/// Key usage is read at the start and after every [`BUDGET_CHECK_EVERY`] calls; a call is
/// refused once key usage or the summed `usage.cost` since the start exceeds the limit, or
/// once a key check fails. A refusal is permanent for the run.
struct Budget {
    limit_usd: f64,
    start_usage: f64,
    last_usage: f64,
    reported_cost: f64,
    calls: usize,
    calls_since_check: usize,
    stopped: Option<String>,
}

impl Budget {
    fn start(client: &OpenRouter, limit_usd: f64) -> Result<Self> {
        let usage = client.key_usage()?;
        Ok(Self::with_usage(limit_usd, usage))
    }

    fn with_usage(limit_usd: f64, start_usage: f64) -> Self {
        Self {
            limit_usd,
            start_usage,
            last_usage: start_usage,
            reported_cost: 0.0,
            calls: 0,
            calls_since_check: 0,
            stopped: None,
        }
    }

    /// Returns the stop reason when no further call may be made.
    fn admit(&mut self, key_usage: impl FnOnce() -> Result<f64>) -> Result<(), String> {
        if let Some(reason) = &self.stopped {
            return Err(reason.clone());
        }
        if self.calls_since_check >= BUDGET_CHECK_EVERY {
            self.calls_since_check = 0;
            match key_usage() {
                Ok(usage) => self.last_usage = usage,
                Err(err) => return self.stop(format!("key usage check failed: {err:#}")),
            }
        }
        let spent = self.last_usage - self.start_usage;
        if spent > self.limit_usd {
            return self.stop(format!(
                "key usage ${spent:.4} exceeds budget ${:.2}",
                self.limit_usd
            ));
        }
        if self.reported_cost > self.limit_usd {
            return self.stop(format!(
                "reported cost ${:.4} exceeds budget ${:.2}",
                self.reported_cost, self.limit_usd
            ));
        }
        Ok(())
    }

    fn stop(&mut self, reason: String) -> Result<(), String> {
        self.stopped = Some(reason.clone());
        Err(reason)
    }

    fn record(&mut self, cost: Option<f64>) {
        self.calls += 1;
        self.calls_since_check += 1;
        self.reported_cost += cost.unwrap_or(0.0);
    }

    fn finish(&mut self, client: &OpenRouter) -> Value {
        let final_check = match client.key_usage() {
            Ok(usage) => {
                self.last_usage = usage;
                None
            }
            Err(err) => Some(format!("{err:#}")),
        };
        let delta = self.last_usage - self.start_usage;
        eprintln!(
            "OpenRouter: {} calls, reported cost ${:.4}, key usage delta ${delta:.4}, budget ${:.2}",
            self.calls, self.reported_cost, self.limit_usd
        );
        json!({
            "budget_usd": self.limit_usd,
            "calls": self.calls,
            "reported_cost_usd": self.reported_cost,
            "key_usage_delta_usd": delta,
            "stopped": self.stopped,
            "final_key_check_error": final_check,
        })
    }
}

/// One scored call: scores aligned with the pool, or the failure.
struct Scored {
    scores: Vec<f64>,
    latency_ms: f64,
    short_doc_retry: bool,
}

/// Runs one OpenRouter arm over the first `limit` queries. Sources are interleaved per
/// query so a stop leaves them on the same subset; three consecutive failures stop the arm.
fn run_hosted_arm(
    run: &mut HostedRun,
    budget: &mut Budget,
    client: &OpenRouter,
    queries: &[Query],
    results: &[SourceRun],
    limit: usize,
    mut score: impl FnMut(&Query, &[String]) -> (Option<f64>, Result<Scored>),
) {
    let limit = limit.min(queries.len());
    eprintln!(
        "{}: {limit} queries x {} sources",
        run.name,
        run.per_source.len()
    );
    let mut consecutive_failures = 0usize;
    'queries: for (qi, query) in queries.iter().enumerate().take(limit) {
        for out in run.per_source.iter_mut() {
            let Some(pool) = results
                .iter()
                .find(|r| r.source == out.source)
                .and_then(|r| r.pools.get(qi))
            else {
                continue;
            };
            if pool.is_empty() {
                out.push(qi, pool, Vec::new());
                continue;
            }
            if let Err(reason) = budget.admit(|| client.key_usage()) {
                run.status = format!("aborted: {reason}");
                break 'queries;
            }
            let (cost, scored) = score(query, pool);
            budget.record(cost);
            run.calls += 1;
            run.reported_cost += cost.unwrap_or(0.0);
            match scored {
                Ok(scored) => {
                    consecutive_failures = 0;
                    if scored.short_doc_retry {
                        run.short_doc_retries
                            .push(format!("{} query {}", out.source, query.id));
                    }
                    out.latencies_ms.push(scored.latency_ms);
                    out.push(qi, pool, scored.scores);
                }
                Err(err) => {
                    consecutive_failures += 1;
                    let message = format!("{} query {}: {err:#}", out.source, query.id);
                    eprintln!("{} failure: {message}", run.name);
                    run.failures.push(message);
                    if consecutive_failures >= MAX_CONSECUTIVE_LLM_FAILURES {
                        run.status = "aborted: repeated failures".to_string();
                        break 'queries;
                    }
                }
            }
        }
        if (qi + 1) % 10 == 0 {
            eprintln!(
                "  {}: {}/{limit} queries, reported cost ${:.4}",
                run.name,
                qi + 1,
                run.reported_cost
            );
        }
    }
    eprintln!(
        "{} {}: {} calls, reported cost ${:.4}",
        run.name, run.status, run.calls, run.reported_cost
    );
}

fn run_llm_arm(
    args: &Args,
    client: &OpenRouter,
    budget: &mut Budget,
    queries: &[Query],
    results: &[SourceRun],
    corpus: &mut Corpus,
) -> HostedRun {
    let mut run = HostedRun::new(
        format!("llm:{}", args.llm_model),
        "chat/completions",
        &args.llm_model,
        results,
        &args.llm_sources,
    );
    let limit = args.llm_queries.unwrap_or(queries.len());
    run_hosted_arm(
        &mut run,
        budget,
        client,
        queries,
        results,
        limit,
        |query, pool| {
            let docs: Vec<String> = pool
                .iter()
                .map(|id| corpus.text(id, args.llm_doc_chars))
                .collect();
            let started = Instant::now();
            match client.judge(&args.llm_model, &query.text, &docs) {
                Ok(reply) => {
                    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
                    let scored =
                        parse_probabilities(&reply.content, docs.len()).map(|scores| Scored {
                            scores,
                            latency_ms,
                            short_doc_retry: false,
                        });
                    (reply.cost, scored)
                }
                Err(err) => (None, Err(err)),
            }
        },
    );
    run
}

fn run_remote_reranker(
    args: &Args,
    model: &str,
    client: &OpenRouter,
    budget: &mut Budget,
    queries: &[Query],
    results: &[SourceRun],
    corpus: &mut Corpus,
) -> HostedRun {
    let mut run = HostedRun::new(
        format!("or:{model}"),
        "rerank",
        model,
        results,
        &args.sources,
    );
    let limit = args.remote_rerank_queries.unwrap_or(queries.len());
    run_hosted_arm(
        &mut run,
        budget,
        client,
        queries,
        results,
        limit,
        |query, pool| {
            let docs: Vec<String> = pool
                .iter()
                .map(|id| corpus.text(id, args.rerank_doc_chars))
                .collect();
            let started = Instant::now();
            let (reply, short_doc_retry) = match client.rerank(model, &query.text, &docs) {
                Err(RerankError::TooLong(err)) => {
                    eprintln!("{model}: {err:#}; retrying with {SHORT_DOC_CHARS}-char documents");
                    let short: Vec<String> = docs
                        .iter()
                        .map(|d| truncate_chars(d, SHORT_DOC_CHARS))
                        .collect();
                    (client.rerank(model, &query.text, &short), true)
                }
                other => (other, false),
            };
            let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
            match reply {
                Ok(reply) => (
                    reply.cost,
                    Ok(Scored {
                        scores: reply.scores,
                        latency_ms,
                        short_doc_retry,
                    }),
                ),
                Err(RerankError::TooLong(err) | RerankError::Other(err)) => (None, Err(err)),
            }
        },
    );
    run
}

struct LlmReply {
    content: String,
    cost: Option<f64>,
}

struct RerankReply {
    scores: Vec<f64>,
    cost: Option<f64>,
}

enum RerankError {
    /// A 4xx response that says the input exceeds the model's length limit.
    TooLong(anyhow::Error),
    Other(anyhow::Error),
}

enum PostError {
    Transport(anyhow::Error),
    /// A non-2xx status with the redacted response body.
    Status {
        status: u16,
        body: String,
    },
    Invalid(anyhow::Error),
}

impl PostError {
    fn is_transient(&self) -> bool {
        match self {
            Self::Transport(_) => true,
            Self::Status { status, .. } => *status == 429 || *status >= 500,
            Self::Invalid(_) => false,
        }
    }

    fn into_error(self, what: &str) -> anyhow::Error {
        match self {
            Self::Transport(err) | Self::Invalid(err) => err.context(what.to_string()),
            Self::Status { status, body } => anyhow!(
                "{what} returned HTTP {status}: {}",
                truncate_chars(&body, MAX_ERROR_EXCERPT_CHARS)
            ),
        }
    }
}

struct OpenRouter<'a> {
    agent: Agent,
    key: &'a str,
}

impl<'a> OpenRouter<'a> {
    fn new(key: &'a str) -> Self {
        let tls = TlsConfig::builder()
            .provider(TlsProvider::NativeTls)
            .root_certs(RootCerts::PlatformVerifier)
            .build();
        let agent: Agent = Agent::config_builder()
            .tls_config(tls)
            .http_status_as_error(false)
            .max_redirects(0)
            .timeout_global(Some(LLM_TIMEOUT))
            .build()
            .into();
        Self { agent, key }
    }

    fn bearer(&self) -> String {
        format!("Bearer {}", self.key)
    }

    /// Replaces any occurrence of the API key so it never reaches an error message.
    fn redact(&self, text: &str) -> String {
        if self.key.is_empty() {
            text.to_string()
        } else {
            text.replace(self.key, "<redacted>")
        }
    }

    /// Returns all-time key usage in OpenRouter credits (USD).
    fn key_usage(&self) -> Result<f64> {
        let response = self
            .agent
            .get(OPENROUTER_KEY_URL)
            .header("Authorization", &self.bearer())
            .call()
            .map_err(|e| anyhow!("key request failed: {}", self.redact(&e.to_string())))?;
        let status = response.status().as_u16();
        let raw = response
            .into_body()
            .with_config()
            .limit(MAX_RESPONSE_BYTES)
            .read_to_vec()
            .map_err(|e| anyhow!("key response read failed: {}", self.redact(&e.to_string())))?;
        if !(200..300).contains(&status) {
            bail!(
                "key endpoint returned HTTP {status}: {}",
                truncate_chars(
                    &self.redact(&String::from_utf8_lossy(&raw)),
                    MAX_ERROR_EXCERPT_CHARS
                )
            );
        }
        let value: Value = serde_json::from_slice(&raw).context("key response is not JSON")?;
        value
            .pointer("/data/usage")
            .and_then(Value::as_f64)
            .ok_or_else(|| anyhow!("key response has no data.usage"))
    }

    fn judge(&self, model: &str, query: &str, docs: &[String]) -> Result<LlmReply> {
        let mut user = format!("Query: {query}\n\nDocuments:\n");
        for (i, doc) in docs.iter().enumerate() {
            user.push_str(&format!("\n{}: {}\n", doc_key(i), doc.replace('\n', " ")));
        }
        let keys: Vec<String> = (0..docs.len()).map(doc_key).collect();
        user.push_str(&format!(
            "\nReturn one JSON object with exactly these keys: {}.",
            keys.join(", ")
        ));
        let mut body = json!({
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        });
        loop {
            match self.post_with_retry(OPENROUTER_CHAT_URL, &body) {
                Ok(value) => {
                    let content = value
                        .pointer("/choices/0/message/content")
                        .and_then(Value::as_str)
                        .ok_or_else(|| anyhow!("response has no message content"))?
                        .to_string();
                    let cost = value.pointer("/usage/cost").and_then(Value::as_f64);
                    return Ok(LlmReply { content, cost });
                }
                Err(PostError::Status {
                    status: 400,
                    body: text,
                }) if text.contains("response_format") => {
                    let removed = body
                        .as_object_mut()
                        .and_then(|map| map.remove("response_format"));
                    if removed.is_none() {
                        return Err(PostError::Status {
                            status: 400,
                            body: text,
                        }
                        .into_error("chat"));
                    }
                    eprintln!("model rejected response_format; retrying without it");
                }
                Err(err) => return Err(err.into_error("chat")),
            }
        }
    }

    fn rerank(
        &self,
        model: &str,
        query: &str,
        docs: &[String],
    ) -> Result<RerankReply, RerankError> {
        let body = rerank_body(model, query, docs);
        match self.post_with_retry(OPENROUTER_RERANK_URL, &body) {
            Ok(value) => {
                let scores = parse_rerank_scores(&value, docs.len()).map_err(RerankError::Other)?;
                let cost = value.pointer("/usage/cost").and_then(Value::as_f64);
                Ok(RerankReply { scores, cost })
            }
            Err(PostError::Status { status, body }) if is_length_rejection(status, &body) => Err(
                RerankError::TooLong(PostError::Status { status, body }.into_error("rerank")),
            ),
            Err(err) => Err(RerankError::Other(err.into_error("rerank"))),
        }
    }

    /// Posts with up to [`LLM_ATTEMPTS`] attempts for transport errors, 429, and 5xx.
    fn post_with_retry(&self, url: &str, body: &Value) -> Result<Value, PostError> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            match self.post_json(url, body) {
                Err(err) if err.is_transient() && attempt < LLM_ATTEMPTS => {
                    eprintln!(
                        "transient OpenRouter error (attempt {attempt}): {:#}",
                        err.into_error(url)
                    );
                    thread::sleep(LLM_BACKOFF * 2u32.saturating_pow(attempt - 1));
                }
                other => return other,
            }
        }
    }

    fn post_json(&self, url: &str, body: &Value) -> Result<Value, PostError> {
        let response = self
            .agent
            .post(url)
            .header("Authorization", &self.bearer())
            .send_json(body)
            .map_err(|e| PostError::Transport(anyhow!("{}", self.redact(&e.to_string()))))?;
        let status = response.status().as_u16();
        let mut response_body = response.into_body();
        if !(200..300).contains(&status) {
            let mut raw = Vec::new();
            // A partial error body still yields a useful message.
            let _ = response_body
                .as_reader()
                .take(MAX_ERROR_BODY_BYTES)
                .read_to_end(&mut raw);
            return Err(PostError::Status {
                status,
                body: self.redact(&String::from_utf8_lossy(&raw)),
            });
        }
        let raw = response_body
            .with_config()
            .limit(MAX_RESPONSE_BYTES)
            .read_to_vec()
            .map_err(|e| {
                PostError::Transport(anyhow!(
                    "response read failed: {}",
                    self.redact(&e.to_string())
                ))
            })?;
        serde_json::from_slice(&raw)
            .map_err(|e| PostError::Invalid(anyhow!("response is not JSON: {e}")))
    }
}

/// Request body for `POST /api/v1/rerank`; `top_n` asks for every document back.
fn rerank_body(model: &str, query: &str, docs: &[String]) -> Value {
    json!({
        "model": model,
        "query": query,
        "documents": docs,
        "top_n": docs.len(),
    })
}

/// Scores aligned with the request documents from a rerank response.
///
/// Each result's `index` is its input position. Results with an index out of range, a
/// missing index, or a non-numeric score are skipped, a repeated index keeps its first
/// score, and documents without a result score 0. Fails when no result is usable.
fn parse_rerank_scores(value: &Value, count: usize) -> Result<Vec<f64>> {
    let results = value
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("rerank response has no results array"))?;
    let mut scores = vec![0.0; count];
    let mut seen = vec![false; count];
    let mut matched = 0usize;
    for result in results {
        let Some(index) = result
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|i| usize::try_from(i).ok())
        else {
            continue;
        };
        let Some(score) = result
            .get("relevance_score")
            .and_then(Value::as_f64)
            .filter(|s| s.is_finite())
        else {
            continue;
        };
        if let (Some(slot), Some(flag)) = (scores.get_mut(index), seen.get_mut(index))
            && !*flag
        {
            *slot = score;
            *flag = true;
            matched += 1;
        }
    }
    if matched == 0 && count > 0 {
        bail!("rerank response has no usable results");
    }
    Ok(scores)
}

/// Whether a rejection says the input exceeds the model's length limit.
fn is_length_rejection(status: u16, body: &str) -> bool {
    const MARKERS: [&str; 10] = [
        "too long",
        "too many tokens",
        "too large",
        "context length",
        "context window",
        "maximum context",
        "token limit",
        "exceeds the maximum",
        "input length",
        "max tokens",
    ];
    if status == 413 {
        return true;
    }
    let body = body.to_ascii_lowercase();
    (400..500).contains(&status)
        && status != 429
        && MARKERS.iter().any(|marker| body.contains(marker))
}

fn doc_key(index: usize) -> String {
    format!("D{index:02}")
}

/// Parses `{"D00": p, ...}` from model output into probabilities aligned with candidates.
///
/// Text around the outermost braces is ignored, a single nested object is unwrapped,
/// keys are matched case-insensitively by number, values may be numbers or numeric
/// strings (a trailing `%` divides by 100), out-of-range values are clamped to [0, 1],
/// and missing ids get 0. Fails when no key matches a candidate.
fn parse_probabilities(content: &str, count: usize) -> Result<Vec<f64>> {
    let start = content
        .find('{')
        .ok_or_else(|| anyhow!("no JSON object in model output"))?;
    let end = content
        .rfind('}')
        .filter(|end| *end > start)
        .ok_or_else(|| anyhow!("unterminated JSON object in model output"))?;
    let slice = content
        .get(start..=end)
        .ok_or_else(|| anyhow!("invalid JSON object bounds"))?;
    let mut map: Map<String, Value> =
        serde_json::from_str(slice).context("model output is not a JSON object")?;
    if !map.keys().any(|k| doc_index(k).is_some())
        && map.len() == 1
        && let Some(Value::Object(inner)) = map.values().next()
    {
        map = inner.clone();
    }
    let mut probs = vec![0.0; count];
    let mut matched = 0;
    for (key, value) in &map {
        if let Some(slot) = doc_index(key).and_then(|i| probs.get_mut(i)) {
            *slot = value_probability(value);
            matched += 1;
        }
    }
    if matched == 0 {
        bail!("model output has no candidate ids");
    }
    Ok(probs)
}

fn doc_index(key: &str) -> Option<usize> {
    let key = key.trim();
    let digits = key.strip_prefix('D').or_else(|| key.strip_prefix('d'))?;
    digits.parse().ok()
}

fn value_probability(value: &Value) -> f64 {
    let raw = match value {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => {
            let s = s.trim();
            match s.strip_suffix('%') {
                Some(pct) => pct.trim().parse::<f64>().map_or(0.0, |p| p / 100.0),
                None => s.parse().unwrap_or(0.0),
            }
        }
        Value::Bool(true) => 1.0,
        _ => 0.0,
    };
    if raw.is_finite() {
        raw.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Candidate indexes by descending score; equal scores keep candidate order.
fn order_by_scores(scores: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| {
        let sa = scores.get(*a).copied().unwrap_or(0.0);
        let sb = scores.get(*b).copied().unwrap_or(0.0);
        sb.total_cmp(&sa)
    });
    order
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct QueryMetrics {
    ndcg10: f64,
    mrr10: f64,
    p1: f64,
    recall: f64,
}

/// nDCG@k with linear gain equal to the grade and a `log2(rank + 1)` discount; the ideal
/// ranking uses every judged document, so documents missing from the pool lower the score.
fn ndcg_at(ranking: &[String], rels: &HashMap<String, f64>, k: usize) -> f64 {
    let dcg: f64 = ranking
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, doc)| rels.get(doc).copied().unwrap_or(0.0) / discount(i))
        .sum();
    let mut ideal: Vec<f64> = rels.values().copied().filter(|g| *g > 0.0).collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, g)| g / discount(i))
        .sum();
    if idcg > 0.0 { dcg / idcg } else { 0.0 }
}

fn discount(zero_based_rank: usize) -> f64 {
    (zero_based_rank as f64 + 2.0).log2()
}

fn is_relevant(doc: &str, rels: &HashMap<String, f64>) -> bool {
    rels.get(doc).is_some_and(|g| *g > 0.0)
}

fn mrr_at(ranking: &[String], rels: &HashMap<String, f64>, k: usize) -> f64 {
    ranking
        .iter()
        .take(k)
        .position(|doc| is_relevant(doc, rels))
        .map_or(0.0, |i| 1.0 / (i as f64 + 1.0))
}

fn precision_at_1(ranking: &[String], rels: &HashMap<String, f64>) -> f64 {
    match ranking.first() {
        Some(doc) if is_relevant(doc, rels) => 1.0,
        _ => 0.0,
    }
}

fn recall_at(ranking: &[String], rels: &HashMap<String, f64>, k: usize) -> f64 {
    let total = rels.values().filter(|g| **g > 0.0).count();
    if total == 0 {
        return 0.0;
    }
    let found = ranking
        .iter()
        .take(k)
        .filter(|doc| is_relevant(doc, rels))
        .count();
    found as f64 / total as f64
}

fn query_metrics(ranking: &[String], rels: &HashMap<String, f64>, k: usize) -> QueryMetrics {
    QueryMetrics {
        ndcg10: ndcg_at(ranking, rels, NDCG_K),
        mrr10: mrr_at(ranking, rels, NDCG_K),
        p1: precision_at_1(ranking, rels),
        recall: recall_at(ranking, rels, k),
    }
}

fn mean_metrics(rows: &[QueryMetrics]) -> QueryMetrics {
    if rows.is_empty() {
        return QueryMetrics::default();
    }
    let n = rows.len() as f64;
    QueryMetrics {
        ndcg10: rows.iter().map(|m| m.ndcg10).sum::<f64>() / n,
        mrr10: rows.iter().map(|m| m.mrr10).sum::<f64>() / n,
        p1: rows.iter().map(|m| m.p1).sum::<f64>() / n,
        recall: rows.iter().map(|m| m.recall).sum::<f64>() / n,
    }
}

/// Nearest-rank percentile of unsorted samples; 0 for no samples.
fn percentile(samples: &[f64], p: f64) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((p * sorted.len() as f64).ceil() as usize).max(1);
    sorted.get(rank - 1).copied().unwrap_or(0.0)
}

#[derive(Debug, PartialEq)]
struct ThresholdRow {
    threshold: f64,
    mean_kept: f64,
    /// Mean over queries with a relevant document in the pool of the share kept.
    pool_recall_retained: f64,
}

/// Threshold view of LLM probabilities, relative to the relevant documents in each pool.
fn calibration(
    pools: &[&[String]],
    probabilities: &[Vec<f64>],
    rels: &[&HashMap<String, f64>],
) -> Vec<ThresholdRow> {
    THRESHOLDS
        .iter()
        .map(|&threshold| {
            let mut kept_total = 0usize;
            let mut recall_sum = 0.0;
            let mut recall_queries = 0usize;
            for ((pool, probs), rels) in pools.iter().zip(probabilities).zip(rels) {
                let mut relevant = 0usize;
                let mut retained = 0usize;
                for (doc, p) in pool.iter().zip(probs) {
                    let kept = *p >= threshold;
                    kept_total += usize::from(kept);
                    if is_relevant(doc, rels) {
                        relevant += 1;
                        retained += usize::from(kept);
                    }
                }
                if relevant > 0 {
                    recall_sum += retained as f64 / relevant as f64;
                    recall_queries += 1;
                }
            }
            ThresholdRow {
                threshold,
                mean_kept: if pools.is_empty() {
                    0.0
                } else {
                    kept_total as f64 / pools.len() as f64
                },
                pool_recall_retained: if recall_queries == 0 {
                    0.0
                } else {
                    recall_sum / recall_queries as f64
                },
            }
        })
        .collect()
}

fn metrics_json(m: &QueryMetrics, k: usize) -> Value {
    json!({
        "ndcg@10": m.ndcg10,
        "mrr@10": m.mrr10,
        "p@1": m.p1,
        format!("recall@{k}"): m.recall,
    })
}

/// Per-query metrics for one arm restricted to the given query indexes.
fn arm_rows(
    rankings: &[Vec<String>],
    queries: &[Query],
    subset: &[usize],
    k: usize,
) -> Vec<(usize, QueryMetrics)> {
    subset
        .iter()
        .filter_map(|&qi| {
            let query = queries.get(qi)?;
            let ranking = rankings.get(qi)?;
            Some((qi, query_metrics(ranking, &query.rels, k)))
        })
        .collect()
}

fn arm_json(arm: &Arm, queries: &[Query], subset: &[usize], k: usize, per_query: bool) -> Value {
    let rows = arm_rows(&arm.rankings, queries, subset, k);
    let metrics: Vec<QueryMetrics> = rows.iter().map(|(_, m)| *m).collect();
    let mut value = json!({
        "queries": rows.len(),
        "metrics": metrics_json(&mean_metrics(&metrics), k),
    });
    if let Some(map) = value.as_object_mut() {
        if !arm.latencies_ms.is_empty() {
            map.insert(
                "latency_ms".into(),
                json!({
                    "p50": percentile(&arm.latencies_ms, 0.50),
                    "p95": percentile(&arm.latencies_ms, 0.95),
                    "mean": arm.latencies_ms.iter().sum::<f64>() / arm.latencies_ms.len() as f64,
                    "samples": arm.latencies_ms.len(),
                }),
            );
        }
        if let Some(load) = arm.load_seconds {
            map.insert("load_seconds".into(), json!(load));
        }
        if per_query {
            let rows: Vec<Value> = rows
                .iter()
                .filter_map(|(qi, m)| {
                    let query = queries.get(*qi)?;
                    let top: Vec<&String> = arm.rankings.get(*qi)?.iter().take(NDCG_K).collect();
                    Some(json!({
                        "qid": query.id,
                        "metrics": metrics_json(m, k),
                        "top10": top,
                    }))
                })
                .collect();
            map.insert("per_query".into(), Value::Array(rows));
        }
    }
    value
}

/// A judged arm's rankings for one source as a full-length list, empty where not judged.
fn judged_ranking_arm(name: &str, run: &JudgedSource, query_count: usize) -> Arm {
    let mut rankings = vec![Vec::new(); query_count];
    for (qi, ranking) in run.query_indexes.iter().zip(&run.rankings) {
        if let Some(slot) = rankings.get_mut(*qi) {
            *slot = ranking.clone();
        }
    }
    let mut arm = Arm::new(name, rankings);
    arm.latencies_ms = run.latencies_ms.clone();
    arm
}

/// Queries judged by every judged arm covering `source`, ascending, with those arms as
/// full ranking lists; `None` when fewer than two judged arms cover the source.
fn judged_intersection(
    judged: &[JudgedArm],
    source: CandidateSource,
    query_count: usize,
) -> Option<(Vec<usize>, Vec<Arm>)> {
    let covering: Vec<(&str, &JudgedSource)> = judged
        .iter()
        .filter_map(|arm| {
            let jrun = arm.per_source.iter().find(|j| j.source == source)?;
            Some((arm.name.as_str(), jrun))
        })
        .collect();
    if covering.len() < 2 {
        return None;
    }
    let subset: Vec<usize> = (0..query_count)
        .filter(|qi| covering.iter().all(|(_, j)| j.query_indexes.contains(qi)))
        .collect();
    let arms = covering
        .iter()
        .map(|(name, jrun)| judged_ranking_arm(name, jrun, query_count))
        .collect();
    Some((subset, arms))
}

fn build_report(
    args: &Args,
    dataset: &Path,
    memex: &Memex,
    queries: &[Query],
    results: &[SourceRun],
    judged: &[JudgedArm],
    fixes: &QueryFixes,
) -> Value {
    let k = args.candidates;
    let all: Vec<usize> = (0..queries.len()).collect();
    let mut sources = Map::new();
    for run in results {
        let mut arms = Map::new();
        for arm in &run.arms {
            arms.insert(arm.name.clone(), arm_json(arm, queries, &all, k, true));
        }
        let mut subsets = Map::new();
        for arm in judged {
            if let Some(jrun) = arm.per_source.iter().find(|j| j.source == run.source) {
                subsets.insert(
                    arm.name.clone(),
                    subset_json(&arm.name, jrun, run, queries, k),
                );
            }
        }
        let intersection =
            judged_intersection(judged, run.source, queries.len()).map(|(subset, judged_arms)| {
                let mut arms = Map::new();
                for arm in run.arms.iter().chain(&judged_arms) {
                    arms.insert(arm.name.clone(), arm_json(arm, queries, &subset, k, false));
                }
                json!({"queries": subset.len(), "arms": arms})
            });
        let pool_sizes: Vec<usize> = run.pools.iter().map(Vec::len).collect();
        let source = json!({
            "queries": queries.len(),
            "mean_pool_size": pool_sizes.iter().sum::<usize>() as f64 / pool_sizes.len().max(1) as f64,
            "empty_pools": pool_sizes.iter().filter(|n| **n == 0).count(),
            "queries_sanitized": fixes.sanitized.len(),
            "sanitized_query_ids": fixes.sanitized,
            "queries_skipped": fixes.skipped.len(),
            "skipped_query_ids": fixes.skipped,
            "arms": arms,
            "judged_subsets": subsets,
            "judged_intersection": intersection,
            "candidates": run.pools.iter().zip(queries).map(|(p, q)| json!({"qid": q.id, "docs": p})).collect::<Vec<_>>(),
        });
        sources.insert(run.source.name().into(), source);
    }
    json!({
        "dataset": dataset.display().to_string(),
        "memex_bin": memex.bin.display().to_string(),
        "memex_root": memex.root.display().to_string(),
        "embedding_model": EMBED_MODEL,
        "candidates": k,
        "recency_weight": args.recency_weight,
        "rerank_doc_chars": args.rerank_doc_chars,
        "llm_doc_chars": args.llm_doc_chars,
        "queries_evaluated": queries.len(),
        "metric_notes": "nDCG@10 uses linear graded gain and log2(rank+1) discount with the ideal DCG over all judged documents; MRR@10 and P@1 count grade > 0; recall is over all judged documents within the candidate pool; judged-arm threshold recall is relative to relevant documents in the pool; each judged subset re-scores every arm on the queries that judged arm covers.",
        "judged_arms": judged.iter().map(|a| json!({"name": a.name, "meta": a.meta})).collect::<Vec<_>>(),
        "sources": sources,
    })
}

fn subset_inputs<'a>(
    jrun: &JudgedSource,
    run: &'a SourceRun,
    queries: &'a [Query],
) -> (Vec<&'a [String]>, Vec<&'a HashMap<String, f64>>) {
    let pools = jrun
        .query_indexes
        .iter()
        .filter_map(|qi| run.pools.get(*qi).map(Vec::as_slice))
        .collect();
    let rels = jrun
        .query_indexes
        .iter()
        .filter_map(|qi| queries.get(*qi).map(|q| &q.rels))
        .collect();
    (pools, rels)
}

fn subset_json(
    name: &str,
    jrun: &JudgedSource,
    run: &SourceRun,
    queries: &[Query],
    k: usize,
) -> Value {
    let subset = &jrun.query_indexes;
    let mut arms = Map::new();
    for arm in &run.arms {
        arms.insert(arm.name.clone(), arm_json(arm, queries, subset, k, false));
    }
    let arm = judged_ranking_arm(name, jrun, queries.len());
    arms.insert(name.to_string(), arm_json(&arm, queries, subset, k, true));
    let (pools, rels) = subset_inputs(jrun, run, queries);
    let calibration: Vec<Value> = calibration(&pools, &jrun.probabilities, &rels)
        .iter()
        .map(|row| {
            json!({
                "threshold": row.threshold,
                "mean_kept": row.mean_kept,
                "pool_recall_retained": row.pool_recall_retained,
            })
        })
        .collect();
    json!({
        "queries": subset.len(),
        "arms": arms,
        "calibration": calibration,
    })
}

fn print_tables(
    dataset: &Path,
    queries: &[Query],
    results: &[SourceRun],
    judged: &[JudgedArm],
    fixes: &QueryFixes,
    k: usize,
) {
    let name = dataset_name(dataset);
    let all: Vec<usize> = (0..queries.len()).collect();
    let sanitized: Vec<String> = results
        .iter()
        .map(|run| format!("{} {}", run.source, fixes.sanitized.len()))
        .collect();
    println!(
        "\n### {name}: all evaluated queries (queries_sanitized: {}; skipped: {})\n",
        sanitized.join(", "),
        fixes.skipped.len()
    );
    print_header(k);
    for run in results {
        for arm in &run.arms {
            print_row(run.source, arm, queries, &all, k);
        }
    }
    for judged_arm in judged {
        println!(
            "\n### {name}: {} subset ({})\n",
            judged_arm.name, judged_arm.summary
        );
        print_header(k);
        for run in results {
            let Some(jrun) = judged_arm
                .per_source
                .iter()
                .find(|j| j.source == run.source)
            else {
                continue;
            };
            for arm in &run.arms {
                print_row(run.source, arm, queries, &jrun.query_indexes, k);
            }
            let arm = judged_ranking_arm(&judged_arm.name, jrun, queries.len());
            print_row(run.source, &arm, queries, &jrun.query_indexes, k);
        }
        for run in results {
            let Some(jrun) = judged_arm
                .per_source
                .iter()
                .find(|j| j.source == run.source)
            else {
                continue;
            };
            let (pools, rels) = subset_inputs(jrun, run, queries);
            println!(
                "\n{} {} thresholds (recall relative to relevant docs in pool)\n",
                run.source, judged_arm.name
            );
            println!("| threshold | mean kept | pool recall retained |");
            println!("|---|---|---|");
            for row in calibration(&pools, &jrun.probabilities, &rels) {
                println!(
                    "| {:.1} | {:.2} | {:.3} |",
                    row.threshold, row.mean_kept, row.pool_recall_retained
                );
            }
        }
    }
    for run in results {
        let Some((subset, judged_arms)) = judged_intersection(judged, run.source, queries.len())
        else {
            continue;
        };
        println!(
            "\n### {name}: {} queries judged by every judged arm\n",
            run.source
        );
        print_header(k);
        for arm in run.arms.iter().chain(&judged_arms) {
            print_row(run.source, arm, queries, &subset, k);
        }
    }
}

fn dataset_name(dataset: &Path) -> String {
    dataset.file_name().map_or_else(
        || dataset.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

fn print_header(k: usize) {
    println!("| source | arm | n | nDCG@10 | MRR@10 | P@1 | R@{k} | p50 ms | p95 ms |");
    println!("|---|---|---|---|---|---|---|---|---|");
}

fn print_row(source: CandidateSource, arm: &Arm, queries: &[Query], subset: &[usize], k: usize) {
    let rows: Vec<QueryMetrics> = arm_rows(&arm.rankings, queries, subset, k)
        .into_iter()
        .map(|(_, m)| m)
        .collect();
    let m = mean_metrics(&rows);
    let (p50, p95) = if arm.latencies_ms.is_empty() {
        ("-".to_string(), "-".to_string())
    } else {
        (
            format!("{:.1}", percentile(&arm.latencies_ms, 0.50)),
            format!("{:.1}", percentile(&arm.latencies_ms, 0.95)),
        )
    };
    println!(
        "| {source} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {p50} | {p95} |",
        arm.name,
        rows.len(),
        m.ndcg10,
        m.mrr10,
        m.p1,
        m.recall
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rels(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(d, g)| (d.to_string(), *g)).collect()
    }

    fn ranking(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        let r = rels(&[("a", 2.0), ("b", 1.0)]);
        assert!(close(ndcg_at(&ranking(&["a", "b", "c"]), &r, 10), 1.0));
    }

    #[test]
    fn ndcg_uses_graded_gain_and_log2_discount() {
        let r = rels(&[("a", 2.0), ("b", 1.0)]);
        let dcg = 1.0 / 2f64.log2() + 2.0 / 3f64.log2();
        let idcg = 2.0 / 2f64.log2() + 1.0 / 3f64.log2();
        assert!(close(ndcg_at(&ranking(&["b", "a"]), &r, 10), dcg / idcg));
    }

    #[test]
    fn ndcg_ideal_includes_judged_docs_missing_from_pool() {
        let r = rels(&[("a", 1.0), ("missing", 1.0)]);
        let idcg = 1.0 + 1.0 / 3f64.log2();
        assert!(close(ndcg_at(&ranking(&["a", "x"]), &r, 10), 1.0 / idcg));
    }

    #[test]
    fn ndcg_ignores_ranks_beyond_cutoff() {
        let r = rels(&[("z", 1.0)]);
        let mut ids: Vec<String> = (0..10).map(|i| format!("n{i}")).collect();
        ids.push("z".into());
        assert!(close(ndcg_at(&ids, &r, 10), 0.0));
    }

    #[test]
    fn mrr_and_precision_use_first_relevant_hit() {
        let r = rels(&[("c", 1.0), ("d", 3.0)]);
        let list = ranking(&["a", "b", "c", "d"]);
        assert!(close(mrr_at(&list, &r, 10), 1.0 / 3.0));
        assert!(close(precision_at_1(&list, &r), 0.0));
        assert!(close(precision_at_1(&ranking(&["d"]), &r), 1.0));
        assert!(close(mrr_at(&list, &r, 2), 0.0));
    }

    #[test]
    fn recall_counts_relevant_docs_within_cutoff() {
        let r = rels(&[("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)]);
        assert!(close(recall_at(&ranking(&["a", "x", "c"]), &r, 30), 0.5));
        assert!(close(recall_at(&ranking(&["a", "x", "c"]), &r, 1), 0.25));
        assert!(close(recall_at(&[], &r, 30), 0.0));
    }

    #[test]
    fn percentile_is_nearest_rank() {
        let samples: Vec<f64> = (1..=20).map(f64::from).collect();
        assert!(close(percentile(&samples, 0.5), 10.0));
        assert!(close(percentile(&samples, 0.95), 19.0));
        assert!(close(percentile(&[], 0.5), 0.0));
    }

    #[test]
    fn parses_wrapped_json_with_missing_and_string_values() -> Result<()> {
        let content = "Here you go:\n```json\n{\"D00\": 0.9, \"d02\": \"0.4\", \"D1\": \"50%\", \"D07\": 0.2}\n```";
        let probs = parse_probabilities(content, 4)?;
        assert_eq!(probs, vec![0.9, 0.5, 0.4, 0.0]);
        Ok(())
    }

    #[test]
    fn clamps_out_of_range_and_ignores_non_numeric() -> Result<()> {
        let probs = parse_probabilities(
            r#"{"D00": 1.7, "D01": -0.2, "D02": "high", "D03": null}"#,
            4,
        )?;
        assert_eq!(probs, vec![1.0, 0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn unwraps_single_nested_object() -> Result<()> {
        let probs = parse_probabilities(r#"{"scores": {"D00": 0.3, "D01": 0.8}}"#, 2)?;
        assert_eq!(probs, vec![0.3, 0.8]);
        Ok(())
    }

    #[test]
    fn rejects_output_without_candidate_ids() {
        assert!(parse_probabilities("no json here", 3).is_err());
        assert!(parse_probabilities(r#"{"foo": 1}"#, 3).is_err());
        assert!(parse_probabilities("{not json}", 3).is_err());
    }

    #[test]
    fn order_by_scores_is_stable_for_ties() {
        assert_eq!(order_by_scores(&[0.2, 0.9, 0.2, 0.9]), vec![1, 3, 0, 2]);
    }

    #[test]
    fn calibration_counts_kept_docs_and_pool_recall() {
        let pool = ranking(&["a", "b", "c"]);
        let r = rels(&[("a", 1.0), ("c", 1.0), ("outside", 1.0)]);
        let rows = calibration(&[pool.as_slice()], &[vec![0.95, 0.5, 0.15]], &[&r]);
        let at = |t: f64| rows.iter().find(|row| close(row.threshold, t));
        assert_eq!(
            at(0.1),
            Some(&ThresholdRow {
                threshold: 0.1,
                mean_kept: 3.0,
                pool_recall_retained: 1.0
            })
        );
        assert_eq!(
            at(0.5),
            Some(&ThresholdRow {
                threshold: 0.5,
                mean_kept: 2.0,
                pool_recall_retained: 0.5
            })
        );
        assert_eq!(
            at(0.9),
            Some(&ThresholdRow {
                threshold: 0.9,
                mean_kept: 1.0,
                pool_recall_retained: 0.5
            })
        );
    }

    #[test]
    fn extracts_transcript_text_from_string_and_blocks() {
        let raw = concat!(
            r#"{"message":{"content":"plain text"}}"#,
            "\n",
            r#"{"message":{"content":[{"type":"text","text":"block"},{"type":"tool_use","name":"x"}]}}"#,
            "\nnot-json\n"
        );
        assert_eq!(transcript_text(raw), "plain text\nblock\nnot-json");
    }

    #[test]
    fn search_output_dedupes_and_skips_non_json() {
        let out = "progress line\n{\"session_id\":\"a\"}\n{\"session_id\":\"b\"}\n{\"session_id\":\"a\"}\n{\"session_id\":\"c\"}\n";
        assert_eq!(parse_search_output(out, 2), ranking(&["a", "b"]));
    }

    #[test]
    fn doc_ids_cannot_escape_corpus_dir() {
        assert!(is_safe_doc_id("10009203"));
        assert!(is_safe_doc_id("abc-1_2.x"));
        assert!(!is_safe_doc_id("../etc"));
        assert!(!is_safe_doc_id("a/b"));
        assert!(!is_safe_doc_id(""));
    }

    fn query(id: &str, pairs: &[(&str, f64)]) -> Query {
        Query {
            id: id.into(),
            text: id.into(),
            rels: rels(pairs),
        }
    }

    #[test]
    fn external_scores_evaluate_on_intersection_and_tolerate_extra_fields() -> Result<()> {
        let raw = r#"[
            {"arm": "jev", "dataset": "other", "source": "hybrid", "queries": {"q1": {"a": 1.0}}},
            {"arm": "jev", "dataset": "tiny", "source": "hybrid", "model": "m",
             "models_served": ["m"], "prompt_version": "generic-1",
             "queries": {"q1": {"c": 0.9, "a": 0.2, "b": null}, "q3": {"x": 0.5}, "unknown": {}},
             "usage": {"input_tokens": 10, "output_tokens": 2, "est_cost_usd": 0.0012, "calls": 2,
                       "estimated_calls": 0, "latency_ms_p50": null, "latency_ms_p95": 12.5},
             "truncated": true, "dropped_queries": 1, "budget_stop": true, "error": null}
        ]"#;
        let file: ExternalFile = serde_json::from_str(raw)?;
        let ExternalFile::Many(entries) = file else {
            bail!("expected a list");
        };
        let queries = vec![
            query("q1", &[("c", 2.0)]),
            query("q2", &[("y", 1.0)]),
            query("q3", &[("x", 1.0)]),
        ];
        let pools = vec![
            ranking(&["a", "b", "c", "d"]),
            ranking(&["y"]),
            ranking(&["w", "x"]),
        ];
        let results = vec![SourceRun {
            source: CandidateSource::Hybrid,
            arms: vec![Arm::new("none", pools.clone())],
            pools,
        }];
        let arm = external_arm(
            "jev",
            Path::new("s.json"),
            &entries,
            "tiny",
            &queries,
            &results,
        )?;
        let judged = arm.per_source.first().ok_or_else(|| anyhow!("no source"))?;
        assert_eq!(judged.query_indexes, vec![0, 2]);
        assert_eq!(
            judged.rankings,
            vec![ranking(&["c", "a", "b", "d"]), ranking(&["x", "w"])]
        );
        assert_eq!(
            judged.probabilities.first(),
            Some(&vec![0.2, 0.0, 0.9, 0.0])
        );
        assert!(arm.summary.contains("truncated true"));
        assert!(arm.summary.contains("dropped_queries 1"));
        assert!(arm.summary.contains("latency p50/p95 -/12 ms"));
        assert!(arm.summary.contains("cost $0.0012"));
        assert_eq!(
            arm.meta.pointer("/sources/hybrid/queries_not_evaluated"),
            Some(&json!(1))
        );

        let full = judged_ranking_arm(&arm.name, judged, queries.len());
        let rows = arm_rows(&full.rankings, &queries, &judged.query_indexes, 30);
        let mean = mean_metrics(&rows.iter().map(|(_, m)| *m).collect::<Vec<_>>());
        assert!(close(mean.p1, 1.0));
        assert!(close(mean.ndcg10, 1.0));
        Ok(())
    }

    #[test]
    fn external_scores_accept_single_object_without_dataset_or_source() -> Result<()> {
        let file: ExternalFile = serde_json::from_str(r#"{"queries": {"q1": {"b": 2, "a": 1}}}"#)?;
        let ExternalFile::One(entry) = file else {
            bail!("expected one object");
        };
        let queries = vec![query("q1", &[("b", 1.0)])];
        let results: Vec<SourceRun> = [CandidateSource::Lexical, CandidateSource::Hybrid]
            .into_iter()
            .map(|source| SourceRun {
                source,
                pools: vec![ranking(&["a", "b"])],
                arms: Vec::new(),
            })
            .collect();
        let arm = external_arm(
            "x",
            Path::new("s.json"),
            &[*entry],
            "any",
            &queries,
            &results,
        )?;
        assert_eq!(arm.per_source.len(), 2);
        assert!(
            arm.per_source
                .iter()
                .all(|j| j.rankings == vec![ranking(&["b", "a"])])
        );
        Ok(())
    }

    #[test]
    fn external_scores_without_matching_entry_fail() {
        let entries = vec![ExternalScores {
            arm: None,
            dataset: Some("other".into()),
            source: None,
            model: None,
            queries: HashMap::new(),
            usage: None,
            truncated: None,
            dropped_queries: None,
            budget_stop: None,
            error: None,
        }];
        let results = vec![SourceRun {
            source: CandidateSource::Lexical,
            pools: Vec::new(),
            arms: Vec::new(),
        }];
        assert!(external_arm("x", Path::new("s.json"), &entries, "tiny", &[], &results).is_err());
    }

    #[test]
    fn judged_intersection_keeps_queries_every_arm_judged() {
        let judged_source = |indexes: Vec<usize>| JudgedSource {
            source: CandidateSource::Hybrid,
            rankings: indexes.iter().map(|_| ranking(&["a"])).collect(),
            probabilities: indexes.iter().map(|_| vec![1.0]).collect(),
            latencies_ms: Vec::new(),
            query_indexes: indexes,
        };
        let arm = |name: &str, indexes: Vec<usize>| JudgedArm {
            name: name.into(),
            summary: String::new(),
            meta: Value::Null,
            per_source: vec![judged_source(indexes)],
        };
        let judged = vec![arm("llm", vec![0, 1, 3]), arm("jev", vec![3, 1, 2])];
        let (subset, arms) =
            judged_intersection(&judged, CandidateSource::Hybrid, 4).unwrap_or_default();
        assert_eq!(subset, vec![1, 3]);
        assert_eq!(arms.len(), 2);
        assert!(judged_intersection(&judged, CandidateSource::Lexical, 4).is_none());
        assert!(judged_intersection(&judged[..1], CandidateSource::Hybrid, 4).is_none());
    }

    #[test]
    fn rerank_scores_follow_input_index_not_response_order() -> Result<()> {
        let response = json!({
            "id": "r1",
            "model": "cohere/rerank-v3.5",
            "provider": "Cohere",
            "results": [
                {"index": 2, "relevance_score": 0.91, "document": {"text": "c"}},
                {"index": 0, "relevance_score": 0.40, "document": {"text": "a"}, "extra": true},
                {"index": 7, "relevance_score": 0.99},
                {"index": 1, "relevance_score": "high"},
                {"relevance_score": 0.5},
                {"index": 2, "relevance_score": 0.10}
            ],
            "usage": {"cost": 0.001, "search_units": 1}
        });
        let scores = parse_rerank_scores(&response, 4)?;
        assert_eq!(scores, vec![0.40, 0.0, 0.91, 0.0]);
        assert_eq!(order_by_scores(&scores), vec![2, 0, 1, 3]);
        Ok(())
    }

    #[test]
    fn rerank_scores_reject_unusable_responses() {
        assert!(parse_rerank_scores(&json!({"error": "x"}), 2).is_err());
        assert!(parse_rerank_scores(&json!({"results": []}), 2).is_err());
        assert!(
            parse_rerank_scores(
                &json!({"results": [{"index": 5, "relevance_score": 1.0}]}),
                2
            )
            .is_err()
        );
        assert!(parse_rerank_scores(&json!({"results": []}), 0).is_ok());
    }

    #[test]
    fn rerank_body_sends_every_document_in_order() {
        let docs = vec!["first".to_string(), "second".to_string()];
        assert_eq!(
            rerank_body("voyageai/rerank-2.5-lite", "q", &docs),
            json!({
                "model": "voyageai/rerank-2.5-lite",
                "query": "q",
                "documents": ["first", "second"],
                "top_n": 2,
            })
        );
    }

    #[test]
    fn length_rejections_are_recognised_only_for_client_errors() {
        assert!(is_length_rejection(
            400,
            "Input is Too Long for model context window"
        ));
        assert!(is_length_rejection(413, ""));
        assert!(!is_length_rejection(400, "invalid model id"));
        assert!(!is_length_rejection(429, "too many tokens per minute"));
        assert!(!is_length_rejection(500, "too long"));
    }

    #[test]
    fn budget_refuses_after_cost_or_key_usage_exceeds_limit() {
        let mut budget = Budget::with_usage(0.30, 10.0);
        assert!(budget.admit(|| Ok(10.0)).is_ok());
        budget.record(Some(0.31));
        assert!(budget.admit(|| Ok(10.0)).is_err());
        assert!(budget.admit(|| Ok(10.0)).is_err());

        let mut budget = Budget::with_usage(0.30, 10.0);
        for _ in 0..BUDGET_CHECK_EVERY {
            budget.record(None);
        }
        assert!(budget.admit(|| Ok(10.5)).is_err());

        let mut budget = Budget::with_usage(0.30, 10.0);
        for _ in 0..BUDGET_CHECK_EVERY {
            budget.record(None);
        }
        assert!(budget.admit(|| bail!("offline")).is_err());
        assert!(budget.stopped.is_some());
    }

    #[test]
    fn sanitize_query_keeps_alphanumerics_and_collapses_whitespace() {
        assert_eq!(
            sanitize_query("C++ Renaissance - marketing slogan?"),
            "C Renaissance marketing slogan"
        );
        assert_eq!(sanitize_query("foo: bar"), "foo bar");
        assert_eq!(
            sanitize_query("\"unbalanced  quote\tx"),
            "unbalanced quote x"
        );
        assert_eq!(sanitize_query("Ünïcode café 42"), "Ünïcode café 42");
        assert_eq!(sanitize_query("+-:\"()"), "");
    }

    #[test]
    fn only_syntax_errors_on_original_text_are_retried() {
        let syntax = SearchFailure::Syntax(anyhow!("Syntax Error"));
        let other = SearchFailure::Other(anyhow!("index missing"));
        assert!(is_query_syntax_error(
            "all machine searches failed: local: Syntax Error: foo: bar"
        ));
        assert!(!is_query_syntax_error("index not found"));
        assert!(retry_sanitized(&syntax, false, "foo bar"));
        assert!(!retry_sanitized(&syntax, true, "foo bar"));
        assert!(!retry_sanitized(&syntax, false, ""));
        assert!(!retry_sanitized(&other, false, "foo bar"));
    }

    #[test]
    fn external_arg_requires_arm_and_path() {
        assert_eq!(
            parse_external_arg("jev=/tmp/s.json"),
            Ok(("jev".to_string(), PathBuf::from("/tmp/s.json")))
        );
        assert!(parse_external_arg("jev").is_err());
        assert!(parse_external_arg("=x").is_err());
    }
}
