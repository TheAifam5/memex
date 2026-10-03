//! Reranking client for Cohere-style `/rerank` endpoints.

use std::fmt;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::remote_http::{ClientLimits, HttpClient, Purpose, RemoteEndpoint};
use crate::rerank::RerankBackend;

/// Longest accepted provider model name, in characters.
pub const MAX_MODEL_NAME_CHARS: usize = 200;
const LIMITS: ClientLimits = ClientLimits {
    max_retries: 3,
    max_timeout: Duration::from_secs(30),
    max_success_body_mib: 1,
    max_total: Some(Duration::from_secs(30)),
};

/// Request body layout of a rerank server.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RerankDialect {
    /// `{"model", "query", "documents"}`: Cohere, Jina, Voyage, vLLM, llama.cpp, and
    /// OpenRouter.
    #[default]
    Documents,
    /// `{"query", "texts", "truncate": true}`: Hugging Face text-embeddings-inference,
    /// which serves one model and truncates inputs to its window.
    Texts,
}

impl RerankDialect {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Documents => "documents",
            Self::Texts => "texts",
        }
    }
}

/// Resolved settings of a remote reranker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteRerankConfig {
    /// `base_url` is the full rerank endpoint URL, used verbatim.
    pub endpoint: RemoteEndpoint,
    /// The provider's model name; required by [`RerankDialect::Documents`] and not sent
    /// in [`RerankDialect::Texts`].
    pub model: Option<String>,
    pub dialect: RerankDialect,
}

/// Check a provider model name: non-empty, at most [`MAX_MODEL_NAME_CHARS`] characters,
/// and free of whitespace and control characters.
pub fn validate_model_name(model: &str) -> Result<()> {
    if model.is_empty() {
        bail!("remote rerank model name must not be empty");
    }
    if model.chars().count() > MAX_MODEL_NAME_CHARS {
        bail!("remote rerank model name is longer than {MAX_MODEL_NAME_CHARS} characters");
    }
    if model.chars().any(|c| c.is_control() || c.is_whitespace()) {
        bail!(
            "remote rerank model name {model:?} must not contain whitespace or control characters"
        );
    }
    Ok(())
}

/// Blocking client that scores documents through a remote rerank endpoint.
pub struct RemoteReranker {
    http: HttpClient,
    model: Option<String>,
    dialect: RerankDialect,
}

impl fmt::Debug for RemoteReranker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RemoteReranker")
            .field("endpoint", self.http.endpoint())
            .field("model", &self.model)
            .field("dialect", &self.dialect)
            .finish_non_exhaustive()
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum RerankRequest<'a> {
    Documents {
        model: &'a str,
        query: &'a str,
        documents: &'a [String],
    },
    Texts {
        query: &'a str,
        texts: &'a [String],
        truncate: bool,
    },
}

impl RemoteReranker {
    /// Creates a client for `config`.
    ///
    /// The endpoint's `max_retries` is clamped to 3 and its `timeout` to 30 s, and a
    /// request with its retries gets at most 30 s; responses larger than 1 MiB are
    /// rejected. Fails without a model in the [`RerankDialect::Documents`] dialect.
    pub fn new(config: &RemoteRerankConfig) -> Result<Self> {
        match (&config.model, config.dialect) {
            (Some(model), _) => validate_model_name(model)?,
            (None, RerankDialect::Documents) => {
                bail!("the documents rerank dialect requires a model name")
            }
            (None, RerankDialect::Texts) => {}
        }
        Ok(Self {
            http: HttpClient::new(config.endpoint.clone(), Purpose::Rerank, LIMITS)?,
            model: config.model.clone(),
            dialect: config.dialect,
        })
    }

    #[cfg(test)]
    pub(crate) fn set_backoff_base(&mut self, base: Duration) {
        self.http.set_backoff_base(base);
    }
}

impl RerankBackend for RemoteReranker {
    /// Relevance probabilities in `[0, 1]`, one per document in input order.
    ///
    /// Fails when the request fails or the response does not score every document
    /// exactly once with a finite number.
    fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let body = match (self.dialect, self.model.as_deref()) {
            (RerankDialect::Documents, Some(model)) => RerankRequest::Documents {
                model,
                query,
                documents,
            },
            (RerankDialect::Documents, None) => {
                bail!("the documents rerank dialect requires a model name")
            }
            (RerankDialect::Texts, _) => RerankRequest::Texts {
                query,
                texts: documents,
                truncate: true,
            },
        };
        let url = &self.http.endpoint().base_url;
        self.http.post_json_with(
            url,
            &body,
            |status, message| match status {
                401 | 403 => Some(anyhow!(
                    "rerank server rejected the request (HTTP {status}): {message}; check \
                     rerank_api_key or rerank_api_key_env"
                )),
                404 => Some(anyhow!(
                    "rerank server returned HTTP 404: {message}; rerank_url must be the full \
                     endpoint URL, such as https://api.cohere.com/v2/rerank"
                )),
                _ => None,
            },
            |raw| parse_scores(raw, documents.len()),
        )
    }
}

/// Scores in input order from a rerank response for `expected` documents.
///
/// The items are the top-level array, or the array under `results` or `data`. Each item
/// needs an integer `index` and a numeric `relevance_score` or `score`, and every index
/// below `expected` must appear exactly once.
fn parse_scores(raw: &[u8], expected: usize) -> Result<Vec<f32>> {
    let response: Value = serde_json::from_slice(raw)
        .map_err(|e| anyhow!("rerank server returned an invalid response body: {e}"))?;
    let items = match &response {
        Value::Array(items) => items,
        Value::Object(object) => match (object.get("results"), object.get("data")) {
            (Some(Value::Array(items)), _) | (None, Some(Value::Array(items))) => items,
            _ => bail!("rerank response has no `results` or `data` array"),
        },
        _ => bail!("rerank response is neither an object nor an array"),
    };
    if items.len() != expected {
        bail!(
            "rerank server returned {} results for {expected} documents",
            items.len()
        );
    }
    let mut slots: Vec<Option<f32>> = vec![None; expected];
    for item in items {
        let index = item
            .get("index")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow!("rerank result has no integer `index`"))?;
        let score = item
            .get("relevance_score")
            .or_else(|| item.get("score"))
            .ok_or_else(|| anyhow!("rerank result {index} has no `relevance_score` or `score`"))?
            .as_f64()
            .ok_or_else(|| anyhow!("rerank result {index} has a non-numeric score"))?;
        // A finite f64 beyond the f32 range becomes infinite here.
        let score = score as f32;
        if !score.is_finite() {
            bail!("rerank result {index} has a non-finite score");
        }
        let Some(slot) = usize::try_from(index)
            .ok()
            .and_then(|index| slots.get_mut(index))
        else {
            bail!("rerank server returned index {index} for {expected} documents");
        };
        if slot.replace(score).is_some() {
            bail!("rerank server returned index {index} twice");
        }
    }
    // Equal counts with no duplicates and no out-of-range index fill every slot.
    let scores = slots
        .into_iter()
        .enumerate()
        .map(|(index, slot)| slot.ok_or_else(|| anyhow!("rerank server omitted index {index}")))
        .collect::<Result<Vec<f32>>>()?;
    // Search ranking expects probabilities, as the local reranker returns. Hosted APIs
    // return relevance probabilities in [0, 1], while some self-hosted servers return raw
    // logits; a score outside [0, 1] marks the response as logits, and the sigmoid maps
    // every score of it. The sigmoid is monotonic, so the order is the same either way.
    if scores.iter().all(|score| (0.0..=1.0).contains(score)) {
        Ok(scores)
    } else {
        scores.into_iter().map(crate::rerank::sigmoid).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::remote_server::{
        KEY, Reply, endpoint, expect_err, join, reply, serve,
    };
    use serde_json::json;

    fn config(url: &str, key: Option<&str>, dialect: RerankDialect) -> RemoteRerankConfig {
        RemoteRerankConfig {
            endpoint: endpoint(url, key, 0),
            model: Some("rerank-test-1".to_owned()),
            dialect,
        }
    }

    fn client(url: &str, key: Option<&str>, retries: u32) -> Result<RemoteReranker> {
        let mut config = config(url, key, RerankDialect::Documents);
        config.endpoint.max_retries = retries;
        let mut reranker = RemoteReranker::new(&config)?;
        reranker.set_backoff_base(Duration::from_millis(1));
        Ok(reranker)
    }

    fn documents(count: usize) -> Vec<String> {
        (0..count).map(|i| format!("doc {i}")).collect()
    }

    /// Scores each document `0.1 * (index + 1)`, listed in reverse order.
    fn scored(_: usize, body: &Value) -> Reply {
        let count = body["documents"]
            .as_array()
            .or_else(|| body["texts"].as_array())
            .map_or(0, Vec::len);
        let results: Vec<Value> = (0..count)
            .rev()
            .map(|i| json!({"index": i, "relevance_score": 0.1 * (i as f64 + 1.0)}))
            .collect();
        reply(200, json!({"results": results}))
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{actual:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn documents_dialect_sends_model_and_bearer_key() -> Result<()> {
        let (base, server) = serve(scored)?;
        let url = format!("{base}/rerank");
        let scores = client(&url, Some(KEY), 0)?.rerank("needle", &documents(3))?;
        let captured = join(server)?;
        assert_close(&scores, &[0.1, 0.2, 0.3]);
        assert_eq!(captured.len(), 1);
        let request = captured.first().ok_or_else(|| anyhow!("no request"))?;
        assert_eq!(request.path, "/v1/rerank");
        assert_eq!(request.auth, Some(format!("Bearer {KEY}")));
        assert_eq!(
            request.body,
            json!({
                "model": "rerank-test-1",
                "query": "needle",
                "documents": ["doc 0", "doc 1", "doc 2"]
            })
        );
        Ok(())
    }

    #[test]
    fn texts_dialect_omits_model_and_keyless_authorization() -> Result<()> {
        let (base, server) = serve(|_, body| {
            let count = body["texts"].as_array().map_or(0, Vec::len);
            let items: Vec<Value> = (0..count)
                .map(|i| json!({"index": i, "score": 0.5}))
                .collect();
            reply(200, Value::Array(items))
        })?;
        let url = format!("{base}/rerank");
        let mut unnamed = config(&url, None, RerankDialect::Texts);
        unnamed.model = None;
        for config in [config(&url, None, RerankDialect::Texts), unnamed] {
            let scores = RemoteReranker::new(&config)?.rerank("needle", &documents(2))?;
            assert_close(&scores, &[0.5, 0.5]);
        }
        let captured = join(server)?;
        assert_eq!(captured.len(), 2);
        for request in &captured {
            assert_eq!(request.auth, None);
            assert_eq!(
                request.body,
                json!({"query": "needle", "texts": ["doc 0", "doc 1"], "truncate": true})
            );
        }
        Ok(())
    }

    #[test]
    fn empty_documents_send_no_request() -> Result<()> {
        let mut reranker = client("http://127.0.0.1:1/rerank", None, 0)?;
        assert!(reranker.rerank("needle", &[])?.is_empty());
        Ok(())
    }

    #[test]
    fn parses_results_data_and_bare_arrays_in_any_order() -> Result<()> {
        for body in [
            json!({"results": [
                {"index": 2, "relevance_score": 0.9, "document": {"text": "x"}},
                {"index": 0, "relevance_score": 0.2},
                {"index": 1, "relevance_score": 0.4}
            ]}),
            json!({"object": "list", "data": [
                {"index": 1, "relevance_score": 0.4},
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2}
            ]}),
            json!([
                {"index": 1, "score": 0.4},
                {"index": 0, "score": 0.2},
                {"index": 2, "score": 0.9}
            ]),
        ] {
            let scores = parse_scores(body.to_string().as_bytes(), 3)?;
            assert_close(&scores, &[0.2, 0.4, 0.9]);
        }
        Ok(())
    }

    #[test]
    fn rejects_incomplete_or_invalid_results() {
        for (body, expected) in [
            (
                r#"{"results":[{"index":0,"relevance_score":0.1}]}"#,
                "1 results for 2",
            ),
            (
                r#"{"results":[{"index":0,"relevance_score":0.1},{"index":0,"relevance_score":0.2}]}"#,
                "index 0 twice",
            ),
            (
                r#"{"results":[{"index":0,"relevance_score":0.1},{"index":2,"relevance_score":0.2}]}"#,
                "index 2 for 2 documents",
            ),
            (
                r#"{"results":[{"index":0,"relevance_score":0.1},{"relevance_score":0.2}]}"#,
                "no integer `index`",
            ),
            (
                r#"{"results":[{"index":0,"relevance_score":0.1},{"index":1.0,"relevance_score":0.2}]}"#,
                "no integer `index`",
            ),
            (
                r#"{"results":[{"index":0,"relevance_score":0.1},{"index":-1,"relevance_score":0.2}]}"#,
                "no integer `index`",
            ),
            (
                r#"{"results":[{"index":0,"relevance_score":0.1},{"index":1,"relevance_score":"0.2"}]}"#,
                "non-numeric score",
            ),
            (
                r#"{"results":[{"index":0,"relevance_score":0.1},{"index":1}]}"#,
                "no `relevance_score` or `score`",
            ),
            (
                r#"{"results":[{"index":0,"relevance_score":0.1},{"index":1,"relevance_score":1e39}]}"#,
                "non-finite score",
            ),
            (
                r#"{"results":[{"index":0,"relevance_score":0.1},{"index":1,"relevance_score":NaN}]}"#,
                "invalid response body",
            ),
            (r#"{"scores":[0.1,0.2]}"#, "no `results` or `data` array"),
            (r#""ok""#, "neither an object nor an array"),
        ] {
            let error = match parse_scores(body.as_bytes(), 2) {
                Ok(scores) => panic!("{body} parsed as {scores:?}"),
                Err(error) => format!("{error:#}"),
            };
            assert!(error.contains(expected), "{body}: {error}");
        }
    }

    #[test]
    fn logits_outside_the_unit_interval_go_through_the_sigmoid_once() -> Result<()> {
        let logits = parse_scores(
            br#"[{"index":0,"score":-2.0},{"index":1,"score":0.0},{"index":2,"score":3.5}]"#,
            3,
        )?;
        let expected = [-2.0, 0.0, 3.5]
            .into_iter()
            .map(crate::rerank::sigmoid)
            .collect::<Result<Vec<f32>>>()?;
        assert_close(&logits, &expected);
        let probabilities = parse_scores(
            br#"[{"index":0,"score":0.0},{"index":1,"score":1.0},{"index":2,"score":0.25}]"#,
            3,
        )?;
        assert_eq!(probabilities, vec![0.0, 1.0, 0.25]);
        Ok(())
    }

    #[test]
    fn retries_429_and_5xx_but_not_client_errors() -> Result<()> {
        let (base, server) = serve(|n, body| match n {
            0 => reply(429, json!({"error": {"message": "slow down"}})),
            1 => reply(502, json!({"error": "bad gateway"})),
            _ => scored(n, body),
        })?;
        let url = format!("{base}/rerank");
        let scores = client(&url, None, 3)?.rerank("q", &documents(2))?;
        assert_eq!(join(server)?.len(), 3);
        assert_close(&scores, &[0.1, 0.2]);

        for (status, expected) in [
            (400, "rerank server returned HTTP 400: bad input"),
            (
                401,
                "rerank server rejected the request (HTTP 401): bad input",
            ),
        ] {
            let (base, server) =
                serve(move |_, _| reply(status, json!({"error": {"message": "bad input"}})))?;
            let url = format!("{base}/rerank");
            let error = expect_err(client(&url, None, 3)?.rerank("q", &documents(2)))?;
            assert_eq!(join(server)?.len(), 1, "{status}");
            assert!(error.starts_with(expected), "{error}");
        }
        Ok(())
    }

    #[test]
    fn api_key_never_appears_in_errors_or_debug() -> Result<()> {
        let (base, server) = serve(|_, _| {
            reply(
                401,
                json!({"error": {"message": format!("invalid key {KEY} for query secret-query")}}),
            )
        })?;
        let url = format!("{base}/rerank");
        let mut reranker = client(&url, Some(KEY), 0)?;
        let error = expect_err(reranker.rerank("q", &documents(1)))?;
        join(server)?;
        assert!(error.contains("HTTP 401"), "{error}");
        assert!(!error.contains(KEY), "{error}");
        assert!(!format!("{reranker:?}").contains(KEY));
        assert!(!format!("{:?}", config(&url, Some(KEY), RerankDialect::Documents)).contains(KEY));

        let mut unreachable = client("http://127.0.0.1:1/rerank", Some(KEY), 0)?;
        let error = expect_err(unreachable.rerank("q", &documents(1)))?;
        assert!(!error.contains(KEY), "{error}");
        assert!(!error.contains("doc 0"), "{error}");
        Ok(())
    }

    #[test]
    fn rejects_invalid_model_names_and_endpoints() {
        for model in ["", "a b", "a\tb", "a\u{7}b"] {
            for dialect in [RerankDialect::Documents, RerankDialect::Texts] {
                let mut config = config("https://api.example.test/v1/rerank", None, dialect);
                config.model = Some(model.to_owned());
                assert!(RemoteReranker::new(&config).is_err(), "{model:?}");
            }
        }
        let mut unnamed = config(
            "https://api.example.test/v1/rerank",
            None,
            Default::default(),
        );
        unnamed.model = None;
        let error = expect_err(RemoteReranker::new(&unnamed)).unwrap_or_default();
        assert!(error.contains("requires a model name"), "{error}");
        let long = "m".repeat(MAX_MODEL_NAME_CHARS + 1);
        assert!(validate_model_name(&long).is_err());
        assert!(validate_model_name(&long[1..]).is_ok());
        assert!(validate_model_name("cohere/rerank-v3.5").is_ok());
        assert!(
            RemoteReranker::new(&config(
                "http://example.com/rerank",
                Some(KEY),
                Default::default()
            ))
            .is_err()
        );
        assert!(
            RemoteReranker::new(&config(
                "ftp://example.com/rerank",
                None,
                Default::default()
            ))
            .is_err()
        );
    }

    #[test]
    fn not_found_names_the_full_endpoint_url_requirement() -> Result<()> {
        let (base, server) = serve(|_, _| reply(404, json!({"error": "no route"})))?;
        let url = format!("{base}/tok-abc");
        let error = expect_err(client(&url, None, 3)?.rerank("q", &documents(1)))?;
        assert_eq!(join(server)?.len(), 1);
        assert_eq!(
            error,
            "rerank server returned HTTP 404: no route; rerank_url must be the full endpoint \
             URL, such as https://api.cohere.com/v2/rerank"
        );
        Ok(())
    }

    #[test]
    fn retry_after_beyond_the_time_budget_fails_without_waiting() -> Result<()> {
        let (base, server) = serve(|_, _| Reply {
            status: 429,
            body: json!({"error": {"message": "slow down"}}).to_string(),
            retry_after: Some("30"),
        })?;
        let url = format!("{base}/rerank");
        let started = std::time::Instant::now();
        let error = expect_err(client(&url, None, 3)?.rerank("q", &documents(1)))?;
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(join(server)?.len(), 1);
        assert!(error.contains("time budget"), "{error}");
        assert!(
            error.contains("rerank server returned HTTP 429: slow down"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn mixed_and_saturated_logits_all_go_through_the_sigmoid() -> Result<()> {
        let mixed = parse_scores(
            br#"[{"index":0,"score":-1.0},{"index":1,"score":0.5},{"index":2,"score":3.0}]"#,
            3,
        )?;
        let expected = [-1.0, 0.5, 3.0]
            .into_iter()
            .map(crate::rerank::sigmoid)
            .collect::<Result<Vec<f32>>>()?;
        assert_close(&mixed, &expected);
        // Large logits saturate to equal probabilities; search keeps the retrieval
        // order of ties.
        let saturated = parse_scores(
            br#"[{"index":0,"score":40.0},{"index":1,"score":60.0},{"index":2,"score":-200.0}]"#,
            3,
        )?;
        assert_eq!(saturated, vec![1.0, 1.0, 0.0]);
        Ok(())
    }

    #[test]
    fn clamps_retries_and_timeout() -> Result<()> {
        let mut settings = config("http://localhost:8080/rerank", None, Default::default());
        settings.endpoint.max_retries = 100;
        settings.endpoint.timeout = Duration::from_secs(3600);
        let reranker = RemoteReranker::new(&settings)?;
        assert_eq!(reranker.http.endpoint().max_retries, LIMITS.max_retries);
        assert_eq!(reranker.http.endpoint().timeout, LIMITS.max_timeout);
        Ok(())
    }
}
