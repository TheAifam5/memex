//! KiloCode CLI sessions, stored as rows in one SQLite database.
//!
//! KiloCode keeps every session of every project in a single store at
//! `$XDG_DATA_HOME/kilo/kilo.db` (`~/.local/share/kilo/kilo.db` by default).
//! Sessions live in `session` (with `parent_id` linking subagent sessions to
//! their parent), turns in `message` (JSON `data` carrying role, model, and
//! tokens), and content items in `part`, discriminated by `data.type`: `text`,
//! `reasoning`, and `tool` (input and output in one part), plus skippable
//! `step-start`, `step-finish`, `patch`, and `file` markers. Timestamps are
//! millisecond epochs, and rows carry no sequence column, so messages and
//! parts order by `(time_created, id)`.
//!
//! The store grows in place as sessions progress. Virtual source paths and
//! content fingerprints let ingestion replace only changed sessions. Token
//! usage comes from the `tokens` object on each assistant `message`, one row
//! per model request with its own token buckets.

use super::{
    ConversationKind, IndexParseOutput, IndexParseState, ParseDiagnostics, ParserVersions,
    SourceFile, UsageDependency, UsageParseOutput,
};
use crate::types::{Record, RecordLinks, SourceKind};
use crate::usage::{TokenBuckets, UsageEvent};
use anyhow::{Context, Result, anyhow};
use rusqlite::{Connection, OpenFlags};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub const VERSIONS: ParserVersions = ParserVersions {
    identity: 1,
    index: 3,
    usage: 1,
};

pub fn matches_path(path: &str) -> bool {
    split_virtual_path(Path::new(path)).is_some()
        || path.ends_with("/kilo/kilo.db")
        || path.ends_with("\\kilo\\kilo.db")
}

/// Encode session ids into a single safe component, including slashes and dots.
pub fn virtual_path(database: &Path, session_id: &str) -> PathBuf {
    let encoded: String = session_id
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    database.join(format!("kilocode-session-{encoded}"))
}

pub fn split_virtual_path(path: &Path) -> Option<(PathBuf, String)> {
    let encoded = path
        .file_name()?
        .to_str()?
        .strip_prefix("kilocode-session-")?;
    let database = path.parent()?;
    if database.file_name()? != "kilo.db" || !encoded.len().is_multiple_of(2) {
        return None;
    }
    let bytes = encoded
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok())
        .collect::<Option<Vec<_>>>()?;
    Some((database.to_path_buf(), String::from_utf8(bytes).ok()?))
}

/// KiloCode state roots: `$KILO_DATA_DIR` (replacing the default) or
/// `$XDG_DATA_HOME/kilo`. An extra root keeps a synced copy of another
/// machine's store indexable next to the live one.
pub fn roots() -> Vec<PathBuf> {
    roots_for(
        std::env::var_os("KILO_DATA_DIR").as_deref(),
        &default_data_root(),
    )
}

fn default_data_root() -> PathBuf {
    // Kilo uses XDG paths on macOS and Windows too, rather than native data dirs.
    std::env::var_os("XDG_DATA_HOME")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| super::common::home().join(".local/share"))
        .join("kilo")
}

fn roots_for(kilo_data_dir: Option<&std::ffi::OsStr>, default: &Path) -> Vec<PathBuf> {
    let mut roots: Vec<String> = match kilo_data_dir {
        Some(root) => root
            .to_string_lossy()
            .split(',')
            .map(|root| root.trim().to_string())
            .collect(),
        None => Vec::new(),
    };
    if roots.iter().all(|root| root.is_empty()) {
        roots.push(default.to_string_lossy().into_owned());
    }
    roots
        .into_iter()
        .filter(|root| !root.is_empty())
        .map(PathBuf::from)
        .collect()
}

/// The session database under each root, when it exists.
pub fn db_paths() -> Vec<PathBuf> {
    db_paths_for(&roots())
}

fn db_paths_for(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = roots
        .iter()
        .map(|root| root.join("kilo.db"))
        .filter(|path| path.is_file())
        .collect();
    paths.sort();
    paths.dedup();
    paths
}

pub fn discover() -> Vec<SourceFile> {
    discover_from_roots(&roots())
}

pub fn discover_from_roots(roots: &[PathBuf]) -> Vec<SourceFile> {
    db_paths_for(roots)
        .into_iter()
        .map(|path| SourceFile {
            source: SourceKind::Kilocode,
            path,
        })
        .collect()
}

pub fn usage_files() -> Vec<PathBuf> {
    db_paths()
}

/// Directories holding the session databases: the watcher's narrowest KiloCode
/// roots, clear of the rest of the CLI's churn under the data directory.
pub fn db_dirs() -> Vec<PathBuf> {
    db_dirs_for(&roots())
}

fn db_dirs_for(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = roots
        .iter()
        .filter(|root| root.join("kilo.db").is_file())
        .cloned()
        .collect();
    dirs.sort();
    dirs.dedup();
    dirs
}

fn open_readonly(path: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("open KiloCode database {}", path.display()))?;
    // The live store is written while we read; never block it or race it long.
    let _ = conn.busy_timeout(std::time::Duration::from_millis(500));
    Ok(conn)
}

fn table_names(conn: &Connection) -> Result<std::collections::HashSet<String>> {
    let mut statement = conn.prepare("SELECT name FROM sqlite_master WHERE type='table'")?;
    Ok(statement
        .query_map([], |row| row.get::<_, String>(0))?
        .filter_map(Result::ok)
        .collect())
}

pub(crate) struct SessionFingerprint {
    pub id: String,
    pub fingerprint: String,
    /// Number of transcript parts, including compaction boundaries and reasoning.
    /// Skipped accounting/structural parts do not advance the parser offset.
    pub size: u64,
}

fn is_transcript_part(part_type: &str) -> bool {
    matches!(part_type, "text" | "reasoning" | "tool" | "compaction")
}

/// Only fields used by the record projection belong in its content fingerprint.
fn transcript_part_data(part: &Value) -> Option<Value> {
    let part_type = part.get("type").and_then(Value::as_str)?;
    match part_type {
        "text" | "reasoning" => Some(serde_json::json!({
            "type": part_type,
            "text": part.get("text").and_then(Value::as_str),
        })),
        "tool" => {
            let is_error = part.pointer("/state/status").and_then(Value::as_str) == Some("error");
            let output = part
                .pointer("/state/output")
                .and_then(value_to_string)
                .or_else(|| part.pointer("/state/error").and_then(value_to_string))
                .or_else(|| is_error.then(|| "[tool error]".to_string()));
            Some(serde_json::json!({
                "type": part_type,
                "tool": part.get("tool").and_then(Value::as_str).unwrap_or("unknown"),
                "call_id": part.get("callID").and_then(Value::as_str).filter(|id| !id.is_empty()),
                "input": part.pointer("/state/input").and_then(value_to_string),
                "output": output,
                "is_error": is_error,
            }))
        }
        "compaction" => Some(serde_json::json!({"type": part_type})),
        _ => None,
    }
}

/// Hash actual content from one SQLite snapshot; usage-only writes do not
/// invalidate conversation records, and edits need not update a timestamp.
pub(crate) fn enumerate_sessions(database: &Path) -> Result<Vec<SessionFingerprint>> {
    let conn = open_readonly(database)?;
    let transaction = conn.unchecked_transaction()?;
    let mut sessions = transaction.prepare("SELECT id FROM session ORDER BY id")?;
    let ids = sessions
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    drop(sessions);
    let mut metadata_statement = transaction.prepare(
        "SELECT id, parent_id, directory, title, time_created FROM session WHERE id = ?1",
    )?;
    let mut parts_statement = transaction.prepare(
        "SELECT m.id, m.time_created, m.data, p.id, p.time_created, p.data \
         FROM message m JOIN part p ON p.message_id = m.id \
         WHERE m.session_id = ?1 ORDER BY m.time_created, m.id, p.time_created, p.id",
    )?;
    let mut result = Vec::with_capacity(ids.len());
    for id in ids {
        let mut hash = Sha256::new();
        // Keep labels, hierarchy, cwd, and timestamp fallbacks current without
        // replaying for session accounting fields such as cost or time_updated.
        let metadata = metadata_statement.query_row([&id], |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0)?,
                "parent_id": row.get::<_, Option<String>>(1)?,
                "directory": row.get::<_, Option<String>>(2)?,
                "title": row.get::<_, Option<String>>(3)?,
                "time_created": row.get::<_, Option<i64>>(4)?.unwrap_or_default(),
            }))
        })?;
        hash.update(serde_json::to_vec(&metadata)?);
        let mut size = 0;
        let mut rows = parts_statement.query([&id])?;
        while let Some(row) = rows.next()? {
            let message_data = row.get::<_, Option<String>>(2)?.unwrap_or_default();
            let part_data = row.get::<_, Option<String>>(5)?.unwrap_or_default();
            let Ok(message) = serde_json::from_str::<Value>(&message_data) else {
                continue;
            };
            let Ok(part) = serde_json::from_str::<Value>(&part_data) else {
                continue;
            };
            let Some(part) = transcript_part_data(&part) else {
                continue;
            };
            let transcript = serde_json::json!({
                "message_id": row.get::<_, String>(0)?,
                "message_time_created": row.get::<_, Option<i64>>(1)?.unwrap_or_default(),
                "role": message
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("assistant"),
                "cwd": message
                    .pointer("/path/cwd")
                    .and_then(Value::as_str)
                    .filter(|cwd| !cwd.is_empty()),
                "part_id": row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                "part_time_created": row.get::<_, Option<i64>>(4)?.unwrap_or_default(),
                "part": part,
            });
            hash.update([0xff]);
            hash.update(serde_json::to_vec(&transcript)?);
            size += 1;
        }
        result.push(SessionFingerprint {
            id,
            fingerprint: format!("{:x}", hash.finalize()),
            size,
        });
    }
    drop(parts_statement);
    drop(metadata_statement);
    transaction.commit()?;
    Ok(result)
}

struct SessionRow {
    id: String,
    parent_id: Option<String>,
    directory: Option<String>,
    time_created: i64,
}

pub(crate) fn parse_index_records(
    path: &Path,
    state: IndexParseState,
    include_reasoning: bool,
    next_doc_id: &AtomicU64,
    mut emit: impl FnMut(Record) -> Result<()>,
) -> Result<IndexParseOutput> {
    // Each changed session is replaced, so state.offset is intentionally ignored.
    let source_path = path.to_string_lossy().to_string();
    let virtual_session = split_virtual_path(path);
    let database = virtual_session
        .as_ref()
        .map_or(path, |(database, _)| database.as_path());
    let selected_session = virtual_session.as_ref().map(|(_, id)| id.as_str());
    let mut diagnostics = ParseDiagnostics::default();
    let conn = open_readonly(database)?;
    // One snapshot across sessions, messages, and parts: in WAL mode KiloCode
    // commits rows while we read.
    let transaction = conn.unchecked_transaction()?;
    let tables = table_names(&transaction)?;
    if !(tables.contains("session") && tables.contains("message") && tables.contains("part")) {
        return Err(anyhow!(
            "KiloCode database {} has no session/message/part tables",
            path.display()
        ));
    }
    let mut sessions_statement = transaction
        .prepare(
            "SELECT id, parent_id, directory, time_created FROM session \
             WHERE (?1 IS NULL OR id = ?1) ORDER BY time_created, id",
        )
        .with_context(|| format!("query sessions in {}", path.display()))?;
    let sessions = sessions_statement
        .query_map([selected_session], |row| {
            Ok(SessionRow {
                id: row.get(0)?,
                parent_id: row.get(1)?,
                directory: row.get(2)?,
                time_created: row.get::<_, Option<i64>>(3)?.unwrap_or_default(),
            })
        })
        .with_context(|| format!("read sessions in {}", path.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let mut messages_statement = transaction
        .prepare(
            "SELECT m.time_created, m.data, p.id, p.time_created, p.data \
             FROM message m JOIN part p ON p.message_id = m.id \
             WHERE m.session_id = ?1 ORDER BY m.time_created, m.id, p.time_created, p.id",
        )
        .with_context(|| format!("prepare messages in {}", path.display()))?;

    let mut row_count = 0;
    let mut turn_id = state.turn_id;
    let mut session_cwd: Option<String> = None;
    for session in &sessions {
        let directory = session.directory.as_deref().filter(|it| !it.is_empty());
        let project = directory
            .map(Path::new)
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .map(str::to_string);
        if session_cwd.is_none()
            && let Some(directory) = directory
        {
            session_cwd = Some(directory.to_string());
        }
        let rows = messages_statement
            .query_map([session.id.as_str()], |row| {
                Ok((
                    row.get::<_, Option<i64>>(0)?.unwrap_or_default(), // message time_created
                    row.get::<_, Option<String>>(1)?.unwrap_or_default(), // message data
                    row.get::<_, Option<String>>(2)?.unwrap_or_default(), // part id
                    row.get::<_, Option<i64>>(3)?.unwrap_or_default(), // part time_created
                    row.get::<_, Option<String>>(4)?.unwrap_or_default(), // part data
                ))
            })
            .with_context(|| format!("read messages in {}", path.display()))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        // A compaction part closes the prompt that preceded it, so the summary
        // that follows reads as its own conversation until the next user turn.
        let mut compacted = false;
        for (message_time_created, message_data, part_id, part_time_created, part_data) in rows {
            let Ok(message) = serde_json::from_str::<Value>(&message_data) else {
                diagnostics.malformed_json_lines += 1;
                continue;
            };
            let Ok(part) = serde_json::from_str::<Value>(&part_data) else {
                diagnostics.malformed_json_lines += 1;
                continue;
            };
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("assistant");
            let part_type = part.get("type").and_then(Value::as_str).unwrap_or("");
            row_count += u64::from(is_transcript_part(part_type));
            if part_type == "compaction" {
                compacted = true;
                continue;
            }
            // The next prompt opens a fresh turn, so anything the compaction
            // summarised stays in the compaction and the new turn does not.
            if part_type == "text" && role == "user" {
                compacted = false;
            }
            // `path.cwd` is the working directory of the turn that produced the
            // record; sessions that never wrote one keep `session.directory`.
            let cwd = message
                .pointer("/path/cwd")
                .and_then(Value::as_str)
                .filter(|it| !it.is_empty())
                .or(directory)
                .map(str::to_string);
            if session_cwd.is_none()
                && let Some(cwd) = cwd.as_deref()
            {
                session_cwd = Some(cwd.to_string());
            }
            let project = cwd
                .as_deref()
                .map(super::common::project_from_path)
                .or_else(|| project.clone());
            let ts = [
                part_time_created,
                message_time_created,
                session.time_created,
            ]
            .iter()
            .find(|&&candidate| candidate > 0)
            .copied()
            .unwrap_or_default()
            .max(0) as u64;
            let mut links = RecordLinks::default();
            if !part_id.is_empty() {
                links.event_id = Some(part_id.clone());
            }
            if let Some(parent) = session
                .parent_id
                .as_deref()
                .filter(|parent| !parent.is_empty())
            {
                links.parent_session_id = Some(parent.to_string());
                links.conversation_kind = Some(ConversationKind::Subagent.as_str().to_string());
            } else {
                links.conversation_kind = Some(ConversationKind::Main.as_str().to_string());
            }
            if compacted {
                links.conversation_kind = Some(ConversationKind::Compaction.as_str().to_string());
                links.thread_source = Some(ConversationKind::Compaction.as_str().to_string());
            }
            let base = || Record {
                source: SourceKind::Kilocode,
                doc_id: next_doc_id.fetch_add(1, Ordering::SeqCst),
                ts,
                project: project
                    .clone()
                    .unwrap_or_else(|| SourceKind::Kilocode.label().to_string()),
                session_id: session.id.clone(),
                turn_id,
                role: String::new(),
                text: String::new(),
                tool_name: None,
                tool_input: None,
                tool_output: None,
                links: links.clone(),
                source_path: source_path.clone(),
            };
            match part_type {
                "text" => {
                    let Some(text) = part.get("text").and_then(Value::as_str) else {
                        continue;
                    };
                    emit(Record {
                        role: role.to_string(),
                        text: text.to_string(),
                        ..base()
                    })?;
                    turn_id += 1;
                }
                "reasoning" => {
                    if !include_reasoning {
                        continue;
                    }
                    let Some(text) = part.get("text").and_then(Value::as_str) else {
                        continue;
                    };
                    emit(Record {
                        role: "reasoning".to_string(),
                        text: text.to_string(),
                        ..base()
                    })?;
                    turn_id += 1;
                }
                "tool" => {
                    let tool_name = part
                        .get("tool")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                        .to_string();
                    let input = part.pointer("/state/input").and_then(value_to_string);
                    let is_error =
                        part.pointer("/state/status").and_then(Value::as_str) == Some("error");
                    let output = part
                        .pointer("/state/output")
                        .and_then(value_to_string)
                        .or_else(|| part.pointer("/state/error").and_then(value_to_string))
                        .or_else(|| is_error.then(|| "[tool error]".to_string()));
                    let call_id = part
                        .get("callID")
                        .and_then(Value::as_str)
                        .filter(|id| !id.is_empty())
                        .unwrap_or(&part_id)
                        .to_string();
                    let mut tool_links = links.clone();
                    tool_links.event_id = Some(call_id.clone());
                    emit(Record {
                        role: "tool_use".to_string(),
                        text: input.clone().unwrap_or_default(),
                        tool_name: Some(tool_name.clone()),
                        tool_input: input,
                        links: tool_links,
                        ..base()
                    })?;
                    if let Some(output) = output {
                        let mut result_links = links.clone();
                        result_links.event_id = Some(format!("{part_id}:result"));
                        result_links.parent_event_id = Some(call_id.clone());
                        result_links.parent_tool_use_id = Some(call_id);
                        result_links.tool_result_is_error = Some(is_error);
                        emit(Record {
                            role: "tool_result".to_string(),
                            text: output.clone(),
                            tool_name: Some(tool_name),
                            tool_output: Some(output),
                            links: result_links,
                            ..base()
                        })?;
                    }
                    turn_id += 1;
                }
                "step-start" | "step-finish" | "patch" | "file" => {}
                other => {
                    diagnostics.increment_unknown_top_level(&format!("part_type_{other}"));
                }
            }
        }
    }
    drop(messages_statement);
    drop(sessions_statement);
    transaction.commit()?;
    Ok(IndexParseOutput {
        offset: row_count,
        turn_id,
        legacy_turn_id: None,
        pending_tool_calls: state.pending_tool_calls,
        session_id: selected_session.map(str::to_owned),
        diagnostics,
        session_cwd,
    })
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) => Some(text.clone()),
        other => serde_json::to_string(other).ok(),
    }
}

struct MessageUsageRow {
    id: String,
    session_id: String,
    parent_id: Option<String>,
    directory: Option<String>,
    time_created: i64,
    data: String,
}

pub(crate) fn parse_usage_file(path: &Path) -> Result<UsageParseOutput> {
    let mut wal = path.as_os_str().to_os_string();
    wal.push("-wal");
    let wal = PathBuf::from(wal);
    let wal_before = UsageDependency::from_path_or_absent(&wal);
    let conn = open_readonly(path)?;
    let transaction = conn.unchecked_transaction()?;
    let tables = table_names(&transaction)?;
    let source_path: Arc<str> = Arc::from(path.to_string_lossy().to_string());
    let mut events = Vec::new();
    if tables.contains("message") {
        let mut statement = transaction
            .prepare(
                "SELECT m.id, m.session_id, m.time_created, m.data, s.parent_id, s.directory \
                 FROM message m JOIN session s ON s.id = m.session_id \
                 ORDER BY m.time_created, m.id",
            )
            .with_context(|| format!("query message usage in {}", path.display()))?;
        let rows = statement
            .query_map([], |row| {
                Ok(MessageUsageRow {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    time_created: row.get::<_, Option<i64>>(2)?.unwrap_or_default(),
                    data: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    parent_id: row.get(4)?,
                    directory: row.get(5)?,
                })
            })
            .context("read message usage")?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for (order, row) in rows.into_iter().enumerate() {
            let Ok(message) = serde_json::from_str::<Value>(&row.data) else {
                continue;
            };
            // Only the model request that produced an assistant turn reports
            // usage; user turns and aborted requests leave the object absent.
            if message.get("role").and_then(Value::as_str) != Some("assistant") {
                continue;
            }
            let Some(tokens) = message.get("tokens").filter(|value| value.is_object()) else {
                continue;
            };
            let count = |pointer: &str| {
                tokens
                    .pointer(pointer)
                    .and_then(Value::as_i64)
                    .unwrap_or_default()
                    .max(0) as u64
            };
            let mut buckets = TokenBuckets::disjoint(
                count("/input"),
                count("/cache/read"),
                count("/cache/write"),
                count("/output"),
            );
            // Reasoning is a subset of the reported output, not an extra bucket.
            buckets.reasoning = count("/reasoning").min(buckets.output);
            if buckets.additive_total() == 0 {
                continue;
            }
            let string = |pointer: &str| {
                message
                    .pointer(pointer)
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
            };
            events.push(UsageEvent {
                source: "kilocode",
                source_path: source_path.clone(),
                source_record_id: Some(row.id.clone()),
                session_id: Some(row.session_id.clone()),
                request_id: string("/requestID"),
                message_id: Some(row.id),
                timestamp_ms: row.time_created.max(0) as u64,
                project: row.directory.filter(|value| !value.is_empty()),
                provider: string("/providerID"),
                model: string("/modelID"),
                tokens: buckets,
                credits: None,
                token_usage_available: true,
                source_cost_usd: message.get("cost").and_then(Value::as_f64),
                cost_authoritative: false,
                dedupe_confidence: "exact",
                conservative_undercount: false,
                cache_chain_excluded: false,
                permission_review: false,
                sidechain: row
                    .parent_id
                    .as_deref()
                    .is_some_and(|parent| !parent.is_empty()),
                source_order: order as u64,
            });
        }
    }
    // Keep the read on one SQLite snapshot, and never cache a read that raced
    // a WAL commit.
    transaction.commit()?;
    let wal_after = UsageDependency::from_path_or_absent(&wal);
    Ok(UsageParseOutput {
        events,
        cacheable: wal_before == wal_after,
        deps: vec![wal_after],
    })
}

/// Working directory of one session, for analytics and session metadata.
pub fn session_cwd(path: &Path, session_id: &str) -> Option<PathBuf> {
    let virtual_session = split_virtual_path(path);
    let path = virtual_session
        .as_ref()
        .map_or(path, |(database, _)| database.as_path());
    let conn = open_readonly(path).ok()?;
    conn.query_row(
        "SELECT directory FROM session WHERE id = ?1",
        [session_id],
        |row| row.get::<_, Option<String>>(0),
    )
    .ok()
    .flatten()
    .filter(|directory| !directory.is_empty())
    .map(PathBuf::from)
}

/// Stored title of one session, for labels.
pub fn session_title(path: &Path, session_id: &str) -> Option<String> {
    let virtual_session = split_virtual_path(path);
    let path = virtual_session
        .as_ref()
        .map_or(path, |(database, _)| database.as_path());
    let conn = open_readonly(path).ok()?;
    conn.query_row(
        "SELECT title FROM session WHERE id = ?1",
        [session_id],
        |row| row.get::<_, Option<String>>(0),
    )
    .ok()
    .flatten()
    .filter(|title| !title.trim().is_empty())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    /// A database shaped like the real store: sessions, messages, and parts
    /// with `(time_created, id)` ordering instead of a sequence column.
    pub(crate) fn fixture_db(path: &Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT, directory TEXT,
                title TEXT, time_created INTEGER, time_updated INTEGER);
             CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER,
                time_updated INTEGER, data TEXT);
             CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, session_id TEXT,
                time_created INTEGER, time_updated INTEGER, data TEXT);
             CREATE INDEX part_message_id_id_idx ON part (message_id, id);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session VALUES ('ses_main', NULL, '/work/nipponhomes', 'Main', 1000, 9000)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session VALUES ('ses_child', 'ses_main', '/work/nipponhomes', 'Child', 1100, 9500)",
            [],
        )
        .unwrap();
        let messages: &[(&str, &str, i64, serde_json::Value)] = &[
            (
                "msg_u",
                "ses_main",
                1000,
                json!({"role":"user","time":{"created":1000}}),
            ),
            (
                "msg_a",
                "ses_main",
                2000,
                json!({"role":"assistant","modelID":"kilo/space-bunny","providerID":"kilo",
                       "cost":0.5,
                       "tokens":{"total":30,"input":10,"output":20,"reasoning":4,
                                 "cache":{"read":5,"write":2}}}),
            ),
            (
                "msg_c",
                "ses_child",
                3000,
                json!({"role":"assistant","cost":0.0,
                       "tokens":{"input":1,"output":0,"cache":{"read":0,"write":0}}}),
            ),
        ];
        for (id, session, time, data) in messages {
            conn.execute(
                "INSERT INTO message VALUES (?1, ?2, ?3, ?3, ?4)",
                rusqlite::params![id, session, time, data.to_string()],
            )
            .unwrap();
        }
        let parts: &[(&str, &str, &str, i64, serde_json::Value)] = &[
            (
                "prt_t",
                "msg_u",
                "ses_main",
                1000,
                json!({"type":"text","text":"first prompt"}),
            ),
            (
                "prt_s",
                "msg_a",
                "ses_main",
                2000,
                json!({"type":"step-start","snapshot":"abc"}),
            ),
            (
                "prt_r",
                "msg_a",
                "ses_main",
                2100,
                json!({"type":"reasoning","text":"thinking"}),
            ),
            (
                "prt_c",
                "msg_a",
                "ses_main",
                2200,
                json!({"type":"tool","tool":"read","callID":"call-1",
                       "state":{"status":"completed","input":{"path":"a.txt"},
                                "output":"contents"}}),
            ),
            (
                "prt_f",
                "msg_a",
                "ses_main",
                2300,
                json!({"type":"step-finish","reason":"stop","tokens":{"input":1}}),
            ),
            (
                "prt_z",
                "msg_c",
                "ses_child",
                3000,
                json!({"type":"text","text":"child answer"}),
            ),
        ];
        for (id, message, session, time, data) in parts {
            conn.execute(
                "INSERT INTO part VALUES (?1, ?2, ?3, ?4, ?4, ?5)",
                rusqlite::params![id, message, session, time, data.to_string()],
            )
            .unwrap();
        }
    }

    fn parse(path: &Path, include_reasoning: bool) -> (Vec<Record>, IndexParseOutput) {
        let ids = AtomicU64::new(1);
        let mut records = Vec::new();
        let parsed = parse_index_records(
            path,
            IndexParseState::default(),
            include_reasoning,
            &ids,
            |record| {
                records.push(record);
                Ok(())
            },
        )
        .unwrap();
        (records, parsed)
    }

    #[test]
    fn projects_transcript_rows_into_records() {
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("kilo.db");
        fixture_db(&database);
        let (records, parsed) = parse(&database, false);
        assert_eq!(parsed.diagnostics, ParseDiagnostics::default());
        assert_eq!(parsed.offset, 4);
        assert_eq!(parsed.session_cwd.as_deref(), Some("/work/nipponhomes"));
        assert_eq!(records.len(), 4);
        assert_eq!(records[0].role, "user");
        assert_eq!(records[0].text, "first prompt");
        assert_eq!(records[0].project, "nipponhomes");
        assert_eq!(records[0].links.conversation_kind.as_deref(), Some("main"));
        assert_eq!(records[1].role, "tool_use");
        assert_eq!(records[1].tool_name.as_deref(), Some("read"));
        assert_eq!(records[1].links.event_id.as_deref(), Some("call-1"));
        assert_eq!(records[2].role, "tool_result");
        assert_eq!(records[2].tool_output.as_deref(), Some("contents"));
        assert_eq!(records[2].links.parent_event_id.as_deref(), Some("call-1"));
        assert_eq!(
            records[2].links.parent_tool_use_id.as_deref(),
            Some("call-1")
        );
        // The subagent session is a separate logical session under its parent.
        assert_eq!(records[3].session_id, "ses_child");
        assert_eq!(records[3].text, "child answer");
        assert_eq!(
            records[3].links.parent_session_id.as_deref(),
            Some("ses_main")
        );
        assert_eq!(
            records[3].links.conversation_kind.as_deref(),
            Some("subagent")
        );
        let (reasoning, _) = parse(&database, true);
        assert_eq!(reasoning.len(), 5);
        assert_eq!(reasoning[1].role, "reasoning");
        assert_eq!(reasoning[1].text, "thinking");
    }

    #[test]
    fn a_compaction_part_owns_the_records_until_the_next_prompt() {
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("kilo.db");
        fixture_db(&database);
        let conn = Connection::open(&database).unwrap();
        conn.execute_batch(
            "INSERT INTO message VALUES ('msg_s', 'ses_main', 4000, 4000,
                '{\"role\":\"assistant\",\"path\":{\"cwd\":\"/work/other\"}}');
             INSERT INTO part VALUES ('prt_k', 'msg_s', 'ses_main', 4100, 4100,
                '{\"type\":\"compaction\",\"auto\":true}');
             INSERT INTO part VALUES ('prt_s1', 'msg_s', 'ses_main', 4200, 4200,
                '{\"type\":\"text\",\"text\":\"summary of earlier work\"}');
             INSERT INTO message VALUES ('msg_u2', 'ses_main', 5000, 5000,
                '{\"role\":\"user\"}');
             INSERT INTO part VALUES ('prt_u2', 'msg_u2', 'ses_main', 5000, 5000,
                '{\"type\":\"text\",\"text\":\"next question\"}');",
        )
        .unwrap();
        drop(conn);
        let (records, _) = parse(&database, false);
        let summary = records
            .iter()
            .find(|record| record.text == "summary of earlier work")
            .unwrap();
        assert_eq!(
            summary.links.conversation_kind.as_deref(),
            Some("compaction")
        );
        assert_eq!(summary.links.thread_source.as_deref(), Some("compaction"));
        // The turn's own working directory wins over the session's.
        assert_eq!(summary.project, "other");
        let prompt = records
            .iter()
            .find(|record| record.text == "next question")
            .unwrap();
        assert_eq!(prompt.links.conversation_kind.as_deref(), Some("main"));
        assert_eq!(prompt.links.thread_source, None);
        assert_eq!(prompt.project, "nipponhomes");
    }

    #[test]
    fn virtual_paths_round_trip_and_select_one_session() {
        let temp = tempfile::tempdir().unwrap();
        // The raw store only classifies under a KiloCode data directory, so the
        // shape a real install has is what the round-trip has to survive.
        let database = temp.path().join("kilo/kilo.db");
        std::fs::create_dir_all(database.parent().unwrap()).unwrap();
        fixture_db(&database);
        let encoded = virtual_path(&database, "ses/child.with.dots");
        let (owner, id) = split_virtual_path(&encoded).unwrap();
        assert_eq!(owner, database);
        assert_eq!(id, "ses/child.with.dots");
        assert!(matches_path(&encoded.to_string_lossy()));
        assert!(matches_path(&database.to_string_lossy()));
        // A `kilo.db` outside a KiloCode data directory is somebody else's file.
        let stranger = temp.path().join("elsewhere/kilo.db");
        assert!(!matches_path(&stranger.to_string_lossy()));
        let ids = AtomicU64::new(1);
        let mut selected = Vec::new();
        let parsed = parse_index_records(
            &virtual_path(&database, "ses_child"),
            IndexParseState::default(),
            false,
            &ids,
            |record| {
                selected.push(record);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].session_id, "ses_child");
        assert_eq!(parsed.session_id.as_deref(), Some("ses_child"));
        assert_eq!(parsed.offset, 1);
    }

    #[test]
    fn usage_events_carry_disjoint_token_buckets() {
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("kilo.db");
        fixture_db(&database);
        let output = parse_usage_file(&database).unwrap();
        assert_eq!(output.events.len(), 2);
        let main = &output.events[0];
        assert_eq!(main.source, "kilocode");
        assert_eq!(main.session_id.as_deref(), Some("ses_main"));
        assert_eq!(main.model.as_deref(), Some("kilo/space-bunny"));
        assert_eq!(main.provider.as_deref(), Some("kilo"));
        assert_eq!(main.tokens.uncached_input, 10);
        assert_eq!(main.tokens.cache_read, 5);
        assert_eq!(main.tokens.cache_write, 2);
        assert_eq!(main.tokens.output, 20);
        assert_eq!(main.tokens.reasoning, 4);
        assert_eq!(main.tokens.additive_total(), 37);
        assert_eq!(main.timestamp_ms, 2000);
        assert_eq!(main.source_cost_usd, Some(0.5));
        assert!(!main.sidechain);
        let child = &output.events[1];
        assert!(child.sidechain);
        assert_eq!(child.tokens.additive_total(), 1);
    }

    #[test]
    fn discovery_and_fingerprints_track_session_content() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("kilo");
        std::fs::create_dir_all(&root).unwrap();
        let database = root.join("kilo.db");
        assert!(discover_from_roots(std::slice::from_ref(&root)).is_empty());
        fixture_db(&database);
        assert_eq!(discover_from_roots(std::slice::from_ref(&root)).len(), 1);
        assert_eq!(db_dirs_for(std::slice::from_ref(&root)).len(), 1);
        let before = enumerate_sessions(&database).unwrap();
        assert_eq!(before.len(), 2);
        assert_eq!(before[0].id, "ses_child");
        assert_eq!(before[0].size, 1);
        assert_eq!(before[1].size, 3);
        let conn = Connection::open(&database).unwrap();
        conn.execute(
            "INSERT INTO part VALUES ('prt_n', 'msg_u', 'ses_main', 1500, 1500, ?1)",
            [json!({"type":"text","text":"appended"}).to_string()],
        )
        .unwrap();
        drop(conn);
        let after = enumerate_sessions(&database).unwrap();
        assert_ne!(before[1].fingerprint, after[1].fingerprint);
        assert_eq!(before[0].fingerprint, after[0].fingerprint);
        assert_eq!(after[1].size, 4);
    }

    #[test]
    fn data_root_prefers_the_configured_directory() {
        let default = PathBuf::from("/data/kilo");
        assert_eq!(roots_for(None, &default), vec![PathBuf::from("/data/kilo")]);
        assert_eq!(
            roots_for(Some(std::ffi::OsStr::new("/a,/b")), &default),
            vec![PathBuf::from("/a"), PathBuf::from("/b")]
        );
        assert_eq!(
            roots_for(Some(std::ffi::OsStr::new("")), &default),
            vec![PathBuf::from("/data/kilo")]
        );
    }

    #[test]
    fn default_data_root_uses_xdg_on_every_platform() {
        let _guard = crate::test_support::env_lock();
        let temp = tempfile::tempdir().unwrap();
        let data_root = temp.path().join("xdg-data");
        let fallback = super::super::common::home().join(".local/share/kilo");
        for (configured, expected) in [
            (None, fallback.clone()),
            (Some(std::ffi::OsStr::new("")), fallback),
            (Some(data_root.as_os_str()), data_root.join("kilo")),
        ] {
            let _env = crate::test_support::EnvVarGuard::set_os(&[("XDG_DATA_HOME", configured)]);
            assert_eq!(default_data_root(), expected);
        }
    }

    #[test]
    fn fingerprints_ignore_accounting_and_skipped_parts() {
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("kilo.db");
        fixture_db(&database);
        let before = enumerate_sessions(&database).unwrap();
        let (records_before, parsed_before) = parse(&database, true);
        let conn = Connection::open(&database).unwrap();
        conn.execute_batch(
            "ALTER TABLE session ADD COLUMN cost REAL;
             UPDATE session SET cost = 4.2, time_updated = 10000 WHERE id = 'ses_main';
             UPDATE message SET data = json_set(data, '$.tokens.input', 999, '$.cost', 4.2,
                '$.time.completed', 10000, '$.finish', 'stop'), time_updated = 10000
                WHERE id = 'msg_a';
             UPDATE part SET data = json_set(data, '$.tokens.input', 999),
                time_updated = 10000 WHERE id = 'prt_f';
             INSERT INTO part VALUES ('prt_accounting', 'msg_a', 'ses_main', 10000, 10000,
                '{\"type\":\"step-finish\",\"tokens\":{\"input\":999}}');
             INSERT INTO part VALUES ('prt_patch', 'msg_a', 'ses_main', 10001, 10001,
                '{\"type\":\"patch\",\"hash\":\"edited\"}');
             INSERT INTO part VALUES ('prt_file', 'msg_a', 'ses_main', 10002, 10002,
                '{\"type\":\"file\",\"url\":\"file:///work/example\"}');
             INSERT INTO part VALUES ('prt_start', 'msg_a', 'ses_main', 10003, 10003,
                '{\"type\":\"step-start\",\"snapshot\":\"new\"}');",
        )
        .unwrap();
        let after = enumerate_sessions(&database).unwrap();
        for (before, after) in before.iter().zip(&after) {
            assert_eq!(before.id, after.id);
            assert_eq!(before.fingerprint, after.fingerprint);
            assert_eq!(before.size, after.size);
        }
        let (records_after, parsed_after) = parse(&database, true);
        assert_eq!(parsed_before.offset, parsed_after.offset);
        assert_eq!(
            serde_json::to_value(records_before).unwrap(),
            serde_json::to_value(records_after).unwrap(),
        );
        let usage = parse_usage_file(&database).unwrap();
        assert_eq!(usage.events[0].tokens.uncached_input, 999);
        assert_eq!(usage.events[0].source_cost_usd, Some(4.2));

        // Tool output is transcript content even when the part count is unchanged.
        conn.execute(
            "UPDATE part SET data = json_set(data, '$.state.output', 'new contents') WHERE id = 'prt_c'",
            [],
        )
        .unwrap();
        let changed = enumerate_sessions(&database).unwrap();
        assert_ne!(after[1].fingerprint, changed[1].fingerprint);
        assert_eq!(after[1].size, changed[1].size);
        assert_eq!(after[0].fingerprint, changed[0].fingerprint);

        // Removing non-content rows must also leave the fingerprint unchanged.
        conn.execute_batch(
            "DELETE FROM part WHERE id IN ('prt_accounting', 'prt_patch', 'prt_file', 'prt_start');",
        )
        .unwrap();
        let removed = enumerate_sessions(&database).unwrap();
        assert_eq!(changed[1].fingerprint, removed[1].fingerprint);
        assert_eq!(changed[1].size, removed[1].size);
    }

    #[test]
    fn fingerprints_track_session_and_message_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("kilo.db");
        fixture_db(&database);
        let conn = Connection::open(&database).unwrap();
        for mutation in [
            "UPDATE session SET title = 'Renamed' WHERE id = 'ses_main'",
            "UPDATE session SET directory = '/work/other' WHERE id = 'ses_main'",
            "UPDATE session SET parent_id = 'ses_parent' WHERE id = 'ses_main'",
            "UPDATE session SET time_created = 900 WHERE id = 'ses_main'",
            "UPDATE message SET data = json_set(data, '$.path.cwd', '/work/turn') WHERE id = 'msg_a'",
            "UPDATE message SET data = json_set(data, '$.role', 'user') WHERE id = 'msg_a'",
        ] {
            let before = enumerate_sessions(&database).unwrap();
            conn.execute(mutation, []).unwrap();
            let after = enumerate_sessions(&database).unwrap();
            assert_ne!(before[1].fingerprint, after[1].fingerprint, "{mutation}");
            assert_eq!(before[1].size, after[1].size);
            assert_eq!(before[0].fingerprint, after[0].fingerprint);
        }
    }

    #[test]
    fn session_metadata_reads_titles_and_directories() {
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("kilo.db");
        fixture_db(&database);
        assert_eq!(
            session_cwd(&database, "ses_main").as_deref(),
            Some(Path::new("/work/nipponhomes"))
        );
        assert_eq!(
            session_title(&database, "ses_child").as_deref(),
            Some("Child")
        );
        assert!(session_title(&database, "missing").is_none());
    }

    #[test]
    fn unreadable_databases_and_malformed_rows_are_reported() {
        let temp = tempfile::tempdir().unwrap();
        let database = temp.path().join("kilo.db");
        Connection::open(&database)
            .unwrap()
            .execute_batch("CREATE TABLE unrelated (id TEXT)")
            .unwrap();
        let ids = AtomicU64::new(1);
        let error = parse_index_records(&database, IndexParseState::default(), false, &ids, |_| {
            Ok(())
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("no session/message/part tables"), "{error}");
        std::fs::remove_file(&database).unwrap();
        fixture_db(&database);
        let conn = Connection::open(&database).unwrap();
        conn.execute(
            "INSERT INTO part VALUES ('prt_bad', 'msg_u', 'ses_main', 1700, 1700, 'not json')",
            [],
        )
        .unwrap();
        drop(conn);
        let (records, parsed) = parse(&database, false);
        assert_eq!(parsed.diagnostics.malformed_json_lines, 1);
        assert_eq!(records.len(), 4);
    }
}
