use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn corpus(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/retrieval-quality")
        .join(name)
}

fn evaluate(dataset: &Path, surface: &str, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_memex"))
        .args(["--no-update-check", "debug", "eval-retrieval"])
        .arg(dataset)
        .arg("--records")
        .arg(corpus("records.jsonl"))
        .args(["--surface", surface])
        .args(extra)
        .output()
        .unwrap()
}

fn report(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn lexical_quality_matches_versioned_baselines_for_actual_surfaces() {
    let temp = tempfile::tempdir().unwrap();
    // UI surfaces do not expose multi-query or session-scoped search. Preserve original
    // case bytes for hashes and omit those cases rather than silently dropping filters.
    let ui = std::fs::read_to_string(corpus("cases.jsonl"))
        .unwrap()
        .lines()
        .filter(|line| {
            let case: Value = serde_json::from_str(line).unwrap();
            case.get("queries").is_none() && case["filters"]["session"].is_null()
        })
        .map(|line| format!("{line}\n"))
        .collect::<String>();
    let ui_dataset = temp.path().join("ui.jsonl");
    std::fs::write(&ui_dataset, ui).unwrap();
    for surface in ["cli", "tui", "web"] {
        let baseline = corpus(&format!("baseline-{surface}.json"));
        let dataset = if surface == "cli" {
            corpus("cases.jsonl")
        } else {
            ui_dataset.clone()
        };
        let output = evaluate(
            &dataset,
            surface,
            &["--baseline", baseline.to_str().unwrap()],
        );
        let result = report(&output);
        assert!(result["regressions"].as_array().unwrap().is_empty());
        assert_eq!(
            result["per_case"].as_array().unwrap().len(),
            if surface == "cli" { 9 } else { 7 }
        );
    }
}

#[test]
fn baseline_fails_on_an_individual_regression_even_if_other_cases_improve() {
    let mut baseline: Value =
        serde_json::from_slice(&std::fs::read(corpus("baseline-cli.json")).unwrap()).unwrap();
    let cases = baseline["per_case"].as_array_mut().unwrap();
    let case = cases
        .iter_mut()
        .find(|case| case["id"] == "compound-identifier")
        .unwrap();
    case["metrics"]["mrr_at_k"] = Value::from(1.0);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("better.json");
    std::fs::write(&path, serde_json::to_vec(&baseline).unwrap()).unwrap();
    let output = evaluate(
        &corpus("cases.jsonl"),
        "cli",
        &["--baseline", path.to_str().unwrap()],
    );
    assert!(!output.status.success());
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(
        result["regressions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|metric| metric["id"] == "compound-identifier" && metric["metric"] == "mrr_at_k")
    );
}

#[test]
fn baseline_rejects_a_report_from_a_changing_index() {
    let mut baseline: Value =
        serde_json::from_slice(&std::fs::read(corpus("baseline-cli.json")).unwrap()).unwrap();
    baseline["index_changed_during_run"] = Value::from(true);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("unstable.json");
    std::fs::write(&path, serde_json::to_vec(&baseline).unwrap()).unwrap();
    let output = evaluate(
        &corpus("cases.jsonl"),
        "cli",
        &["--baseline", path.to_str().unwrap()],
    );
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("stable index throughout"));
}

#[test]
fn evaluation_rejects_missing_judgments_and_silent_mode_or_scope_changes() {
    let temp = tempfile::tempdir().unwrap();
    let mut case: Value = serde_json::from_str(
        std::fs::read_to_string(corpus("cases.jsonl"))
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    case["relevant"][0]["doc_id"] = Value::from(99999);
    let dataset = temp.path().join("missing.jsonl");
    std::fs::write(&dataset, serde_json::to_vec(&case).unwrap()).unwrap();
    let missing = evaluate(&dataset, "cli", &[]);
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("judged record missing"));
    case["relevant"][0]["doc_id"] = Value::from(1);
    case["filters"] = serde_json::json!({"tool":"Bash"});
    std::fs::write(&dataset, serde_json::to_vec(&case).unwrap()).unwrap();
    let unknown = evaluate(&dataset, "cli", &[]);
    assert!(!unknown.status.success());
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("unknown field"));
    case["filters"] = serde_json::json!({});
    case["cwd"] = Value::from("/fixture/checkout");
    std::fs::write(&dataset, serde_json::to_vec(&case).unwrap()).unwrap();
    let scoped = evaluate(&dataset, "cli", &[]);
    assert!(!scoped.status.success());
    assert!(String::from_utf8_lossy(&scoped.stderr).contains("cannot replay checkout metadata"));
    for (surface, args, message) in [
        ("cli", vec!["--mode", "semantic"], "prepared vector index"),
        (
            "web",
            vec![],
            "single query and project/source filters only",
        ),
        (
            "tui",
            vec![],
            "single query and project/source filters only",
        ),
    ] {
        let output = evaluate(&corpus("cases.jsonl"), surface, &args);
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(message),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
