# Synthetic retrieval regression corpus

These nine cases and eleven records are wholly invented fixtures about a fictional museum, wardrobe and toy collection. They contain no user search tasks, query history or transcript excerpts.

They exercise artifact identifiers, quoted phrases, project/session scopes, query fusion, late snippet evidence, compound tokens and no-answer queries. Baselines preserve current behavior, including failures; they are regression floors, not absolute quality targets.

The CLI uses all nine cases. TUI/web use the seven cases their interfaces support, preserving filters rather than silently dropping them. Per-case comparisons prevent aggregate improvements from concealing regressions.

Record JSONL optionally supplies repo_project for repository grouping. Relevance judgments identify records and expected verbatim evidence spans. Snapshot runs never consult host source files or repositories.
