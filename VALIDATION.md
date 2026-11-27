# Validation

Checked on 27 November 2026 with the interpreter and dependencies in `environment.json`.

- Release integrity and both 1,036-item benchmark files verified.
- Signal reference tests: 12/12 passed.
- Statistics tests: 16/16 passed.
- Counterfactual scoring tests: 18/18 passed.
- `bash scripts/run_all.sh --analysis` completed in an isolated copy, including DSP recomputation and plots.
- Expanded `analysis.json` and `code_behaviour.json` exactly matched the recorded JSON values.
- Original `metrics.json` matched except for its generation timestamp.

Fresh model inference, source-recording downloads and full-rate regeneration were not executed during this packaging check. All original recorded responses remain intact in this release.
