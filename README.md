# GroundCheck

Reproduction code and recorded measurements for an audit of signal grounding in language-model answers to wearable IMU questions.

## Setup

Use Python 3.9 or 3.10. The pinned environment was checked with Python 3.9.6 on Apple Silicon. The recorded-result analysis runs on CPU. Local inference uses Apple MPS when available and otherwise CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_artifact.py
```

## Reproduce from the recorded runs

```bash
bash scripts/run_all.sh --check
bash scripts/run_all.sh --analysis
```

The first command checks data integrity and runs the numerical reference, scoring and statistics checks. The second recomputes the DSP reference rows, the original six-model aggregation, the expanded analysis and the code-behaviour analysis. Outputs are written to `results/metrics.json`, `results/RESULTS.md`, `results/rev2611/analysis.json`, `results/rev2611/code_behaviour.json` and `figures/`.

The released benchmark contains **1,036 items and 2,254 counterfactual variants** across UCI-HAR, WISDM v1.1 and PAMAP2. Both original and corrected body-location questions are available. The fixed evaluation subsets and seed are defined in `src/evalplan.py`; the benchmark-generation settings are in `data/benchmark/config.json`.

## Regenerate the benchmark

```bash
bash scripts/download_data.sh
python scripts/generate_benchmark.py --seed 0
python scripts/rev2611_gt_validation.py
python scripts/rev2611_fix_wording.py
```

The downloader acquires all three source datasets, verifies the recorded SHA-256 hashes, and extracts nested archives. Download URLs and hashes are in `data/source_archives.json`. Allow several GB of disk space for extraction. The supplied serialized items are sufficient for the recorded-run analysis and fresh inference; source recordings are needed for benchmark regeneration and the full-rate reference validation.

## Run local models

The runners resume by skipping stored item/condition keys. Use a fresh working copy with empty `results/raw/` or `results/raw_rev2611/` to collect independent runs, retaining the existing folders separately for comparison.

```bash
python scripts/run_grid.py --model Qwen/Qwen2.5-0.5B-Instruct \
  --tier A --dtype float32 --sections grid contam cf format decode
```

The original model IDs are `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-2-9b-it`. Tiers A/B/C choose nested subsets; use the per-row recorded tier, dtype, token budgets and condition coverage when matching a specific run. Access-controlled checkpoints require access through the model host.

Expanded runs use a local [Ollama](https://ollama.com/) server. Server/model settings are retained in `results/raw_rev2611/config_*.json`; these configuration files contain one JSON object per launch, one object per line. Actual per-query records carry the measured settings and coverage.

```bash
ollama pull qwen3:8b
GROUNDCHECK_OLLAMA=http://127.0.0.1:11434 \
python scripts/rev2611_run_new_models.py --model qwen3:8b \
  --tag qwen3-8b-thinkFalse --think false --tier B --sections grid cf
```

Other retained model names include `gemma4:26b-a4b`, `gemma4:31b`, `mistral-small3.1:24b`, `gpt-oss:120b` and `llama3.1:8b-instruct-q4_K_M`. Select the recorded reasoning mode, context size and token limits. Resource requirements vary with model size. The higher reasoning-effort runs have limited coverage; the analysis reports their actual sample counts.

Use `--benchmark data/benchmark_rev2611/items.jsonl --ids data/benchmark_rev2611/changed_ids.json` for the corrected-wording checks. `rev2611_date_check.py` and `rev2611_version_check.py` implement the date and API-compatibility controls. `src/test_stability_probe.py` is an optional inference-dependent check of the recorded MPS precision failure and requires its local checkpoint.

## Contents and interpretation

| Path | Contents |
| --- | --- |
| `src/` | Signal reference algorithms, serializers, DSP tools, model adapters, scoring and statistics |
| `scripts/` | Acquisition, benchmark construction, experiment runners and analyses |
| `data/benchmark/` | Original serialized benchmark and generation settings |
| `data/benchmark_rev2611/` | Corrected questions and declared item subsets |
| `results/raw/` | Original per-query responses, references and run configuration |
| `results/raw_rev2611/` | Expanded model responses, bridge runs and configuration history |
| `results/rev2611/` | Full-rate validation rows and derived expanded analyses |
| `results/diagnostics/` | The retained numerical-instability negative control |
| `MANIFEST.sha256` | Integrity checks for the release files |

Signal reference accuracy is a measurement ceiling against the chosen algorithmic answers, rather than independently annotated human movement counts. Fresh model responses and timings can vary with hardware, inference versions, precision and load. Statistical outputs can be regenerated from the retained responses without downloading model weights. Files under `results/raw_rev2611/superseded/` are provenance records and are excluded by the active analysis globs.

Dataset sources: [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), [WISDM v1.1](https://www.cis.fordham.edu/wisdm/dataset.php) and [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring). Their original terms apply to the recordings and derived inputs.
