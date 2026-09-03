# deepswe

Daily mirror of the [DeepSWE](https://deepswe.datacurve.ai/) live leaderboard with TrueIQ benchmarking analysis.

## TrueIQ Benchmarking

TrueIQ is a deterministic small-sample benchmarking methodology for evaluating AI model intelligence on coding tasks. Unlike traditional metrics that can be gamed through mechanical repetition (benchmax), TrueIQ:

- **Measures true capability** from observed performance across reasoning effort levels
- **Detects and penalizes spam behavior** — persistent fast mechanical interaction patterns that inflate scores artificially
- **Handles incomplete data** — scores single-effort and multi-effort models fairly without hallucinating missing data points
- **Uses specification curves** — reports the median across a family of reasonable formulas rather than one hand-tuned coefficient set
- **Identifies model phenotypes** along orthogonal axes:
  - **Interaction style**: SURGICAL, CLEAN, COMPUTE-HEAVY, CHURNING, SPAMMER
  - **Effort response**: EFFORT-HUNGRY, STEADY-SCALER, SATURATING, THRESHOLDED, FLAT
  - **Top-end efficiency**: SCALING, EFFICIENT, DIMINISHING, OVERTHINKING
  - **Evidence quality**: STABLE, SUPPORTED, VARIABLE, PARTIAL-CURVE, SINGLE-SETTING

The methodology is designed for small-N datasets (20-30 models, 50-100 runs) where traditional ML approaches overfit.

## Data

| Path | Description |
|------|-------------|
| [`data/leaderboard-live.json`](data/leaderboard-live.json) | Full leaderboard payload |
| [`data/version.txt`](data/version.txt) | Benchmark version (e.g. `v1.1`) |
| [`data/meta.json`](data/meta.json) | Version, update time, row count |
| [`data/trueiq_frontend.json`](data/trueiq_frontend.json) | TrueIQ scores and model phenotype classifications |

Raw URLs:

```
https://raw.githubusercontent.com/benchget/deepswe/main/data/leaderboard-live.json
https://raw.githubusercontent.com/benchget/deepswe/main/data/version.txt
https://raw.githubusercontent.com/benchget/deepswe/main/data/meta.json
https://raw.githubusercontent.com/benchget/deepswe/main/data/trueiq_frontend.json
```

## Client

```bash
# Read mirror metadata
python fetch.py

# Query live DeepSWE directly
python fetch.py --live

# Update local mirror files
python fetch.py --update

# Calculate TrueIQ scores
python trueiq.py --input data/leaderboard-live.json --outdir data
```

Optional env:

- `DEEPSWE_REPO` (default `benchget/deepswe`)
- `DEEPSWE_BRANCH` (default `main`)

## Update

GitHub Actions runs daily at 06:00 UTC (`workflow_dispatch` supported).

The workflow:
1. Fetches the latest leaderboard data from DeepSWE
2. Runs TrueIQ benchmarking analysis on all models
3. Commits updated data and analysis results to the repository

Version is resolved dynamically from `https://deepswe.datacurve.ai/` via URL/artifact detection with fallback probes over:

`v1.1` → `v1` → `v1.2` → `v2` → `v1.0`

