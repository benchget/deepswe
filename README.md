# deepswe

Daily mirror of the [DeepSWE](https://deepswe.datacurve.ai/) live leaderboard.

## Data

| Path | Description |
|------|-------------|
| [`data/leaderboard-live.json`](data/leaderboard-live.json) | Full leaderboard payload |
| [`data/version.txt`](data/version.txt) | Benchmark version (e.g. `v1.1`) |
| [`data/meta.json`](data/meta.json) | Version, update time, row count |

Raw URLs:

```
https://raw.githubusercontent.com/benchget/deepswe/main/data/leaderboard-live.json
https://raw.githubusercontent.com/benchget/deepswe/main/data/version.txt
https://raw.githubusercontent.com/benchget/deepswe/main/data/meta.json
```

## Client

```bash
python fetch.py
```

Optional env:

- `DEEPSWE_REPO` (default `benchget/deepswe`)
- `DEEPSWE_BRANCH` (default `main`)

## Update

GitHub Actions runs daily at 06:00 UTC (`workflow_dispatch` supported).

Version is resolved with sequential `HEAD` requests over:

`v1.1` → `v1` → `v1.2` → `v2` → `v1.0`

First `200` wins. Prepend new versions to the candidate list in the workflow when needed.
