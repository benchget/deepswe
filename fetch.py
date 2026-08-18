from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

REPO = os.environ.get("DEEPSWE_REPO", "benchget/deepswe")
BRANCH = os.environ.get("DEEPSWE_BRANCH", "main")
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data"


def _get(path: str) -> bytes:
    req = urllib.request.Request(
        f"{RAW}/{path}",
        headers={"User-Agent": "deepswe/1.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        return response.read()


def version() -> str:
    return _get("version.txt").decode().strip()


def leaderboard() -> dict[str, Any]:
    return json.loads(_get("leaderboard-live.json"))


def meta() -> dict[str, Any]:
    return json.loads(_get("meta.json"))


if __name__ == "__main__":
    import time

    started = time.perf_counter()
    info = meta()
    print(f"version    = {info['version']}")
    print(f"updated_at = {info['updated_at']}")
    print(f"n_rows     = {info['n_rows']}")
    print(f"source     = {info['source']}")
    print(f"took       = {(time.perf_counter() - started) * 1000:.0f} ms")
