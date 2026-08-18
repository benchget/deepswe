from __future__ import annotations

import datetime
import gzip
import json
import os
import re
import sys
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Any

BASE_URL = "https://deepswe.datacurve.ai"
FALLBACK_VERSIONS = ("v1.1", "v1", "v1.2", "v2", "v1.0")

REPO = os.environ.get("DEEPSWE_REPO", "benchget/deepswe")
BRANCH = os.environ.get("DEEPSWE_BRANCH", "main")
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/data"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "text/html,application/xhtml+xml,application/json,*/*",
    "Accept-Encoding": "gzip, deflate",
}

RE_ARTIFACT_VER = re.compile(r"/artifacts/(v[\d.]+)/leaderboard-live\.json")
RE_DATA_VER = re.compile(
    r"href=[\x22\x27](?:https?://[^\x22\x27]*)?/data/(v[\d.]+)[\x22\x27]"
)


def _http_get(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> bytes:
    req_headers = dict(DEFAULT_HEADERS)
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        encoding = resp.info().get("Content-Encoding", "").lower()
        if encoding == "gzip":
            return gzip.decompress(raw)
        if encoding == "deflate":
            try:
                return zlib.decompress(raw)
            except zlib.error:
                return zlib.decompress(raw, -zlib.MAX_WBITS)
        return raw


def resolve_version(
    base_url: str = BASE_URL,
    timeout: float = 10.0,
) -> str:
    """Resolve DeepSWE version via HTML URL/re parsing with fallback probes."""
    try:
        html_bytes = _http_get(f"{base_url}/", timeout=timeout)
        text = html_bytes.decode("utf-8", errors="ignore")

        # 1. Look for artifact endpoint in page chunks/state
        m = RE_ARTIFACT_VER.search(text)
        if m:
            return m.group(1)

        # 2. Look for navigation data link (e.g. /data/v1.1)
        m = RE_DATA_VER.search(text)
        if m:
            return m.group(1)
    except Exception:
        pass

    # 3. Fallback: probe candidates with HEAD requests
    for ver in FALLBACK_VERSIONS:
        test_url = f"{base_url}/artifacts/{ver}/leaderboard-live.json"
        req = urllib.request.Request(
            test_url,
            method="HEAD",
            headers={"User-Agent": "deepswe/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    return ver
        except Exception:
            continue

    raise RuntimeError("Failed to resolve DeepSWE version")


def leaderboard_url(
    version: str | None = None,
    base_url: str = BASE_URL,
) -> str:
    """Return the direct live artifact URL for leaderboard-live.json."""
    ver = version or resolve_version(base_url=base_url)
    return f"{base_url}/artifacts/{ver}/leaderboard-live.json"


def fetch_live_leaderboard(
    version: str | None = None,
    base_url: str = BASE_URL,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Fetch the live leaderboard JSON payload directly from DeepSWE."""
    url = leaderboard_url(version=version, base_url=base_url)
    raw = _http_get(url, timeout=timeout)
    return json.loads(raw.decode("utf-8"))


def update_mirror(
    data_dir: str | Path = "data",
    base_url: str = BASE_URL,
) -> dict[str, Any]:
    """Fetch live data and update local mirror files (version, payload, meta)."""
    target_dir = Path(data_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    ver = resolve_version(base_url=base_url)
    url = leaderboard_url(version=ver, base_url=base_url)
    raw_payload = _http_get(url, timeout=60.0)
    payload = json.loads(raw_payload.decode("utf-8"))

    (target_dir / "leaderboard-live.json").write_bytes(raw_payload)
    (target_dir / "version.txt").write_text(f"{ver}\n", encoding="utf-8")

    meta_info: dict[str, Any] = {
        "version": ver,
        "updated_at": datetime.datetime.now(datetime.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": url,
        "generated_at": payload.get("generated_at"),
        "n_tasks_in_set": payload.get("n_tasks_in_set"),
        "n_rows": len(payload.get("rows") or []),
    }

    (target_dir / "meta.json").write_text(
        json.dumps(meta_info, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta_info


def _get(path: str) -> bytes:
    return _http_get(f"{RAW}/{path}", headers={"User-Agent": "deepswe/1.0"})


def version() -> str:
    """Read version from the GitHub mirror repository."""
    return _get("version.txt").decode("utf-8").strip()


def leaderboard() -> dict[str, Any]:
    """Read leaderboard payload from the GitHub mirror repository."""
    return json.loads(_get("leaderboard-live.json").decode("utf-8"))


def meta() -> dict[str, Any]:
    """Read metadata from the GitHub mirror repository."""
    return json.loads(_get("meta.json").decode("utf-8"))


if __name__ == "__main__":
    started = time.perf_counter()
    if "--update" in sys.argv:
        info = update_mirror()
        print(json.dumps(info, indent=2))
    elif "--live" in sys.argv:
        ver = resolve_version()
        data = fetch_live_leaderboard(version=ver)
        print(f"live version = {ver}")
        print(f"n_rows       = {len(data.get('rows') or [])}")
        print(f"generated_at = {data.get('generated_at')}")
        print(f"took         = {(time.perf_counter() - started) * 1000:.0f} ms")
    else:
        try:
            info = meta()
            print(f"version    = {info['version']}")
            print(f"updated_at = {info['updated_at']}")
            print(f"n_rows     = {info['n_rows']}")
            print(f"source     = {info['source']}")
            print(f"took       = {(time.perf_counter() - started) * 1000:.0f} ms")
        except Exception:
            # Fallback to live status if mirror is not yet populated
            ver = resolve_version()
            print(f"live version = {ver}")
            print(f"source       = {leaderboard_url(version=ver)}")
            print(f"took         = {(time.perf_counter() - started) * 1000:.0f} ms")
