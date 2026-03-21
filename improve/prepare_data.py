#!/usr/bin/env python3
"""
Part E -- Step 1: Download and prepare the HellaSwag dataset.

Downloads the HellaSwag validation and training splits from HuggingFace
(via the REST API -- no `datasets` library required) and saves processed
examples to improve/data/ as JSON files.

Each processed example contains:
    - ind:          original dataset index
    - activity:     activity label
    - ctx:          context (ctx_a + " " + ctx_b)
    - endings:      list of 4 candidate endings
    - label:        integer gold label (0-3)
    - split:        "validation" or "train"

Usage:
    python improve/prepare_data.py [--n-val 200] [--n-train 500]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
    from requests.adapters import HTTPAdapter
except ImportError:
    sys.exit("ERROR: 'requests' is required.  pip install requests")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

HF_API = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "Rowan/hellaswag"

# Fallback: direct parquet download via HF Hub
HF_PARQUET_BASE = (
    "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data"
)

# Persistent session with connection pooling
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _fetch_rows_api(split: str, length: int, offset: int = 0) -> list[dict]:
    """Fetch rows from the HuggingFace datasets server REST API."""
    params = {
        "dataset": HF_DATASET,
        "config": "default",
        "split": split,
        "offset": offset,
        "length": min(length, 100),  # API caps at 100 per request
    }
    all_rows: list[dict] = []
    fetched = 0
    session = _get_session()

    while fetched < length:
        params["offset"] = offset + fetched
        params["length"] = min(100, length - fetched)

        resp = session.get(HF_API, params=params, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(
                f"HF API returned {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        rows = data.get("rows", [])
        if not rows:
            break
        all_rows.extend(r["row"] for r in rows)
        fetched += len(rows)
        time.sleep(0.2)  # polite rate-limiting

    return all_rows


def _fetch_split(split: str, n: int) -> list[dict]:
    """Download *n* examples from a HellaSwag split.

    Tries the HuggingFace datasets-server API first; if that fails, falls
    back to downloading the raw JSON-lines file from the HF Hub.
    """
    print(f"[prepare] Fetching {n} examples from '{split}' split ...")

    try:
        rows = _fetch_rows_api(split, n)
        if rows:
            print(f"[prepare]   -> got {len(rows)} rows from datasets-server.")
            return rows[:n]
    except Exception as exc:
        print(f"[prepare]   datasets-server failed ({exc}), trying fallback...")

    # Fallback: download raw jsonl from HF Hub
    session = _get_session()
    url = f"https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/hellaswag_{split}.jsonl"
    resp = session.get(url, timeout=120)
    if resp.status_code != 200:
        # Try another common pattern
        url = f"https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/{split}.jsonl"
        resp = session.get(url, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Could not fetch HellaSwag {split} split (status {resp.status_code})."
        )

    rows = []
    for line in resp.text.strip().split("\n"):
        if line.strip():
            rows.append(json.loads(line))
            if len(rows) >= n:
                break

    print(f"[prepare]   -> got {len(rows)} rows from JSONL fallback.")
    return rows[:n]


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _process_row(row: dict, split: str) -> dict:
    """Normalise a raw HellaSwag row into our standard format."""
    # The HuggingFace dataset stores the label as a string or int
    label = row.get("label", "")
    if isinstance(label, str) and label.isdigit():
        label = int(label)
    elif isinstance(label, str) and label == "":
        label = -1  # unlabelled (test split)
    else:
        label = int(label)

    # Context: concat ctx_a and ctx_b (ctx_b is the partial sentence start)
    ctx_a = row.get("ctx_a", row.get("ctx", ""))
    ctx_b = row.get("ctx_b", "")
    ctx = (ctx_a.strip() + " " + ctx_b.strip()).strip()

    # Endings -- sometimes stored as a JSON string, sometimes a list
    endings = row.get("endings", [])
    if isinstance(endings, str):
        endings = json.loads(endings)

    return {
        "ind": row.get("ind", row.get("id", None)),
        "activity": row.get("activity_label", row.get("activity", "")),
        "ctx": ctx,
        "endings": endings,
        "label": label,
        "split": split,
    }


def prepare(n_val: int = 200, n_train: int = 500) -> tuple[Path, Path]:
    """Download, process, and save HellaSwag data.

    Returns paths to the saved validation and training JSON files.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -- Validation split (used for evaluation) --
    val_raw = _fetch_split("validation", n_val)
    val_processed = [_process_row(r, "validation") for r in val_raw]
    val_path = DATA_DIR / "hellaswag_val.json"
    with open(val_path, "w") as f:
        json.dump(val_processed, f, indent=2)
    print(f"[prepare] Saved {len(val_processed)} validation examples -> {val_path}")

    # -- Training split (used for few-shot selection) --
    train_raw = _fetch_split("train", n_train)
    train_processed = [_process_row(r, "train") for r in train_raw]
    train_path = DATA_DIR / "hellaswag_train.json"
    with open(train_path, "w") as f:
        json.dump(train_processed, f, indent=2)
    print(f"[prepare] Saved {len(train_processed)} training examples -> {train_path}")

    return val_path, train_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare HellaSwag data for the improvement pipeline."
    )
    parser.add_argument(
        "--n-val", type=int, default=200,
        help="Number of validation examples to download (default: 200)",
    )
    parser.add_argument(
        "--n-train", type=int, default=500,
        help="Number of training examples for few-shot pool (default: 500)",
    )
    args = parser.parse_args()

    prepare(n_val=args.n_val, n_train=args.n_train)
    print("[prepare] Done.")


if __name__ == "__main__":
    main()
