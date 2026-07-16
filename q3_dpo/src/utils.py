from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import numpy as np

REQUIRED_FIELDS = [
    "policy_chosen_logp", "policy_rejected_logp",
    "ref_chosen_logp", "ref_rejected_logp",
]

def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records

def extract_logprob_arrays(records: Iterable[dict]):
    records = list(records)
    for i, record in enumerate(records):
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"Record {i} is missing fields: {missing}")
    return tuple(np.asarray([r[field] for r in records], dtype=float) for field in REQUIRED_FIELDS)
