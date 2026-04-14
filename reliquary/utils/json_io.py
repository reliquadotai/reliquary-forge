from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def sha256_json(data: Any) -> str:
    return hashlib.sha256(stable_json_dumps(data).encode("utf-8")).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
