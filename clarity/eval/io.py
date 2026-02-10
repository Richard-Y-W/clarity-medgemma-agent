import json
from typing import Dict, Iterator


def read_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
