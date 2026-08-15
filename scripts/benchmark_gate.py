from __future__ import annotations

import argparse
import json
from pathlib import Path

from braillebyte.codec import BrailleByteCodec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    path = Path(args.input)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    codec = BrailleByteCodec()
    print(json.dumps({"records": len(rows), "sample": codec.explain(rows[0]["text"]) if rows else {}}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
