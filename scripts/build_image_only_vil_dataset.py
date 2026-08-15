from __future__ import annotations

import argparse
import json
from pathlib import Path

from braillebyte.datasets import build_image_only_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lines-per-page", type=int, default=64)
    parser.add_argument("--columns", type=int, default=108)
    parser.add_argument("--font-size", type=int, default=14)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build_image_only_dataset(
        args.source, args.output, lines_per_page=args.lines_per_page, columns=args.columns,
        font_size=args.font_size, resume=args.resume, max_records=args.max_records,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
