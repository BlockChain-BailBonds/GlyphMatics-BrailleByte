from __future__ import annotations

import argparse
import json
from pathlib import Path

from braillebyte.datasets import build_owned_repo_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-file-bytes", type=int, default=500_000)
    args = parser.parse_args()
    print(json.dumps(build_owned_repo_dataset(args.inventory, args.checkout_root, args.output, args.max_file_bytes), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
