from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from braillebyte.compression import BrailleByteCompressor
from braillebyte.codec import BrailleByteCodec
from braillebyte.semantic_graph import SemanticGraph


def load_corpus(path: Path) -> list[str]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row.get("text") or row.get("source") or "")
    return rows


def ratio(original: int, compressed: int) -> float:
    return round(original / compressed, 4) if compressed else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    out = Path(args.output)
    corpus = load_corpus(source)
    codec = BrailleByteCodec()
    compressor = BrailleByteCompressor.train_from_texts(corpus, min_frequency=3, min_len=4, max_len=24)
    rows = []
    for text in corpus:
        plain = text.encode("utf-8")
        prior = compressor.compress_with_prior(text)
        braille = codec.encode_bytes(bytes(prior))
        rows.append({
            "text": text,
            "plain_bytes": len(plain),
            "prior_bytes": len(prior),
            "braille_cells": len(braille),
            "ratio_plain_to_prior": ratio(len(plain), len(prior)),
            "ratio_plain_to_braille": ratio(len(plain), len(braille)),
            "round_trip": compressor.decompress_with_prior(prior) == text,
        })
    report = {
        "format": "GlyphMatics Compression Benchmark Suite",
        "records": len(rows),
        "rows": rows,
        "all_round_trip": all(row["round_trip"] for row in rows),
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
