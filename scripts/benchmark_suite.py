from __future__ import annotations

import argparse
import json
import sys
import struct
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


def pack_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("negative values are unsupported")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def write_text(out: bytearray, text: str) -> None:
    data = text.encode("utf-8")
    out.extend(pack_varint(len(data)))
    out.extend(data)


def encode_binary_report(report: dict[str, object]) -> bytes:
    out = bytearray()
    out.extend(b"BMS1")
    write_text(out, str(report["format"]))
    out.extend(pack_varint(int(report["records"])))
    rows = report["rows"]
    out.extend(pack_varint(len(rows)))
    for row in rows:
        assert isinstance(row, dict)
        write_text(out, str(row["text"]))
        out.extend(pack_varint(int(row["plain_bytes"])))
        out.extend(pack_varint(int(row["prior_bytes"])))
        out.extend(pack_varint(int(row["braille_cells"])))
        out.extend(struct.pack(">d", float(row["ratio_plain_to_prior"])))
        out.extend(struct.pack(">d", float(row["ratio_plain_to_braille"])))
        out.append(1 if row["round_trip"] else 0)
    out.append(1 if report["all_round_trip"] else 0)
    return bytes(out)


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
    out.write_bytes(encode_binary_report(report))
    print(json.dumps(report, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
