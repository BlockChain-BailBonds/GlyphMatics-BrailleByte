from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class FileResult:
    path: str
    input_bytes: int
    chunk_bytes: int
    gzip_bytes: int
    zstd_bytes: int
    chunk_ratio: float
    gzip_ratio: float
    zstd_ratio: float
    round_trip: bool


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


def chunk_file(path: Path, chunk_size: int) -> tuple[bytes, list[dict[str, object]]]:
    raw = path.read_bytes()
    chunks = []
    for offset in range(0, len(raw), chunk_size):
        payload = raw[offset:offset + chunk_size]
        digest = sha256(payload).hexdigest()
        chunks.append({
            "offset": offset,
            "length": len(payload),
            "digest": digest,
            "payload": payload,
        })
    return raw, chunks


def encode_shared_chunk_bundle(files: list[tuple[Path, bytes, list[dict[str, object]]]]) -> bytes:
    out = bytearray()
    out.extend(b"WCB1")
    unique_chunks: dict[str, bytes] = {}
    for _, _, chunks in files:
        for chunk in chunks:
            digest = str(chunk["digest"])
            unique_chunks.setdefault(digest, chunk["payload"])
    out.extend(pack_varint(len(unique_chunks)))
    for digest, payload in unique_chunks.items():
        out.extend(bytes.fromhex(digest))
        out.extend(pack_varint(len(payload)))
        out.extend(payload)
    out.extend(pack_varint(len(files)))
    for path, raw, chunks in files:
        out.extend(pack_varint(len(path.as_posix().encode("utf-8"))))
        out.extend(path.as_posix().encode("utf-8"))
        out.extend(pack_varint(len(raw)))
        out.extend(pack_varint(len(chunks)))
        for chunk in chunks:
            out.extend(bytes.fromhex(str(chunk["digest"])))
            out.extend(pack_varint(int(chunk["offset"])))
            out.extend(pack_varint(int(chunk["length"])))
    return bytes(out)


def decode_shared_chunk_bundle(payload: bytes) -> list[bytes]:
    if not payload.startswith(b"WCB1"):
        raise ValueError("unsupported weight bundle format")
    pos = 4
    unique_count, pos = read_varint(payload, pos)
    unique_chunks: dict[str, bytes] = {}
    for _ in range(unique_count):
        digest = payload[pos:pos + 32].hex()
        pos += 32
        length, pos = read_varint(payload, pos)
        unique_chunks[digest] = payload[pos:pos + length]
        pos += length
    file_count, pos = read_varint(payload, pos)
    restored: list[bytes] = []
    for _ in range(file_count):
        path_len, pos = read_varint(payload, pos)
        pos += path_len
        raw_len, pos = read_varint(payload, pos)
        chunk_count, pos = read_varint(payload, pos)
        file_bytes = bytearray()
        for _ in range(chunk_count):
            digest = payload[pos:pos + 32].hex()
            pos += 32
            offset, pos = read_varint(payload, pos)
            length, pos = read_varint(payload, pos)
            del offset
            file_bytes.extend(unique_chunks[digest][:length])
        restored.append(bytes(file_bytes[:raw_len]))
    return restored


def read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
    raise ValueError("truncated varint")


def gzip_size(payload: bytes) -> int:
    return len(gzip.compress(payload, compresslevel=9))


def zstd_size(payload: bytes) -> int:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as input_file:
        input_file.write(payload)
        input_file.flush()
        input_path = Path(input_file.name)
    try:
        with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as output_file:
            output_path = Path(output_file.name)
        try:
            if output_path.exists():
                output_path.unlink()
            subprocess.run(["zstd", "-q", "-19", "-f", str(input_path), "-o", str(output_path)], check=True)
            return output_path.stat().st_size
        finally:
            if output_path.exists():
                output_path.unlink()
    finally:
        if input_path.exists():
            input_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--chunk-size", type=int, default=1 << 20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = []
    file_payloads = []
    for entry in args.paths:
        path = Path(entry)
        raw, chunks = chunk_file(path, args.chunk_size)
        file_payloads.append((path, raw, chunks))
    payload = encode_shared_chunk_bundle(file_payloads)
    bundle_gzip = gzip_size(payload)
    bundle_zstd = zstd_size(payload)
    restored = decode_shared_chunk_bundle(payload)
    for (path, raw, chunks), restored_raw in zip(file_payloads, restored):
        gzip_bytes = gzip_size(raw)
        zstd_bytes = zstd_size(raw)
        results.append(FileResult(
            path=str(path),
            input_bytes=len(raw),
            chunk_bytes=len(payload),
            gzip_bytes=gzip_bytes,
            zstd_bytes=zstd_bytes,
            chunk_ratio=ratio(len(raw), len(payload)),
            gzip_ratio=ratio(len(raw), gzip_bytes),
            zstd_ratio=ratio(len(raw), zstd_bytes),
            round_trip=restored_raw == raw,
        ))

    report = {
        "format": "GlyphMatics Weight Compression Benchmark",
        "chunk_size": args.chunk_size,
        "bundle_bytes": len(payload),
        "bundle_gzip_bytes": bundle_gzip,
        "bundle_zstd_bytes": bundle_zstd,
        "bundle_gzip_ratio": ratio(sum(r.input_bytes for r in results), bundle_gzip),
        "bundle_zstd_ratio": ratio(sum(r.input_bytes for r in results), bundle_zstd),
        "results": [result.__dict__ for result in results],
    }
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
