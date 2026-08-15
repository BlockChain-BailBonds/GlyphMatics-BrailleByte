"""Build a deterministic example of the four-section model chunk graph."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from braillebyte.glyph_index import ChunkRecord, GlyphChunkIndex, TensorRoute, VocabularyShard


def demo_chunk(chunk_id: str, offset: int) -> ChunkRecord:
    payload = f'GlyphMatics-BrailleByte-example:{chunk_id}'.encode('utf-8')
    return ChunkRecord(
        chunk_id=chunk_id,
        uri=f'chunks/{chunk_id}.zst',
        offset=offset,
        length=1_073_741_824,
        sha256=sha256(payload).hexdigest(),
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    chunks = [
        demo_chunk('embed-0', 0), demo_chunk('embed-1', 1_073_741_824),
        demo_chunk('output-0', 2_147_483_648), demo_chunk('output-1', 3_221_225_472),
        demo_chunk('layer-0-attention', 4_294_967_296), demo_chunk('layer-0-mlp', 5_368_709_120),
    ]
    index = GlyphChunkIndex(
        model_id='example-30b-layout-not-weights',
        chunks=chunks,
        vocabulary=(
            VocabularyShard(0, 31_999, 'embed-0', 'output-0'),
            VocabularyShard(32_000, 63_999, 'embed-1', 'output-1'),
        ),
        tensors=(
            TensorRoute(0, 'attention', ('layer-0-attention',)),
            TensorRoute(0, 'mlp', ('layer-0-mlp',)),
        ),
    )
    result = index.to_dict()
    result['example_only'] = True
    result['example_payload_hash_note'] = 'Hashes validate deterministic example payload labels, not model weights.'
    result['sample_token_route'] = list(index.token_route_glyphs((42, 32_001)))
    output = root / 'data' / 'four_graph_chunk_index.example.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
