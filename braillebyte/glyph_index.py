"""Four-section glyph index for lossless model-chunk retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Sequence


VOCABULARY_GRAPH = 224
ARCHITECTURE_GRAPH = 225
CHUNK_GRAPH = 226
INTEGRITY_GRAPH = 227


def varuint(value: int) -> tuple[int, ...]:
    if value < 0:
        raise ValueError('varuint values must be non-negative')
    result: list[int] = []
    while True:
        byte = value & 0x7F
        value >>= 7
        result.append(byte | (0x80 if value else 0))
        if not value:
            return tuple(result)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    uri: str
    offset: int
    length: int
    sha256: str
    codec: str = 'zstd'


@dataclass(frozen=True)
class VocabularyShard:
    start_token_id: int
    end_token_id: int
    embedding_chunk_id: str
    output_chunk_id: str


@dataclass(frozen=True)
class TensorRoute:
    layer: int
    component: str
    chunk_ids: tuple[str, ...]


class GlyphChunkIndex:
    """Routes model requests through vocabulary, architecture, chunk, and integrity graphs."""

    def __init__(
        self,
        *,
        model_id: str,
        chunks: Iterable[ChunkRecord],
        vocabulary: Iterable[VocabularyShard],
        tensors: Iterable[TensorRoute],
    ) -> None:
        self.model_id = model_id
        self.chunks = tuple(chunks)
        self.vocabulary = tuple(sorted(vocabulary, key=lambda shard: shard.start_token_id))
        self.tensors = tuple(tensors)
        self._chunks = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.validate()

    def validate(self) -> None:
        if not self.model_id:
            raise ValueError('model_id is required')
        if len(self._chunks) != len(self.chunks):
            raise ValueError('chunk IDs must be unique')
        for chunk in self.chunks:
            if chunk.offset < 0 or chunk.length <= 0:
                raise ValueError(f'invalid byte range for chunk {chunk.chunk_id}')
            if len(chunk.sha256) != 64 or any(char not in '0123456789abcdef' for char in chunk.sha256):
                raise ValueError(f'invalid SHA-256 for chunk {chunk.chunk_id}')
        previous_end = -1
        for shard in self.vocabulary:
            if shard.start_token_id < 0 or shard.end_token_id < shard.start_token_id or shard.start_token_id <= previous_end:
                raise ValueError('vocabulary shards must be non-overlapping ordered ranges')
            previous_end = shard.end_token_id
            self._require_chunks((shard.embedding_chunk_id, shard.output_chunk_id))
        for route in self.tensors:
            if route.layer < 0 or not route.component or not route.chunk_ids:
                raise ValueError('tensor routes require layer, component, and chunks')
            self._require_chunks(route.chunk_ids)

    def _require_chunks(self, chunk_ids: Iterable[str]) -> None:
        missing = set(chunk_ids) - self._chunks.keys()
        if missing:
            raise ValueError(f'unknown chunk IDs: {sorted(missing)}')

    def route_tokens(self, token_ids: Sequence[int], *, purpose: str = 'embedding') -> tuple[ChunkRecord, ...]:
        if purpose not in ('embedding', 'output'):
            raise ValueError("purpose must be 'embedding' or 'output'")
        result: list[ChunkRecord] = []
        for token_id in token_ids:
            shard = next((item for item in self.vocabulary if item.start_token_id <= token_id <= item.end_token_id), None)
            if shard is None:
                raise KeyError(f'token ID has no vocabulary route: {token_id}')
            chunk_id = shard.embedding_chunk_id if purpose == 'embedding' else shard.output_chunk_id
            chunk = self._chunks[chunk_id]
            if chunk not in result:
                result.append(chunk)
        return tuple(result)

    def route_tensor(self, layer: int, component: str) -> tuple[ChunkRecord, ...]:
        route = next((item for item in self.tensors if item.layer == layer and item.component == component), None)
        if route is None:
            raise KeyError(f'no architecture route for layer {layer} component {component!r}')
        return tuple(self._chunks[chunk_id] for chunk_id in route.chunk_ids)

    def token_route_glyphs(self, token_ids: Sequence[int], *, purpose: str = 'embedding') -> tuple[int, ...]:
        """Emit a BrailleByte-compatible byte route across all four graph sections."""
        chunks = self.route_tokens(token_ids, purpose=purpose)
        result: list[int] = [VOCABULARY_GRAPH]
        for token_id in token_ids:
            shard_index = next(index for index, shard in enumerate(self.vocabulary) if shard.start_token_id <= token_id <= shard.end_token_id)
            result.extend(varuint(shard_index))
        result.append(ARCHITECTURE_GRAPH)
        result.extend(varuint(0))  # token embedding/output stage in the model graph
        result.append(CHUNK_GRAPH)
        for chunk in chunks:
            result.extend(varuint(self.chunks.index(chunk)))
        result.append(INTEGRITY_GRAPH)
        for chunk in chunks:
            result.extend(varuint(self.chunks.index(chunk)))
        return tuple(result)

    def verify_chunk(self, chunk_id: str, payload: bytes) -> bool:
        return sha256(payload).hexdigest() == self._chunks[chunk_id].sha256

    def to_dict(self) -> dict[str, object]:
        return {
            'format': 'braillebyte-four-graph-chunk-index-v1',
            'model_id': self.model_id,
            'sections': {
                'vocabulary': VOCABULARY_GRAPH,
                'architecture': ARCHITECTURE_GRAPH,
                'chunk': CHUNK_GRAPH,
                'integrity': INTEGRITY_GRAPH,
            },
            'chunks': [chunk.__dict__ for chunk in self.chunks],
            'vocabulary': [shard.__dict__ for shard in self.vocabulary],
            'tensors': [
                {'layer': route.layer, 'component': route.component, 'chunk_ids': list(route.chunk_ids)}
                for route in self.tensors
            ],
        }
