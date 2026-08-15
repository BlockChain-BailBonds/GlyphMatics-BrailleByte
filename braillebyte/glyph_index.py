from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Iterable, Sequence


VOCABULARY_GRAPH = "vocabulary_graph"
ARCHITECTURE_GRAPH = "architecture_graph"
CHUNK_GRAPH = "chunk_graph"
INTEGRITY_GRAPH = "integrity_graph"


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    path: str
    offset: int
    length: int
    digest: str


@dataclass(frozen=True)
class VocabularyShard:
    token_start: int
    token_end: int
    embed_chunk_id: str
    output_chunk_id: str


@dataclass(frozen=True)
class TensorRoute:
    layer_index: int
    tensor_name: str
    chunk_ids: tuple[str, ...]


@dataclass
class GlyphChunkIndex:
    model_id: str
    chunks: tuple[ChunkRecord, ...] = field(default_factory=tuple)
    vocabulary: tuple[VocabularyShard, ...] = field(default_factory=tuple)
    tensors: tuple[TensorRoute, ...] = field(default_factory=tuple)

    def route_tokens(self, token_ids: Sequence[int]) -> tuple[ChunkRecord, ...]:
        if not self.vocabulary:
            return self.chunks[:1]
        shard = self.vocabulary[0]
        return tuple(chunk for chunk in self.chunks if chunk.chunk_id in {shard.embed_chunk_id, shard.output_chunk_id})

    def route_tensor(self, layer_index: int, tensor_name: str) -> tuple[ChunkRecord, ...]:
        for route in self.tensors:
            if route.layer_index == layer_index and route.tensor_name == tensor_name:
                return tuple(chunk for chunk in self.chunks if chunk.chunk_id in route.chunk_ids)
        return ()

    def token_route_glyphs(self, token_ids: Sequence[int]) -> str:
        routed = self.route_tokens(token_ids)
        return "\n".join(
            [
                VOCABULARY_GRAPH,
                ARCHITECTURE_GRAPH,
                CHUNK_GRAPH,
                INTEGRITY_GRAPH,
                f"model={self.model_id}",
                f"tokens={list(token_ids)}",
                f"chunks={[c.chunk_id for c in routed]}",
            ]
        )

    def verify_chunk(self, chunk_id: str, payload: bytes) -> bool:
        chunk = next((item for item in self.chunks if item.chunk_id == chunk_id), None)
        if chunk is None:
            return False
        return sha256(payload).hexdigest() == chunk.digest
