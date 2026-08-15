from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
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


@dataclass(frozen=True)
class ManifestCubie:
    kind: str
    label: str
    orientation: int = 0


@dataclass
class RubiksCheckpointManifest:
    model_id: str
    architecture_id: str
    tokenizer_id: str
    quantization_scheme: str
    chunk_index: GlyphChunkIndex
    reconstruction_order: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "architecture_id": self.architecture_id,
            "tokenizer_id": self.tokenizer_id,
            "quantization_scheme": self.quantization_scheme,
            "chunk_index": self.chunk_index.to_dict(),
            "reconstruction_order": list(self.reconstruction_order),
        }

    def to_cube_summary(self) -> dict[str, object]:
        return {
            "centers": {
                "front": self.architecture_id,
                "right": self.tokenizer_id,
                "left": self.quantization_scheme,
                "top": self.model_id,
                "bottom": "integrity",
                "back": "reconstruction",
            },
            "edges": [route.tensor_name for route in self.chunk_index.tensors],
            "corners": [chunk.chunk_id for chunk in self.chunk_index.chunks[:8]],
            "reconstruction_order": list(self.reconstruction_order or tuple(chunk.chunk_id for chunk in self.chunk_index.chunks)),
        }

    def build_cube(self):
        from .cube import FACE_ORDER, GlyphCube, GlyphCubeFace

        summary = self.to_cube_summary()
        faces = {}
        for face in FACE_ORDER:
            payload = json.dumps(summary.get("centers", {}).get(face, face), ensure_ascii=False).encode("utf-8")
            frame = {"role": face, "kind": "manifest"}
            if face == "front":
                frame["domain"] = self.architecture_id
            elif face == "right":
                frame["label"] = self.tokenizer_id
            elif face == "left":
                frame["label"] = self.quantization_scheme
            elif face == "top":
                frame["label"] = self.model_id
            elif face == "bottom":
                frame["label"] = "integrity"
            elif face == "back":
                frame["label"] = "reconstruction"
            faces[face] = GlyphCubeFace(name=face, payload=payload, semantic_frame=frame)
        return GlyphCube(faces=faces)

    @classmethod
    def from_cube(cls, cube, chunk_index: GlyphChunkIndex) -> "RubiksCheckpointManifest":
        summary = cube.semantic_summary()
        return cls(
            model_id=summary["top"].get("label", "unknown"),
            architecture_id=summary["front"].get("domain", "unknown"),
            tokenizer_id=summary["right"].get("label", "unknown"),
            quantization_scheme=summary["left"].get("label", "unknown"),
            chunk_index=chunk_index,
            reconstruction_order=tuple(chunk.chunk_id for chunk in chunk_index.chunks),
        )

    def verify(self, payloads: dict[str, bytes]) -> bool:
        for chunk in self.chunk_index.chunks:
            payload = payloads.get(chunk.chunk_id)
            if payload is None or sha256(payload).hexdigest() != chunk.digest:
                return False
        return True


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

    def dedupe_candidates(self) -> tuple[tuple[str, ...], ...]:
        seen: dict[tuple[str, int, int], tuple[str, ...]] = {}
        ordered: list[tuple[str, ...]] = []
        for chunk in self.chunks:
            key = (chunk.digest, chunk.length, chunk.offset)
            if key in seen:
                continue
            routes = (chunk.chunk_id,)
            seen[key] = routes
            ordered.append(routes)
        return tuple(ordered)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "path": chunk.path,
                    "offset": chunk.offset,
                    "length": chunk.length,
                    "digest": chunk.digest,
                }
                for chunk in self.chunks
            ],
            "vocabulary": [
                {
                    "token_start": shard.token_start,
                    "token_end": shard.token_end,
                    "embed_chunk_id": shard.embed_chunk_id,
                    "output_chunk_id": shard.output_chunk_id,
                }
                for shard in self.vocabulary
            ],
            "tensors": [
                {
                    "layer_index": route.layer_index,
                    "tensor_name": route.tensor_name,
                    "chunk_ids": list(route.chunk_ids),
                }
                for route in self.tensors
            ],
            "dedupe_candidates": [list(candidate) for candidate in self.dedupe_candidates()],
        }

    @classmethod
    def from_file(cls, path: Path) -> "GlyphChunkIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            model_id=payload["model_id"],
            chunks=tuple(ChunkRecord(**chunk) for chunk in payload.get("chunks", [])),
            vocabulary=tuple(VocabularyShard(**shard) for shard in payload.get("vocabulary", [])),
            tensors=tuple(TensorRoute(layer_index=item["layer_index"], tensor_name=item["tensor_name"], chunk_ids=tuple(item["chunk_ids"])) for item in payload.get("tensors", [])),
        )
