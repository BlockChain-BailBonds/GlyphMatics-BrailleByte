from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .bytecode import OPCODE_TABLE, SemanticBytecode
from .codec import BrailleByteCodec
from .semantic import ConceptRegistry
from .semantic_graph import SemanticGraph, SemanticGraphCodec


DICT_REF = 0xFE
RAW_ESCAPE = 0xFF


def _pack_varint(value: int) -> tuple[int, ...]:
    if value < 0:
        raise ValueError("negative values are unsupported")
    out = []
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return tuple(out)


def _unpack_varint(data: Sequence[int], pos: int) -> tuple[int, int]:
    value = shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
    raise ValueError("truncated varint")


def _escape_bytes(payload: bytes) -> tuple[int, ...]:
    out = [RAW_ESCAPE]
    out.extend(_pack_varint(len(payload)))
    out.extend(payload)
    return tuple(out)


def _unescape_bytes(data: Sequence[int], pos: int) -> tuple[tuple[int, ...], int]:
    length, pos = _unpack_varint(data, pos)
    end = pos + length
    if end > len(data):
        raise ValueError("truncated raw escape")
    return tuple(data[pos:end]), end


@dataclass(frozen=True)
class PriorPhraseModel:
    phrase_to_id: dict[tuple[int, ...], int]
    id_to_phrase: dict[int, tuple[int, ...]]
    registry_version: str


@dataclass
class BrailleByteCompressor:
    model: PriorPhraseModel | None = None
    codec: BrailleByteCodec | None = None
    graph_codec: SemanticGraphCodec | None = None

    @classmethod
    def train(cls, streams: Iterable[Sequence[int]], min_frequency: int = 2) -> "BrailleByteCompressor":
        counts = Counter(tuple(stream) for stream in streams)
        phrase_to_id: dict[tuple[int, ...], int] = {}
        id_to_phrase: dict[int, tuple[int, ...]] = {}
        next_id = 1
        for phrase, freq in counts.items():
            if freq >= min_frequency and len(phrase) >= 2:
                phrase_to_id[phrase] = next_id
                id_to_phrase[next_id] = phrase
                next_id += 1
        registry_version = ConceptRegistry().to_json()[:16]
        return cls(PriorPhraseModel(phrase_to_id=phrase_to_id, id_to_phrase=id_to_phrase, registry_version=registry_version))

    @classmethod
    def train_from_texts(
        cls,
        texts: Iterable[str],
        *,
        min_frequency: int = 3,
        min_len: int = 4,
        max_len: int = 32,
    ) -> "BrailleByteCompressor":
        counts: Counter[tuple[int, ...]] = Counter()
        for text in texts:
            data = tuple(text.encode("utf-8"))
            seen_in_text: set[tuple[int, ...]] = set()
            limit = min(len(data), max_len)
            for size in range(min_len, limit + 1):
                for start in range(0, len(data) - size + 1):
                    phrase = data[start:start + size]
                    if phrase not in seen_in_text:
                        counts[phrase] += 1
                        seen_in_text.add(phrase)
        phrase_to_id: dict[tuple[int, ...], int] = {}
        id_to_phrase: dict[int, tuple[int, ...]] = {}
        next_id = 1
        for phrase, freq in sorted(counts.items(), key=lambda item: (-len(item[0]), -item[1], item[0])):
            if freq >= min_frequency and len(phrase) >= min_len:
                phrase_to_id[phrase] = next_id
                id_to_phrase[next_id] = phrase
                next_id += 1
        registry_version = ConceptRegistry().to_json()[:16]
        return cls(PriorPhraseModel(phrase_to_id=phrase_to_id, id_to_phrase=id_to_phrase, registry_version=registry_version))

    def _ensure_codec(self) -> BrailleByteCodec:
        if self.codec is None:
            self.codec = BrailleByteCodec()
        return self.codec

    def _ensure_graph_codec(self) -> SemanticGraphCodec:
        if self.graph_codec is None:
            self.graph_codec = SemanticGraphCodec(self._ensure_codec().concepts)
        return self.graph_codec

    def encode_graph(self, graph: SemanticGraph) -> tuple[int, ...]:
        return self._ensure_graph_codec().encode(graph)

    def decode_graph(self, stream: Sequence[int]) -> SemanticGraph:
        return self._ensure_graph_codec().decode(tuple(stream))

    def compress_graph(self, graph: SemanticGraph) -> tuple[int, ...]:
        return self.compress(self.encode_graph(graph))

    def decompress_graph(self, stream: Sequence[int]) -> SemanticGraph:
        return self.decode_graph(self.decompress(stream))

    def compress_text(self, text: str) -> tuple[int, ...]:
        return self.compress_with_prior(text)

    def decompress_text(self, stream: Sequence[int]) -> str:
        data = bytes(self.decompress(stream))
        _, _, text_bytes = data.partition(b"\x00")
        return text_bytes.decode("utf-8", errors="replace")

    def compress_any(self, value: Any) -> tuple[int, ...]:
        if isinstance(value, SemanticGraph):
            return self.compress_graph(value)
        if isinstance(value, str):
            return self.compress_text(value)
        return self.compress(tuple(value))

    def predict_graph(self, text: str) -> dict[str, Any]:
        graph = self._ensure_graph_codec().parse(text)
        nodes = []
        for node in graph["nodes"]:
            if "concept_id" in node:
                nodes.append({"surface": node["surface"], "concept_id": node["concept_id"], "predicted": node["concept_id"], "residual": None})
            elif "alternatives" in node:
                nodes.append({"surface": node["surface"], "alternatives": node["alternatives"], "predicted": node["alternatives"][0], "residual": node["surface"]})
            else:
                nodes.append({"surface": node["surface"], "predicted": None, "residual": node["surface"]})
        return {"type": "predicted_graph", "nodes": nodes}

    def _graph_header_bytes(self, text: str) -> bytes:
        predicted = self.predict_graph(text)
        bytecode = SemanticBytecode(OPCODE_TABLE, provisional_registry=True)
        instructions = [bytecode.begin("GRAPH", label="prior")]
        for node in predicted["nodes"]:
            if "concept_id" in node:
                instructions.append({"op": "PAIR", "role": node["surface"], "concept_id": node["concept_id"]})
            elif "alternatives" in node:
                instructions.append({"op": "ALTERNATIVE", "choices": ",".join(node["alternatives"])})
            else:
                instructions.append({"op": "UNKNOWN", "raw": node["surface"]})
        instructions.append(bytecode.end("GRAPH"))
        return bytecode.to_bytes(instructions)

    def encode_residual(self, text: str) -> tuple[int, ...]:
        predicted = self.predict_graph(text)
        ops: list[dict[str, Any]] = []
        tokens = [t for t in text.split() if t]
        for idx, token in enumerate(tokens):
            node = predicted["nodes"][idx] if idx < len(predicted["nodes"]) else {"surface": ""}
            if node.get("concept_id") and self._ensure_codec().concepts.resolve(token):
                ops.append({"op": "KEEP", "ref": node["concept_id"]})
            elif node.get("alternatives") and token in node["alternatives"]:
                ops.append({"op": "ALT", "token": token})
            else:
                ops.append({"op": "LIT", "token": token})
        raw = json.dumps({"format": "GlyphMaticsResidual", "ops": ops}, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return self.compress(tuple(raw))

    def decode_residual(self, stream: Sequence[int]) -> str:
        raw = bytes(self.decompress(stream))
        payload = json.loads(raw.decode("utf-8"))
        tokens = []
        for op in payload["ops"]:
            if op["op"] == "KEEP":
                concept = self._ensure_codec().concepts.get(op["ref"])
                tokens.append(next(iter(concept.surfaces.get("en", [concept.definition]))))
            else:
                tokens.append(op["token"])
        return " ".join(tokens)

    def compress_with_prior(self, text: str) -> tuple[int, ...]:
        graph_bytes = self._graph_header_bytes(text)
        residual = self.encode_residual(text)
        header = (0xF0,) + _pack_varint(len(graph_bytes)) + tuple(graph_bytes) + _pack_varint(len(residual)) + tuple(residual)
        return self.compress(header)

    def decompress_with_prior(self, stream: Sequence[int]) -> str:
        data = self.decompress(stream)
        if not data or data[0] != 0xF0:
            return self.decompress_text(data)
        pos = 1
        graph_len, pos = _unpack_varint(data, pos)
        graph_bytes = bytes(data[pos:pos + graph_len])
        pos += graph_len
        residual_len, pos = _unpack_varint(data, pos)
        residual_bytes = tuple(data[pos:pos + residual_len])
        _ = SemanticBytecode(OPCODE_TABLE, provisional_registry=True).from_bytes(graph_bytes)
        return self.decode_residual(residual_bytes)

    def compress(self, stream: Sequence[int]) -> tuple[int, ...]:
        if self.model is None:
            out = []
            for value in stream:
                if value in (DICT_REF, RAW_ESCAPE):
                    out.extend(_escape_bytes(bytes([value])))
                else:
                    out.append(value)
            return tuple(out)

        source = tuple(stream)
        out: list[int] = []
        i = 0
        phrases = sorted(self.model.phrase_to_id.items(), key=lambda item: len(item[0]), reverse=True)
        while i < len(source):
            matched = False
            for phrase, phrase_id in phrases:
                if source[i:i + len(phrase)] == phrase:
                    out.append(DICT_REF)
                    out.extend(_pack_varint(phrase_id))
                    i += len(phrase)
                    matched = True
                    break
            if matched:
                continue
            value = source[i]
            if value in (DICT_REF, RAW_ESCAPE):
                out.extend(_escape_bytes(bytes([value])))
            else:
                out.append(value)
            i += 1
        return tuple(out)

    def decompress(self, stream: Sequence[int]) -> tuple[int, ...]:
        data = tuple(stream)
        out: list[int] = []
        i = 0
        while i < len(data):
            value = data[i]
            i += 1
            if value == RAW_ESCAPE:
                raw, i = _unescape_bytes(data, i)
                out.extend(raw)
            elif value == DICT_REF:
                if self.model is None:
                    raise ValueError("dictionary reference without trained model")
                index, i = _unpack_varint(data, i)
                phrase = self.model.id_to_phrase.get(index)
                if phrase is None:
                    raise ValueError("unknown dictionary reference")
                out.extend(phrase)
            else:
                out.append(value)
        return tuple(out)

    def compress_to_braille(self, stream: Sequence[int]) -> str:
        return self._ensure_codec().encode_bytes(bytes(self.compress(stream)))

    def decompress_braille(self, cells: str) -> tuple[int, ...]:
        return self.decompress(self._ensure_codec().decode_bytes(cells))

    def compress_graph_to_braille(self, graph: SemanticGraph) -> str:
        return self.compress_to_braille(self.encode_graph(graph))

    def decompress_braille_to_graph(self, cells: str) -> SemanticGraph:
        return self.decode_graph(self.decompress_braille(cells))

    def model_json(self, corpus: list[Sequence[int]]) -> str:
        trained = self.train(corpus, min_frequency=2)
        phrases = [
            {"phrase": list(phrase), "id": phrase_id}
            for phrase, phrase_id in trained.model.phrase_to_id.items()
        ]
        return json.dumps(
            {"format": "BrailleByteCompressionModel", "phrases": phrases, "registry_prior": trained.model.registry_version},
            indent=2,
            ensure_ascii=False,
        )
