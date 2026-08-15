from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .bytecode import SemanticBytecode, OPCODE_TABLE
from .compound import CompoundMeaning, CompoundMeaningCodec
from .semantic import ConceptRegistry


BRAILLE_BASE = 0x2800


@dataclass
class SemanticGraphCodec:
    registry: ConceptRegistry

    def parse(self, text: str) -> dict[str, Any]:
        tokens = [t for t in text.split() if t]
        nodes = []
        for token in tokens:
            matches = self.registry.resolve(token)
            if len(matches) == 1:
                nodes.append({"surface": token, "concept_id": matches[0].concept_id})
            elif matches:
                nodes.append({"surface": token, "alternatives": [m.concept_id for m in matches]})
            else:
                nodes.append({"surface": token, "literal": token.encode("utf-8").hex()})
        return {"type": "semantic_graph", "nodes": nodes}

    def realize(self, graph: dict[str, Any], language: str = "en") -> str:
        words = []
        for node in graph.get("nodes", []):
            if "concept_id" in node:
                concept = self.registry.get(node["concept_id"])
                words.append(concept.surfaces.get(language, [concept.definition])[0])
            else:
                words.append(node["surface"])
        return " ".join(words)


class BrailleByteCodec:
    def encode_bytes(self, payload: bytes) -> str:
        return "".join(chr(BRAILLE_BASE + b) for b in payload)

    def decode_bytes(self, cells: str) -> bytes:
        return bytes(ord(ch) - BRAILLE_BASE for ch in cells)

    def encode(self, text: str) -> str:
        return self.encode_bytes(text.encode("utf-8"))

    def decode(self, cells: str) -> str:
        return self.decode_bytes(cells).decode("utf-8", errors="replace")

    def encode_payload(self, payload: bytes) -> str:
        return self.encode_bytes(payload)

    def decode_payload(self, cells: str) -> bytes:
        return self.decode_bytes(cells)

    def explain(self, text: str) -> dict[str, Any]:
        registry = ConceptRegistry()
        graph = SemanticGraphCodec(registry).parse(text)
        return {
            "input": text,
            "graph": graph,
            "braille": self.encode(text),
            "bytecode": SemanticBytecode(OPCODE_TABLE, provisional_registry=True).begin("TEXT"),
        }

    def encode_compound(self, pairs: list[tuple[str, str]], nested: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        bytecode = SemanticBytecode(OPCODE_TABLE, provisional_registry=True)
        codec = CompoundMeaningCodec(bytecode)
        meaning = CompoundMeaning(pairs=pairs, nested=nested or [])
        compound = codec.encode(meaning)
        return {
            "provisional_registry": True,
            "opcode_table": OPCODE_TABLE,
            "compound": compound,
            "braille": self.encode_payload(bytecode.to_bytes(compound)),
        }

    def decode_compound(self, braille: str) -> dict[str, Any]:
        bytecode = SemanticBytecode(OPCODE_TABLE, provisional_registry=True)
        instructions = bytecode.from_bytes(self.decode_payload(braille))
        return {
            "provisional_registry": True,
            "opcode_table": OPCODE_TABLE,
            "compound": instructions,
            "braille": braille,
        }

    def to_json(self, obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=2)
