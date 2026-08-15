from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OPCODE_TABLE = {
    "BEGIN": 0x01,
    "END": 0x02,
    "CONCEPT": 0x10,
    "ROLE": 0x11,
    "PAIR": 0x12,
    "CAUSE": 0x20,
    "CONDITION": 0x21,
    "GOAL": 0x22,
    "ALTERNATIVE": 0x23,
    "SELECT": 0x24,
    "REASON": 0x25,
    "EVENT": 0x30,
    "ACTION": 0x31,
    "AGENT": 0x32,
    "PATIENT": 0x33,
    "TOOL": 0x34,
    "LOCATION": 0x35,
    "ATTRIBUTE": 0x36,
    "DIRECTION": 0x37,
    "STATE": 0x38,
    "UNKNOWN": 0x3F,
}
OPCODE_BACK = {value: key for key, value in OPCODE_TABLE.items()}
TEXT_FALLBACK = 0xFF


@dataclass
class SemanticBytecode:
    opcode_table: dict[str, int]
    provisional_registry: bool = True

    def to_bytes(self, instructions: list[dict[str, Any]]) -> bytes:
        out = bytearray()
        out.extend(b"GBC1")
        out.append(1 if self.provisional_registry else 0)
        out.append(len(instructions) & 0xFF)
        for inst in instructions:
            op = inst.get("op", "UNKNOWN")
            out.append(self.opcode_table.get(op, self.opcode_table["UNKNOWN"]) & 0xFF)
            if op == "PAIR":
                self._write_symbol(out, inst.get("role", ""))
                self._write_text(out, inst.get("concept_id", ""))
            elif op in {"BEGIN", "END"}:
                self._write_symbol(out, inst.get("kind", ""))
                if op == "BEGIN":
                    self._write_text(out, inst.get("label", ""))
            else:
                self._write_text(out, jsonless(inst))
        return bytes(out)

    def from_bytes(self, data: bytes) -> list[dict[str, Any]]:
        if not data.startswith(b"GBC1"):
            raise ValueError("Unsupported semantic bytecode format")
        pos = 4
        self.provisional_registry = bool(data[pos])
        pos += 1
        count = data[pos]
        pos += 1
        reverse = {value: key for key, value in self.opcode_table.items()}
        instructions: list[dict[str, Any]] = []
        for _ in range(count):
            opcode = data[pos]
            pos += 1
            op = reverse.get(opcode, "UNKNOWN")
            if op == "PAIR":
                role, pos = self._read_symbol(data, pos)
                concept_id, pos = self._read_text(data, pos)
                instructions.append({"op": "PAIR", "role": role, "concept_id": concept_id})
            elif op == "BEGIN":
                kind, pos = self._read_symbol(data, pos)
                label, pos = self._read_text(data, pos)
                item = {"op": "BEGIN", "kind": kind}
                if label:
                    item["label"] = label
                instructions.append(item)
            elif op == "END":
                kind, pos = self._read_symbol(data, pos)
                instructions.append({"op": "END", "kind": kind})
            else:
                blob, pos = self._read_text(data, pos)
                instructions.append({"op": op, "raw": blob})
        return instructions

    def _write_text(self, out: bytearray, text: str) -> None:
        data = text.encode("utf-8")
        if len(data) > 255:
            raise ValueError("Field too long for packed semantic bytecode")
        out.append(len(data))
        out.extend(data)

    def _write_symbol(self, out: bytearray, text: str) -> None:
        opcode = self.opcode_table.get(text)
        if opcode is not None:
            out.append(opcode & 0xFF)
            return
        out.append(TEXT_FALLBACK)
        self._write_text(out, text)

    def _read_text(self, data: bytes, pos: int) -> tuple[str, int]:
        length = data[pos]
        pos += 1
        text = data[pos:pos + length].decode("utf-8")
        return text, pos + length

    def _read_symbol(self, data: bytes, pos: int) -> tuple[str, int]:
        tag = data[pos]
        pos += 1
        if tag == TEXT_FALLBACK:
            return self._read_text(data, pos)
        return OPCODE_BACK.get(tag, "UNKNOWN"), pos

    def encode_pairs(self, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [{"op": "PAIR", "role": role, "concept_id": concept_id} for role, concept_id in pairs]

    def begin(self, kind: str, label: str | None = None) -> dict[str, Any]:
        block = {"op": "BEGIN", "kind": kind}
        if label is not None:
            block["label"] = label
        return block

    def end(self, kind: str) -> dict[str, Any]:
        return {"op": "END", "kind": kind}

    def frame(self, kind: str, body: list[dict[str, Any]], label: str | None = None) -> list[dict[str, Any]]:
        return [self.begin(kind, label), *body, self.end(kind)]


def jsonless(inst: dict[str, Any]) -> str:
    parts = []
    for key in sorted(inst):
        if key != "op":
            parts.append(f"{key}={inst[key]}")
    return ";".join(parts)
