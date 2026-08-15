from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


FACE_ORDER = ("front", "right", "left", "top", "bottom", "back")
FRAME_KEYS = {
    "role": 0x11,
    "domain": 0x10,
    "state": 0x38,
    "confidence": 0x40,
    "kind": 0x01,
    "label": 0x02,
}
FRAME_BACK = {value: key for key, value in FRAME_KEYS.items()}
VALUE_TAGS = {"str": 1, "int": 2, "float": 3, "bool": 4, "null": 5}
VALUE_BACK = {value: key for key, value in VALUE_TAGS.items()}


@dataclass(frozen=True)
class GlyphCubeFace:
    name: str
    payload: bytes = b""
    semantic_frame: dict[str, Any] = field(default_factory=dict)


@dataclass
class GlyphCube:
    faces: dict[str, GlyphCubeFace]

    def validate(self) -> None:
        missing = [face for face in FACE_ORDER if face not in self.faces]
        if missing:
            raise ValueError(f"missing cube faces: {', '.join(missing)}")

    def _pack_u16(self, value: int) -> bytes:
        if not 0 <= value <= 0xFFFF:
            raise ValueError("value outside u16 range")
        return value.to_bytes(2, "big")

    def _unpack_u16(self, data: bytes, pos: int) -> tuple[int, int]:
        return int.from_bytes(data[pos:pos + 2], "big"), pos + 2

    def _encode_frame(self, frame: dict[str, Any]) -> bytes:
        out = bytearray()
        items = sorted(frame.items(), key=lambda item: FRAME_KEYS.get(item[0], 255))
        out.append(len(items) & 0xFF)
        for key, value in items:
            tag = FRAME_KEYS.get(key)
            if tag is None:
                continue
            out.append(tag)
            type_name, raw = self._encode_value(value)
            out.append(VALUE_TAGS[type_name])
            out.extend(self._pack_u16(len(raw)))
            out.extend(raw)
        return bytes(out)

    def _decode_frame(self, data: bytes, pos: int) -> tuple[dict[str, Any], int]:
        count = data[pos]
        pos += 1
        frame: dict[str, Any] = {}
        for _ in range(count):
            tag = data[pos]
            pos += 1
            value_type = VALUE_BACK.get(data[pos])
            pos += 1
            length, pos = self._unpack_u16(data, pos)
            raw = data[pos:pos + length]
            pos += length
            key = FRAME_BACK.get(tag)
            if key is None:
                continue
            frame[key] = self._decode_value(value_type, raw)
        return frame, pos

    def to_bytes(self) -> bytes:
        self.validate()
        out = bytearray()
        out.extend(b"GCB1")
        for face in FACE_ORDER:
            payload = self.faces[face].payload
            frame = self._encode_frame(self.faces[face].semantic_frame)
            out.extend(self._pack_u16(len(payload)))
            out.extend(payload)
            out.extend(self._pack_u16(len(frame)))
            out.extend(frame)
        return bytes(out)

    def as_bytes(self) -> bytes:
        return self.to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "GlyphCube":
        if not data.startswith(b"GCB1"):
            raise ValueError("unsupported cube format")
        pos = 4
        faces: dict[str, GlyphCubeFace] = {}
        for face in FACE_ORDER:
            payload_len = int.from_bytes(data[pos:pos + 2], "big")
            pos += 2
            payload = data[pos:pos + payload_len]
            pos += payload_len
            frame_len = int.from_bytes(data[pos:pos + 2], "big")
            pos += 2
            frame, _ = cls._decode_frame_bytes(data[pos:pos + frame_len])
            pos += frame_len
            faces[face] = GlyphCubeFace(name=face, payload=payload, semantic_frame=frame)
        return cls(faces=faces)

    def semantic_summary(self) -> dict[str, Any]:
        self.validate()
        return {face: self.faces[face].semantic_frame for face in FACE_ORDER}

    @staticmethod
    def _decode_frame_bytes(data: bytes) -> tuple[dict[str, Any], int]:
        pos = 0
        count = data[pos]
        pos += 1
        frame: dict[str, Any] = {}
        for _ in range(count):
            tag = data[pos]
            pos += 1
            value_type = VALUE_BACK.get(data[pos])
            pos += 1
            length = int.from_bytes(data[pos:pos + 2], "big")
            pos += 2
            raw = data[pos:pos + length]
            pos += length
            key = FRAME_BACK.get(tag)
            if key is None:
                continue
            frame[key] = GlyphCube._decode_value_static(value_type, raw)
        return frame, pos

    def _encode_value(self, value: Any) -> tuple[str, bytes]:
        if value is None:
            return "null", b""
        if isinstance(value, bool):
            return "bool", b"\x01" if value else b"\x00"
        if isinstance(value, int) and not isinstance(value, bool):
            return "int", int(value).to_bytes(8, "big", signed=True)
        if isinstance(value, float):
            import struct

            return "float", struct.pack(">d", value)
        return "str", str(value).encode("utf-8")

    def _decode_value(self, value_type: str | None, raw: bytes) -> Any:
        return self._decode_value_static(value_type, raw)

    @staticmethod
    def _decode_value_static(value_type: str | None, raw: bytes) -> Any:
        if value_type == "null":
            return None
        if value_type == "bool":
            return raw != b"\x00"
        if value_type == "int":
            return int.from_bytes(raw, "big", signed=True)
        if value_type == "float":
            import struct

            return struct.unpack(">d", raw)[0]
        return raw.decode("utf-8")


def jsonless(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return "" if value is None else str(value)
    if isinstance(value, dict):
        parts = []
        for key in sorted(value):
            parts.append(f"{key}:{jsonless(value[key])}")
        return "{" + ",".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(jsonless(item) for item in value) + "]"
    return str(value)
