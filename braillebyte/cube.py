from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


FACE_ORDER = ("front", "right", "left", "top", "bottom", "back")


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

    def to_bytes(self) -> bytes:
        self.validate()
        out = bytearray()
        out.extend(b"GCB1")
        for face in FACE_ORDER:
            payload = self.faces[face].payload
            frame = json.dumps(self.faces[face].semantic_frame, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
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
            frame = json.loads(data[pos:pos + frame_len].decode("utf-8"))
            pos += frame_len
            faces[face] = GlyphCubeFace(name=face, payload=payload, semantic_frame=frame)
        return cls(faces=faces)

    def semantic_summary(self) -> dict[str, Any]:
        self.validate()
        return {face: self.faces[face].semantic_frame for face in FACE_ORDER}
