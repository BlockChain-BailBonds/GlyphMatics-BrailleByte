from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


FACE_ORDER = ("front", "right", "left", "top", "bottom", "back")
FACE_INDEX = {face: index for index, face in enumerate(FACE_ORDER)}
FACELET_COUNT = 54
FACELET_ORDER = tuple(f"{face}:{row}{col}" for face in FACE_ORDER for row in range(3) for col in range(3))
TURN_INVERSES = {"R": "R'", "R'": "R", "L": "L'", "L'": "L", "U": "U'", "U'": "U", "D": "D'", "D'": "D", "F": "F'", "F'": "F", "B": "B'", "B'": "B"}
FACELET_TO_FACE = {name: name.split(":", 1)[0] for name in FACELET_ORDER}

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


@dataclass(frozen=True)
class Facelet:
    name: str
    payload: bytes = b""
    semantic_frame: dict[str, Any] = field(default_factory=dict)


@dataclass
class RubiksGlyphCube:
    facelets: dict[str, Facelet]
    history: list[str] = field(default_factory=list)

    @classmethod
    def solved(cls) -> "RubiksGlyphCube":
        facelets = {name: Facelet(name=name, payload=b"", semantic_frame={"face": FACELET_TO_FACE[name], "position": name}) for name in FACELET_ORDER}
        return cls(facelets=facelets)

    def validate(self) -> None:
        missing = [name for name in FACELET_ORDER if name not in self.facelets]
        if missing:
            raise ValueError(f"missing facelets: {', '.join(missing)}")

    def rotate(self, turn: str) -> "RubiksGlyphCube":
        if turn not in {"R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"}:
            raise ValueError(f"unsupported turn: {turn}")
        if turn in TURN_INVERSES and self.history and self.history[-1] == TURN_INVERSES[turn]:
            history = self.history[:-1]
        else:
            history = [*self.history, turn]
        facelets = dict(self.facelets)
        if turn in {"R", "R'"}:
            facelets = self._cycle(facelets, ("top:02", "front:02", "bottom:02", "back:20"), turn == "R'")
        elif turn in {"L", "L'"}:
            facelets = self._cycle(facelets, ("top:00", "back:22", "bottom:00", "front:00"), turn == "L'")
        elif turn in {"U", "U'"}:
            facelets = self._cycle(facelets, ("back:00", "right:00", "front:00", "left:00"), turn == "U'")
        elif turn in {"D", "D'"}:
            facelets = self._cycle(facelets, ("front:20", "right:20", "back:20", "left:20"), turn == "D'")
        elif turn in {"F", "F'"}:
            facelets = self._cycle(facelets, ("top:20", "left:22", "bottom:02", "right:00"), turn == "F'")
        elif turn in {"B", "B'"}:
            facelets = self._cycle(facelets, ("top:00", "right:22", "bottom:20", "left:02"), turn == "B'")
        return RubiksGlyphCube(facelets=facelets, history=history)

    def apply(self, turns: list[str]) -> "RubiksGlyphCube":
        cube = self
        for turn in turns:
            cube = cube.rotate(turn)
        return cube

    def inverse(self) -> "RubiksGlyphCube":
        cube = self
        for turn in reversed(self.history):
            cube = cube.rotate(TURN_INVERSES[turn])
        return cube

    def _cycle(self, facelets: dict[str, Facelet], names: tuple[str, str, str, str], reverse: bool) -> dict[str, Facelet]:
        a, b, c, d = names
        if reverse:
            facelets[a], facelets[b], facelets[c], facelets[d] = facelets[b], facelets[c], facelets[d], facelets[a]
        else:
            facelets[a], facelets[b], facelets[c], facelets[d] = facelets[d], facelets[a], facelets[b], facelets[c]
        return facelets

    def semantic_summary(self) -> dict[str, Any]:
        return {name: self.facelets[name].semantic_frame for name in FACELET_ORDER}

    def to_legacy_faces(self) -> dict[str, GlyphCubeFace]:
        faces = {}
        for face in FACE_ORDER:
            facelets = [self.facelets[f"{face}:{row}{col}"] for row in range(3) for col in range(3)]
            payload = b"".join(item.payload[:1] or b"\x00" for item in facelets)
            semantic_frame = dict(facelets[0].semantic_frame) if facelets else {}
            faces[face] = GlyphCubeFace(name=face, payload=payload, semantic_frame=semantic_frame)
        return faces

    @classmethod
    def from_legacy_faces(cls, faces: dict[str, GlyphCubeFace]) -> "RubiksGlyphCube":
        facelets: dict[str, Facelet] = {}
        for face in FACE_ORDER:
            source = faces[face]
            for row in range(3):
                for col in range(3):
                    name = f"{face}:{row}{col}"
                    payload = source.payload[:1] if source.payload else b""
                    facelets[name] = Facelet(name=name, payload=payload, semantic_frame=dict(source.semantic_frame))
        return cls(facelets=facelets)

    def to_bytes(self) -> bytes:
        self.validate()
        out = bytearray()
        out.extend(b"RGC1")
        out.append(len(self.history) & 0xFF)
        for turn in self.history:
            out.extend(self._encode_text(turn))
        for name in FACELET_ORDER:
            facelet = self.facelets[name]
            out.extend(self._encode_text(name))
            out.extend(self._pack_u16(len(facelet.payload)))
            out.extend(facelet.payload)
            frame = self._encode_frame(facelet.semantic_frame)
            out.extend(self._pack_u16(len(frame)))
            out.extend(frame)
        return bytes(out)

    def as_bytes(self) -> bytes:
        return self.to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "RubiksGlyphCube":
        if not data.startswith(b"RGC1"):
            raise ValueError("unsupported cube format")
        pos = 4
        history_count = data[pos]
        pos += 1
        history = []
        for _ in range(history_count):
            turn, pos = cls._decode_text(data, pos)
            history.append(turn)
        facelets: dict[str, Facelet] = {}
        for _ in range(FACELET_COUNT):
            name, pos = cls._decode_text(data, pos)
            payload_len, pos = cls._unpack_u16_static(data, pos)
            payload = data[pos:pos + payload_len]
            pos += payload_len
            frame_len, pos = cls._unpack_u16_static(data, pos)
            frame, _ = cls._decode_frame_bytes_static(data[pos:pos + frame_len])
            pos += frame_len
            facelets[name] = Facelet(name=name, payload=payload, semantic_frame=frame)
        return cls(facelets=facelets, history=history)

    def _pack_u16(self, value: int) -> bytes:
        if not 0 <= value <= 0xFFFF:
            raise ValueError("value outside u16 range")
        return value.to_bytes(2, "big")

    @staticmethod
    def _unpack_u16_static(data: bytes, pos: int) -> tuple[int, int]:
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

    @staticmethod
    def _decode_frame_bytes_static(data: bytes) -> tuple[dict[str, Any], int]:
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
            if key is not None:
                frame[key] = RubiksGlyphCube._decode_value_static(value_type, raw)
        return frame, pos

    def _encode_text(self, text: str) -> bytes:
        data = text.encode("utf-8")
        if len(data) > 255:
            raise ValueError("text too long")
        return bytes([len(data)]) + data

    @staticmethod
    def _decode_text(data: bytes, pos: int) -> tuple[str, int]:
        length = data[pos]
        pos += 1
        text = data[pos:pos + length].decode("utf-8")
        return text, pos + length

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


@dataclass
class GlyphCube:
    faces: dict[str, GlyphCubeFace]
    history: list[str] = field(default_factory=list)

    def validate(self) -> None:
        missing = [face for face in FACE_ORDER if face not in self.faces]
        if missing:
            raise ValueError(f"missing cube faces: {', '.join(missing)}")

    def as_rubiks(self) -> RubiksGlyphCube:
        return RubiksGlyphCube.from_legacy_faces(self.faces)

    def rotate(self, turn: str) -> "GlyphCube":
        rubiks = self.as_rubiks().rotate(turn)
        return GlyphCube(faces=rubiks.to_legacy_faces(), history=rubiks.history)

    def apply(self, turns: list[str]) -> "GlyphCube":
        return self.as_rubiks().apply(turns).to_legacy_cube()

    def inverse(self) -> "GlyphCube":
        return self.as_rubiks().inverse().to_legacy_cube()

    def to_legacy_cube(self) -> "GlyphCube":
        return GlyphCube(faces=self.as_rubiks().to_legacy_faces(), history=list(self.history))

    def to_bytes(self) -> bytes:
        self.validate()
        return self.as_rubiks().to_bytes()

    def as_bytes(self) -> bytes:
        return self.to_bytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "GlyphCube":
        rubiks = RubiksGlyphCube.from_bytes(data)
        return cls(faces=rubiks.to_legacy_faces(), history=rubiks.history)

    def semantic_summary(self) -> dict[str, Any]:
        self.validate()
        return {face: self.faces[face].semantic_frame for face in FACE_ORDER}


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
