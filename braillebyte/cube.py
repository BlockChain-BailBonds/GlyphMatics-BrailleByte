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


@dataclass(frozen=True)
class CornerCubie:
    name: str
    stickers: tuple[str, str, str]
    orientation: int


@dataclass(frozen=True)
class EdgeCubie:
    name: str
    stickers: tuple[str, str]
    orientation: int


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
        seen = set()
        for name in FACELET_ORDER:
            if name in seen:
                raise ValueError(f"duplicate facelet: {name}")
            seen.add(name)
            self._validate_facelet_orientation(name, self.facelets[name])
        self.cubie_state()

    def _validate_facelet_orientation(self, name: str, facelet: Facelet) -> None:
        expected = self._state_to_name(*self._sticker_state(name))
        if expected != name:
            raise ValueError(f"invalid facelet orientation: {name} -> {expected}")
        face = name.split(":", 1)[0]
        row = int(name.split(":", 1)[1][0])
        col = int(name.split(":", 1)[1][1])
        if (row, col) == (1, 1):
            return
        if row in {0, 2} and col in {0, 2}:
            return
        if row == 1 or col == 1:
            return
        raise ValueError(f"invalid edge/corner placement: {name}")

    def cubie_state(self) -> dict[str, list[CornerCubie | EdgeCubie]]:
        corners = []
        edges = []
        corner_defs = {
            "front:00": ("front", "left", "top"),
            "front:02": ("front", "right", "top"),
            "front:20": ("front", "left", "bottom"),
            "front:22": ("front", "right", "bottom"),
            "back:00": ("back", "right", "top"),
            "back:02": ("back", "left", "top"),
            "back:20": ("back", "right", "bottom"),
            "back:22": ("back", "left", "bottom"),
        }
        edge_defs = {
            "front:01": ("front", "top"),
            "front:10": ("front", "left"),
            "front:12": ("front", "right"),
            "front:21": ("front", "bottom"),
            "back:01": ("back", "top"),
            "back:10": ("back", "right"),
            "back:12": ("back", "left"),
            "back:21": ("back", "bottom"),
            "left:01": ("left", "top"),
            "left:12": ("left", "front"),
            "left:10": ("left", "back"),
            "left:21": ("left", "bottom"),
            "right:01": ("right", "top"),
            "right:12": ("right", "back"),
            "right:10": ("right", "front"),
            "right:21": ("right", "bottom"),
            "top:01": ("top", "back"),
            "top:10": ("top", "left"),
            "top:12": ("top", "right"),
            "top:21": ("top", "front"),
            "bottom:01": ("bottom", "front"),
            "bottom:10": ("bottom", "left"),
            "bottom:12": ("bottom", "right"),
            "bottom:21": ("bottom", "back"),
        }
        for anchor, faces in corner_defs.items():
            stickers = tuple(sorted([anchor, *self._other_corner_stickers(anchor)]))
            corners.append(CornerCubie(name=anchor, stickers=stickers, orientation=0 if anchor.startswith("front") or anchor.startswith("back") else 1))
        for anchor, faces in edge_defs.items():
            stickers = tuple(sorted([anchor, *self._other_edge_sticker(anchor)]))
            edges.append(EdgeCubie(name=anchor, stickers=stickers, orientation=0 if anchor.startswith("front") or anchor.startswith("back") else 1))
        if len(corners) != 8 or len(edges) != 24:
            raise ValueError("invalid cubie state")
        return {"corners": corners, "edges": edges}

    def _other_corner_stickers(self, name: str) -> tuple[str, str]:
        face, rc = name.split(":", 1)
        if face in {"front", "back"}:
            return (f"{'left' if rc[1] == '0' else 'right'}:{'0' if rc[0] == '0' else '2'}",
                    f"{'top' if rc[0] == '0' else 'bottom'}:{'0' if rc[1] == '0' else '2'}")
        return (name, name)

    def _other_edge_sticker(self, name: str) -> tuple[str]:
        face, rc = name.split(":", 1)
        if face in {"front", "back"}:
            return (f"{'top' if rc[0] == '0' else 'bottom'}:{'1' if rc[1] == '1' else '0'}",)
        return (name,)

    def rotate(self, turn: str) -> "RubiksGlyphCube":
        if turn not in {"R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"}:
            raise ValueError(f"unsupported turn: {turn}")
        if turn in TURN_INVERSES and self.history and self.history[-1] == TURN_INVERSES[turn]:
            history = self.history[:-1]
        else:
            history = [*self.history, turn]
        step = self._turn_sign(turn)
        axis = turn[0]
        transformed: dict[str, Facelet] = {}
        for name, facelet in self.facelets.items():
            pos, normal = self._sticker_state(name)
            if self._in_turn_layer(axis, pos):
                pos, normal = self._rotate_state(axis, pos, normal, step)
            transformed[self._state_to_name(pos, normal)] = Facelet(
                name=self._state_to_name(pos, normal),
                payload=facelet.payload,
                semantic_frame=dict(facelet.semantic_frame),
            )
        return RubiksGlyphCube(facelets=transformed, history=history)

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

    def _turn_sign(self, turn: str) -> int:
        return -1 if turn.endswith("'") else 1

    def _in_turn_layer(self, axis: str, pos: tuple[int, int, int]) -> bool:
        x, y, z = pos
        return (axis == "R" and x == 1) or (axis == "L" and x == -1) or (axis == "U" and y == 1) or (axis == "D" and y == -1) or (axis == "F" and z == 1) or (axis == "B" and z == -1)

    def _sticker_state(self, name: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        face, rc = name.split(":", 1)
        row = int(rc[0])
        col = int(rc[1])
        if face == "front":
            return (col - 1, 1 - row, 1), (0, 0, 1)
        if face == "back":
            return (1 - col, 1 - row, -1), (0, 0, -1)
        if face == "right":
            return (1, 1 - row, 1 - col), (1, 0, 0)
        if face == "left":
            return (-1, 1 - row, col - 1), (-1, 0, 0)
        if face == "top":
            return (col - 1, 1, row - 1), (0, 1, 0)
        if face == "bottom":
            return (col - 1, -1, 1 - row), (0, -1, 0)
        raise ValueError(f"unknown facelet: {name}")

    def _state_to_name(self, pos: tuple[int, int, int], normal: tuple[int, int, int]) -> str:
        x, y, z = pos
        nx, ny, nz = normal
        if nz == 1:
            return f"front:{1 - y}{x + 1}"
        if nz == -1:
            return f"back:{1 - y}{1 - x}"
        if nx == 1:
            return f"right:{1 - y}{1 - z}"
        if nx == -1:
            return f"left:{1 - y}{z + 1}"
        if ny == 1:
            return f"top:{z + 1}{x + 1}"
        if ny == -1:
            return f"bottom:{1 - z}{x + 1}"
        raise ValueError(f"invalid sticker state: {pos} {normal}")

    def _rotate_state(self, axis: str, pos: tuple[int, int, int], normal: tuple[int, int, int], step: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        def rotate_once(p: tuple[int, int, int], n: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
            x, y, z = p
            nx, ny, nz = n
            if axis == "R":
                return (x, -z, y), (nx, -nz, ny)
            if axis == "L":
                return (x, z, -y), (nx, nz, -ny)
            if axis == "U":
                return (z, y, -x), (nz, ny, -nx)
            if axis == "D":
                return (-z, y, x), (-nz, ny, nx)
            if axis == "F":
                return (-y, x, z), (-ny, nx, nz)
            if axis == "B":
                return (y, -x, z), (ny, -nx, nz)
            raise ValueError(axis)
        p, n = pos, normal
        for _ in range(abs(step)):
            p, n = rotate_once(p, n)
        return p, n

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
