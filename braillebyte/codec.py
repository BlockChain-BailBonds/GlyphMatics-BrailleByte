from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import json
import re
import unicodedata

from .semantic import ConceptRegistry, Interpretation

BRAILLE_BASE = 0x2800


@dataclass(frozen=True)
class Token:
    byte: int
    name: str
    kind: str
    meaning: str

    @property
    def braille(self) -> str:
        return chr(BRAILLE_BASE + self.byte)

    @property
    def bits(self) -> str:
        return f"{self.byte:08b}"

    @property
    def dots(self) -> tuple[int, ...]:
        return tuple(i + 1 for i in range(8) if self.byte & (1 << i))


@dataclass(frozen=True)
class EncodingResult:
    source: str
    normalized: str
    bytes_: tuple[int, ...]
    braille: str
    spoken: tuple[str, ...]
    tokens: tuple[Token, ...]
    interpretations: tuple[Interpretation, ...]


class BrailleByteCodec:
    def __init__(self, dictionary_path: str | Path | None = None, concept_path: str | Path | None = None) -> None:
        if dictionary_path is None:
            dictionary_path = Path(__file__).resolve().parents[1] / "data" / "dot_dictionary.json"
        with Path(dictionary_path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        self.version = str(data["version"])
        self.dot_syllables = {int(k): str(v) for k, v in data["dot_syllables"].items()}
        self.tokens_by_byte = {
            int(item["byte"]): Token(
                byte=int(item["byte"]),
                name=str(item["name"]),
                kind=str(item["kind"]),
                meaning=str(item["meaning"]),
            )
            for item in data["tokens"]
        }
        self.lexicon = {str(k): tuple(int(v) for v in values) for k, values in data["lexicon"].items()}
        self.patterns = {str(k): tuple(int(v) for v in values) for k, values in data["patterns"].items()}
        self._phrases = sorted(self.lexicon, key=lambda p: (-len(p.split()), -len(p)))
        if concept_path is None:
            concept_path = Path(__file__).resolve().parents[1] / 'data' / 'concepts.json'
        self.concepts = ConceptRegistry.from_file(concept_path)

    @staticmethod
    def normalize(text: str) -> str:
        lowered = unicodedata.normalize('NFKC', text).casefold().strip()
        # Keep combining marks: they carry letters and vowels in scripts such as
        # Devanagari and Bengali, and must not be erased during tokenization.
        lowered = ''.join(
            char if char.isalnum() or unicodedata.category(char).startswith('M') or char.isspace() or char in "'-" else ' '
            for char in lowered
        )
        return re.sub(r"\s+", " ", lowered).strip()

    @staticmethod
    def byte_to_braille(value: int) -> str:
        if not 0 <= value <= 255:
            raise ValueError("BrailleByte values must be between 0 and 255")
        return chr(BRAILLE_BASE + value)

    @staticmethod
    def braille_to_byte(cell: str) -> int:
        if len(cell) != 1:
            raise ValueError("Expected exactly one Braille cell")
        value = ord(cell) - BRAILLE_BASE
        if not 0 <= value <= 255:
            raise ValueError(f"Character {cell!r} is not an 8-dot Unicode Braille pattern")
        return value

    def bytes_to_braille(self, values: Iterable[int]) -> str:
        return "".join(self.byte_to_braille(value) for value in values)

    def braille_to_bytes(self, text: str) -> tuple[int, ...]:
        return tuple(self.braille_to_byte(cell) for cell in text if not cell.isspace())

    def speak_byte(self, value: int) -> str:
        dots = [self.dot_syllables[i] for i in range(1, 9) if value & (1 << (i - 1))]
        return "blank" if not dots else "-".join(dots)

    def token_for(self, value: int) -> Token:
        return self.tokens_by_byte.get(value, Token(value, f"BYTE_{value:03d}", "unassigned", "unassigned byte"))

    def encode(self, text: str) -> EncodingResult:
        normalized = self.normalize(text)
        surfaces = self.concepts.segment(normalized)
        interpretations = tuple(self.concepts.interpret(surface) for surface in surfaces)
        if normalized in self.patterns:
            values = self.patterns[normalized]
        else:
            values = self._encode_lexically(surfaces, interpretations)
        tokens = tuple(self.token_for(value) for value in values)
        return EncodingResult(
            source=text,
            normalized=normalized,
            bytes_=tuple(values),
            braille=self.bytes_to_braille(values),
            spoken=tuple(self.speak_byte(value) for value in values),
            tokens=tokens,
            interpretations=interpretations,
        )

    def interpret(self, text: str) -> tuple[Interpretation, ...]:
        """Return resolved, ambiguous, and unknown forms without inventing a sense."""
        normalized = self.normalize(text)
        return tuple(self.concepts.interpret(surface) for surface in self.concepts.segment(normalized))

    @staticmethod
    def _literal_bytes(surface: str) -> list[int]:
        raw = surface.encode('utf-8')
        if len(raw) > 255:
            raise ValueError('Literal surface form exceeds the v0.2 one-byte length limit')
        return [240, len(raw), *raw]

    @staticmethod
    def _extended_concept_bytes(concept_id: int) -> list[int]:
        values = [255]
        while True:
            byte = concept_id & 0x7F
            concept_id >>= 7
            values.append(byte | (0x80 if concept_id else 0))
            if not concept_id:
                return values

    def _encode_lexically(self, words: Sequence[str], interpretations: Sequence[Interpretation]) -> tuple[int, ...]:
        values: list[int] = [1]
        for word, interpretation in zip(words, interpretations):
            concept = interpretation.resolved
            if concept and concept.bytes_:
                values.extend(concept.bytes_)
            elif concept:
                values.extend(self._extended_concept_bytes(concept.id))
            else:
                # Both unknown and ambiguous words retain their original UTF-8 form.
                values.extend([4, *self._literal_bytes(word)])
        values.append(2)
        return tuple(values)

    def decode_tokens(self, values: Sequence[int]) -> tuple[Token, ...]:
        return tuple(self.token_for(int(value)) for value in values)

    def explain(self, text: str) -> list[dict[str, object]]:
        result = self.encode(text)
        return [
            {
                "index": index,
                "byte": token.byte,
                "hex": f"0x{token.byte:02X}",
                "bits": token.bits,
                "braille": token.braille,
                "dots": token.dots,
                "spoken": result.spoken[index],
                "name": token.name,
                "kind": token.kind,
                "meaning": token.meaning,
            }
            for index, token in enumerate(result.tokens)
        ]
