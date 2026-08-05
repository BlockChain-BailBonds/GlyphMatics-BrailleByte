from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import re
import unicodedata


@dataclass(frozen=True)
class Concept:
    id: int
    identity: str
    gloss: str
    bytes_: tuple[int, ...] | None
    forms: tuple[str, ...]


@dataclass(frozen=True)
class Interpretation:
    surface: str
    candidates: tuple[Concept, ...]

    @property
    def resolved(self) -> Concept | None:
        return self.candidates[0] if len(self.candidates) == 1 else None

    @property
    def status(self) -> str:
        if not self.candidates:
            return 'unknown'
        return 'resolved' if self.resolved else 'ambiguous'


class ConceptRegistry:
    """A versioned, language-neutral index of surface forms to concepts."""

    def __init__(self, concepts: Iterable[Concept], version: str) -> None:
        self.version = version
        self.concepts = tuple(concepts)
        self._forms: dict[str, list[Concept]] = {}
        for concept in self.concepts:
            for form in concept.forms:
                candidates = self._forms.setdefault(self.canonical_form(form), [])
                if all(existing.id != concept.id for existing in candidates):
                    candidates.append(concept)
        self._unspaced_forms = sorted(
            (form for form in self._forms if self._contains_cjk(form)),
            key=len,
            reverse=True,
        )

    @staticmethod
    def canonical_form(surface: str) -> str:
        normalized = unicodedata.normalize('NFKD', surface).casefold()
        return ''.join(char for char in normalized if not unicodedata.combining(char))

    @staticmethod
    def _contains_cjk(surface: str) -> bool:
        return any('\u3040' <= char <= '\u30ff' or '\u3400' <= char <= '\u9fff' for char in surface)

    @classmethod
    def from_file(cls, path: str | Path) -> 'ConceptRegistry':
        with Path(path).open('r', encoding='utf-8') as handle:
            data = json.load(handle)
        concepts = (
            Concept(
                id=int(item['id']),
                identity=str(item['identity']),
                gloss=str(item['gloss']),
                bytes_=tuple(int(value) for value in item['bytes']) if 'bytes' in item else None,
                forms=tuple(str(form) for form in item['forms']),
            )
            for item in data['concepts']
        )
        return cls(concepts, str(data['version']))

    def interpret(self, surface: str) -> Interpretation:
        return Interpretation(surface=surface, candidates=tuple(self._forms.get(self.canonical_form(surface), ())))

    def segment(self, text: str) -> tuple[str, ...]:
        """Tokenize whitespace-delimited text and known unspaced CJK concept forms."""
        if not self._contains_cjk(text):
            return tuple(text.split())
        segments: list[str] = []
        for part in text.split():
            if not self._contains_cjk(part) or self.canonical_form(part) in self._forms:
                segments.append(part)
                continue
            cursor = 0
            canonical = self.canonical_form(part)
            while cursor < len(canonical):
                match = next((form for form in self._unspaced_forms if canonical.startswith(form, cursor)), None)
                if match is None:
                    segments.append(part[cursor])
                    cursor += 1
                else:
                    segments.append(part[cursor:cursor + len(match)])
                    cursor += len(match)
        return tuple(segments)
