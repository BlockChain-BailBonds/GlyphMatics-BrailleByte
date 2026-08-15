from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Concept:
    concept_id: str
    definition: str
    surfaces: dict[str, list[str]] = field(default_factory=dict)
    roles: list[str] = field(default_factory=list)
    provenance: str = "governed"


class ConceptRegistry:
    def __init__(self) -> None:
        self.concepts = {
            "BANK:FINANCIAL": Concept("BANK:FINANCIAL", "Financial institution", {"en": ["bank"], "es": ["banco"], "fr": ["banque"]}, ["entity"]),
            "BANK:RIVER_EDGE": Concept("BANK:RIVER_EDGE", "Edge of a river", {"en": ["bank"]}, ["location"]),
            "TRUST:RELY": Concept("TRUST:RELY", "Act of relying on something", {"en": ["trust"]}, ["relation"]),
            "TRUST:LEGAL_ENTITY": Concept("TRUST:LEGAL_ENTITY", "Legal property arrangement", {"en": ["trust"]}, ["entity"]),
            "SEM:ANIMAL:COW": Concept("SEM:ANIMAL:COW", "A cow", {"en": ["cow"], "es": ["vaca"], "fr": ["vache"], "de": ["kuh"], "it": ["mucca"], "nl": ["koe"], "ru": ["корова"], "ar": ["بقرة"], "hi": ["गाय"], "bn": ["গরু"], "ja": ["牛"], "zh": ["牛"]}, ["entity"]),
            "SEM:ACTION:MOVE": Concept("SEM:ACTION:MOVE", "Move action", {"en": ["move"]}, ["action"]),
            "SEM:ACTION:OPEN": Concept("SEM:ACTION:OPEN", "Open action", {"en": ["open"]}, ["action"]),
            "SEM:ATTRIBUTE:RED": Concept("SEM:ATTRIBUTE:RED", "Red attribute", {"en": ["red"]}, ["attribute"]),
            "SEM:ATTRIBUTE:LEFT": Concept("SEM:ATTRIBUTE:LEFT", "Left direction", {"en": ["left"]}, ["attribute"]),
            "SEM:ENTITY:ROBOT": Concept("SEM:ENTITY:ROBOT", "Robot entity", {"en": ["robot"]}, ["entity"]),
            "SEM:ENTITY:CUBE": Concept("SEM:ENTITY:CUBE", "Cube entity", {"en": ["cube"]}, ["entity"]),
            "SEM:ENTITY:DOOR": Concept("SEM:ENTITY:DOOR", "Door entity", {"en": ["door"]}, ["entity"]),
        }

    def resolve(self, surface: str) -> list[Concept]:
        s = surface.strip().lower()
        matches = []
        for concept in self.concepts.values():
            for forms in concept.surfaces.values():
                if s in [f.lower() for f in forms]:
                    matches.append(concept)
                    break
        return matches

    def get(self, concept_id: str) -> Concept:
        return self.concepts[concept_id]

    def to_json(self) -> str:
        return json.dumps({k: concept.__dict__ for k, concept in self.concepts.items()}, indent=2, ensure_ascii=False)

    @classmethod
    def from_file(cls, path: str | Path) -> "ConceptRegistry":
        self = cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.concepts = {cid: Concept(**data) for cid, data in raw.items()}
        return self
