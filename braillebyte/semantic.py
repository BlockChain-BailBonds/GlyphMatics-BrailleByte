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
    id: int = 0

    @property
    def identity(self) -> str:
        return self.concept_id


class ConceptRegistry:
    def __init__(self) -> None:
        concepts = [
            Concept("BANK:FINANCIAL", "Financial institution", {"en": ["bank"], "es": ["banco"], "fr": ["banque"]}, ["entity"]),
            Concept("BANK:RIVER_EDGE", "Edge of a river", {"en": ["bank"]}, ["location"]),
            Concept("TRUST:RELY", "Act of relying on something", {"en": ["trust"]}, ["relation"]),
            Concept("TRUST:LEGAL_ENTITY", "Legal property arrangement", {"en": ["trust"]}, ["entity"]),
            Concept("SEM:ANIMAL:COW", "A cow", {"en": ["cow"], "es": ["vaca"], "fr": ["vache"], "de": ["kuh"], "it": ["mucca"], "nl": ["koe"], "ru": ["корова"], "ar": ["بقرة"], "hi": ["गाय"], "bn": ["গরু"], "ja": ["牛"], "zh": ["牛"]}, ["entity"]),
            Concept("SEM:ACTION:MOVE", "Move action", {"en": ["move"]}, ["action"]),
            Concept("SEM:ACTION:EAT", "Eat action", {"en": ["eat"], "es": ["come"], "zh": ["吃"]}, ["action"]),
            Concept("SEM:ACTION:OPEN", "Open action", {"en": ["open"]}, ["action"]),
            Concept("SEM:ATTRIBUTE:RED", "Red attribute", {"en": ["red"]}, ["attribute"]),
            Concept("SEM:ATTRIBUTE:LEFT", "Left direction", {"en": ["left"]}, ["attribute"]),
            Concept("SEM:ENTITY:ROBOT", "Robot entity", {"en": ["robot"]}, ["entity"]),
            Concept("SEM:ENTITY:CUBE", "Cube entity", {"en": ["cube"]}, ["entity"]),
            Concept("SEM:ENTITY:DOOR", "Door entity", {"en": ["door"]}, ["entity"]),
            Concept("SEM:SUBSTANCE:FOOD", "Food substance", {"en": ["food"], "es": ["comida"], "zh": ["食物"]}, ["substance"]),
        ]
        for idx, concept in enumerate(concepts, start=1):
            concept.id = idx
        self.concepts = {concept.concept_id: concept for concept in concepts}

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

    def by_id(self, concept_id: int) -> Concept:
        for concept in self.concepts.values():
            if concept.id == concept_id:
                return concept
        raise KeyError(concept_id)

    def to_json(self) -> str:
        return json.dumps({k: concept.__dict__ for k, concept in self.concepts.items()}, indent=2, ensure_ascii=False)

    @classmethod
    def from_file(cls, path: str | Path) -> "ConceptRegistry":
        self = cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.concepts = {cid: Concept(**data) for cid, data in raw.items()}
        return self
