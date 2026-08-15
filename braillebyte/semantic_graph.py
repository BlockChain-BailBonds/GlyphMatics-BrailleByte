from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .semantic import ConceptRegistry


START, END, CLAUSE, EXT = 1, 2, 3, 255
ROLES = {
    "AGENT": 32,
    "ACTION": 33,
    "PATIENT": 34,
    "ATTRIBUTE": 36,
    "LOCATION": 37,
    "RESULT": 197,
    "TIME": 198,
    "DESTINATION": 199,
}
BACK = {value: key for key, value in ROLES.items()}


def _put(value: int) -> tuple[int, ...]:
    out = []
    while True:
        byte, value = value & 127, value >> 7
        out.append(byte | (128 if value else 0))
        if not value:
            return tuple(out)


def _get(data: tuple[int, ...], pos: int) -> tuple[int, int]:
    value = shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, pos
        shift += 7
    raise ValueError("truncated concept reference")


@dataclass(frozen=True)
class SemanticGraph:
    roles: Mapping[str, str]
    evidence: tuple[str, ...] = ()

    def validate(self) -> None:
        if "ACTION" not in self.roles or set(self.roles) - set(ROLES):
            raise ValueError("graph requires ACTION and supported roles")


class SemanticGraphCodec:
    def __init__(self, registry: ConceptRegistry) -> None:
        self.registry = registry

    def parse(self, text: str) -> dict[str, object]:
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

    def encode(self, graph: SemanticGraph) -> tuple[int, ...]:
        graph.validate()
        out = [START, CLAUSE]
        for role in sorted(graph.roles, key=ROLES.get):
            concept = self.registry.get(graph.roles[role])
            if concept is None:
                raise KeyError(graph.roles[role])
            out.extend((ROLES[role], EXT, *_put(concept.id)))
        return tuple((*out, END))

    def decode(self, data: tuple[int, ...]) -> SemanticGraph:
        if data[:2] != (START, CLAUSE) or data[-1:] != (END,):
            raise ValueError("invalid graph frame")
        roles, pos = {}, 2
        while pos < len(data) - 1:
            role = BACK.get(data[pos])
            pos += 1
            if role is None or data[pos] != EXT:
                raise ValueError("invalid graph role")
            concept_id, pos = _get(data, pos + 1)
            concept = self.registry.by_id(concept_id)
            if concept is None or role in roles:
                raise ValueError("invalid graph concept")
            roles[role] = concept.identity
        graph = SemanticGraph(roles)
        graph.validate()
        return graph


FORMS = {
    "en": ("the cow eats food.",),
    "es": ("la vaca come comida.",),
    "zh-Hans": ("牛吃食物。",),
}


def realize(graph: SemanticGraph, language: str) -> str:
    expected = {"AGENT": "SEM:ANIMAL:COW", "ACTION": "SEM:ACTION:EAT", "PATIENT": "SEM:SUBSTANCE:FOOD"}
    if dict(graph.roles) != expected or language not in FORMS:
        raise ValueError("unsupported controlled realization")
    return FORMS[language][0]
