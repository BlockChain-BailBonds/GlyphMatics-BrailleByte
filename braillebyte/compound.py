from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bytecode import SemanticBytecode


@dataclass
class CompoundMeaning:
    pairs: list[tuple[str, str]]
    nested: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CompoundMeaningCodec:
    bytecode: SemanticBytecode

    def encode(self, meaning: CompoundMeaning) -> list[dict[str, Any]]:
        encoded = self.bytecode.encode_pairs(meaning.pairs)
        return [*encoded, *meaning.nested]
