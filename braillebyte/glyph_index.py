from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GlyphIndex:
    vocabulary: dict[str, str] = field(default_factory=dict)
    architecture: dict[str, str] = field(default_factory=dict)
    chunk_graph: dict[str, tuple[str, int, int]] = field(default_factory=dict)
    integrity_graph: dict[str, str] = field(default_factory=dict)

    def manifest(self) -> dict[str, object]:
        return {
            "vocabulary": self.vocabulary,
            "architecture": self.architecture,
            "chunk_graph": self.chunk_graph,
            "integrity_graph": self.integrity_graph,
        }
