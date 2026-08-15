from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

from .codec import BrailleByteCodec


@dataclass
class BrailleByteCompressor:
    codec: BrailleByteCodec

    def train(self, corpus: list[str]) -> dict[str, int]:
        counts = Counter(corpus)
        return dict(counts)

    def compress_to_braille(self, text: str) -> str:
        return self.codec.encode(text)

    def decompress_braille(self, cells: str) -> str:
        return self.codec.decode(cells)

    def model_json(self, corpus: list[str]) -> str:
        return json.dumps({"format": "BrailleByteCompressionModel", "phrases": self.train(corpus)}, indent=2, ensure_ascii=False)
