"""Lossless trained compression for BrailleByte semantic byte streams."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


DICT_REF = 251
RAW_ESCAPE = 252


@dataclass(frozen=True)
class BytePhrase:
    index: int
    bytes_: tuple[int, ...]
    frequency: int


class BrailleByteCompressor:
    """A trained dictionary compressor that never changes semantic bytes.

    The compressed transport uses DICT_REF + one index byte. Raw occurrences of
    either transport-reserved byte are escaped, so decoded output is exactly the
    canonical BrailleByte stream supplied by the semantic encoder.
    """

    def __init__(self, phrases: Iterable[BytePhrase] = ()) -> None:
        self.phrases = tuple(phrases)
        self._by_index = {phrase.index: phrase.bytes_ for phrase in self.phrases}
        if len(self._by_index) != len(self.phrases):
            raise ValueError('phrase indices must be unique')
        self._ordered = tuple(sorted(self.phrases, key=lambda phrase: (-len(phrase.bytes_), -phrase.frequency, phrase.index)))

    @classmethod
    def train(
        cls,
        streams: Iterable[Sequence[int]],
        *,
        min_frequency: int = 4,
        min_length: int = 3,
        max_length: int = 12,
        max_phrases: int = 127,
    ) -> 'BrailleByteCompressor':
        if min_length < 3:
            raise ValueError('phrases must be at least three bytes; references cost two bytes')
        counts: Counter[tuple[int, ...]] = Counter()
        for raw in streams:
            stream = tuple(int(value) for value in raw)
            if any(value < 0 or value > 255 for value in stream):
                raise ValueError('BrailleByte streams require bytes in 0..255')
            for length in range(min_length, min(max_length, len(stream)) + 1):
                for start in range(len(stream) - length + 1):
                    counts[stream[start:start + length]] += 1
        ranked = [
            (phrase, frequency)
            for phrase, frequency in counts.items()
            if frequency >= min_frequency and frequency * (len(phrase) - 2) > 0
        ]
        ranked.sort(key=lambda item: (-item[1] * (len(item[0]) - 2), -len(item[0]), item[0]))
        return cls(BytePhrase(index=index, bytes_=phrase, frequency=frequency) for index, (phrase, frequency) in enumerate(ranked[:max_phrases]))

    def compress(self, stream: Sequence[int]) -> tuple[int, ...]:
        source = tuple(int(value) for value in stream)
        result: list[int] = []
        cursor = 0
        while cursor < len(source):
            match = next((phrase for phrase in self._ordered if source[cursor:cursor + len(phrase.bytes_)] == phrase.bytes_), None)
            if match is not None:
                result.extend((DICT_REF, match.index))
                cursor += len(match.bytes_)
                continue
            value = source[cursor]
            result.extend((RAW_ESCAPE, value) if value in (DICT_REF, RAW_ESCAPE) else (value,))
            cursor += 1
        return tuple(result)

    def decompress(self, stream: Sequence[int]) -> tuple[int, ...]:
        source = tuple(int(value) for value in stream)
        result: list[int] = []
        cursor = 0
        while cursor < len(source):
            value = source[cursor]
            if value == DICT_REF:
                if cursor + 1 >= len(source) or source[cursor + 1] not in self._by_index:
                    raise ValueError('unknown or truncated BrailleByte dictionary reference')
                result.extend(self._by_index[source[cursor + 1]])
                cursor += 2
            elif value == RAW_ESCAPE:
                if cursor + 1 >= len(source) or source[cursor + 1] not in (DICT_REF, RAW_ESCAPE):
                    raise ValueError('invalid or truncated BrailleByte raw escape')
                result.append(source[cursor + 1])
                cursor += 2
            else:
                result.append(value)
                cursor += 1
        return tuple(result)

    @staticmethod
    def _cells(values: Sequence[int]) -> str:
        return ''.join(chr(0x2800 + int(value)) for value in values)

    @staticmethod
    def _bytes(cells: str) -> tuple[int, ...]:
        values = tuple(ord(cell) - 0x2800 for cell in cells)
        if any(value < 0 or value > 255 for value in values):
            raise ValueError('compressed transport must contain only 8-dot Braille cells')
        return values

    def compress_to_braille(self, stream: Sequence[int]) -> str:
        """Losslessly compress a byte stream into an 8-dot-Braille-only payload."""
        return self._cells(self.compress(stream))

    def decompress_braille(self, cells: str) -> tuple[int, ...]:
        """Recover the exact original bytes from an 8-dot-Braille-only payload."""
        return self.decompress(self._bytes(cells))

    def to_dict(self) -> dict[str, object]:
        return {
            'format': 'braillebyte-trained-phrase-compressor-v1',
            'dictionary_reference_byte': DICT_REF,
            'raw_escape_byte': RAW_ESCAPE,
            'phrases': [{'index': phrase.index, 'bytes': list(phrase.bytes_), 'frequency': phrase.frequency} for phrase in self.phrases],
        }
