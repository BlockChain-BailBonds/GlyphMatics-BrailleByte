"""Deterministic spoken framing for 8-dot BrailleByte streams."""
from __future__ import annotations
from typing import Iterable

START_WORD, END_WORD, CELL_SEPARATOR = 'braillebyte', 'end', '/'

class SpokenBrailleByte:
    def __init__(self, dot_syllables: dict[int, str]) -> None:
        self.dot_syllables = dot_syllables
        self._dots = {syllable: dot for dot, syllable in dot_syllables.items()}

    def speak_byte(self, value: int) -> str:
        if not 0 <= value <= 255: raise ValueError('byte outside 0..255')
        return 'blank' if value == 0 else '-'.join(self.dot_syllables[dot] for dot in range(1, 9) if value & (1 << (dot - 1)))

    def hear_byte(self, phrase: str) -> int:
        if phrase == 'blank': return 0
        value = 0
        for syllable in phrase.split('-'):
            if syllable not in self._dots or value & (1 << (self._dots[syllable] - 1)): raise ValueError(f'invalid cell pronunciation: {phrase}')
            value |= 1 << (self._dots[syllable] - 1)
        return value

    def speak(self, values: Iterable[int]) -> str:
        return f"{START_WORD} {f' {CELL_SEPARATOR} '.join(self.speak_byte(value) for value in values)} {END_WORD}"

    def hear(self, utterance: str) -> tuple[int, ...]:
        words = utterance.strip().split()
        if len(words) < 2 or words[0] != START_WORD or words[-1] != END_WORD: raise ValueError('spoken stream requires braillebyte ... end framing')
        cells = ' '.join(words[1:-1]).split(f' {CELL_SEPARATOR} ')
        return tuple(self.hear_byte(cell) for cell in cells if cell)
