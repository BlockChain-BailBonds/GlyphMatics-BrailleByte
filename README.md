# GlyphMatics BrailleByte

BrailleByte is a semantic codec that maps canonical concepts and relations to 8-dot Braille byte cells.

The concept registry is provisional until the opcode table is locked.
Compound meanings are encoded as colored semantic pairs plus nested `BEGIN`/`END` blocks for causes, conditions, goals, and alternatives.

## Quick start

```bash
python -m braillebyte.cli encode "the cow goes moo"
python -m braillebyte.cli decode "⣿⢀⡁"
python -m braillebyte.cli explain "vaca bank"
python -m unittest discover -s tests -v
```
