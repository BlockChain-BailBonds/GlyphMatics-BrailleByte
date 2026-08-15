# GlyphMatics BrailleByte

BrailleByte is a semantic codec that maps canonical concepts and relations to 8-dot Braille byte cells.

The concept registry is provisional until the opcode table is locked.
Compound meanings are encoded as colored semantic pairs plus nested `BEGIN`/`END` blocks for causes, conditions, goals, and alternatives.
The compression engine now uses a graph-first prior, a residual layer for exact fallbacks, and Braille-only storage for the resulting bytes.

## Quick start

```bash
python -m braillebyte.cli encode "the cow goes moo"
python -m braillebyte.cli decode "⣿⢀⡁"
python -m braillebyte.cli explain "vaca bank"
python -m unittest discover -s tests -v
python scripts/benchmark_suite.py --input data/system_sentence_corpus.jsonl --output data/benchmark_suite_report.json
```
