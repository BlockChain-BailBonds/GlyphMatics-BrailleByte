# GlyphMatics BrailleByte

BrailleByte is an experimental semantic codec that represents designed vocabulary as sequences of Unicode 8-dot Braille cells. Each cell corresponds to one byte. The system separates:

- surface text
- normalized concepts
- semantic relations
- byte sequences
- Braille cells
- spoken dot names

## What it does

- Resolves registered surface forms from multiple languages to stable semantic concepts.
- Emits each semantic message as 8-dot Braille cells, ordinary bytes, and speakable dot names.
- Preserves unknown forms as reversible UTF-8 literals instead of replacing them with a guessed meaning.
- Preserves ambiguous forms as multiple candidate concepts; it does not silently choose a sense.
- Trains a lossless byte-phrase dictionary for repeated complete-system semantic messages.

## What it can do today

The included concept registry covers a deliberately small semantic vocabulary across English, Spanish, French, German, Italian, Dutch, Russian, Arabic, Hindi, Bengali, Japanese, and Chinese. It supports unspaced CJK terms registered in the concept graph.

The system-stream trainer generates complete-system messages with explicit component clauses for controller, network, storage, security, power, and telemetry. It trains only on canonical 8-bit BrailleByte semantic streams and verifies byte-for-byte recovery on held-out messages.

BrailleByte is a semantic transport prototype, not a replacement for multilingual speech recognition, production-grade word-sense disambiguation, or a model-weight compression format. The trained phrase dictionary compresses repeated semantic protocol structure; it does not reduce arbitrary neural-weight tensors by itself.

## Evidence-backed scope

This project does not claim arbitrary repository collapse, universal natural-language understanding, or universal human adoption. Its supported claims are deterministic byte/cell/speech round trips, lossless preservation of unknown forms, explicit ambiguity, registered-concept semantic graphs, and tested compression for its supplied corpus. `scripts/verify_glyphmatics_bridge.py` verifies a real GlyphMatics vocabulary's lossless glyph and binary round trips beside the same input's BrailleByte semantic transport.

## Quick start

```bash
python -m braillebyte.cli encode "the cow goes moo"
python -m braillebyte.cli decode "⣿⢀⡁⡂⡃⡄⡅"
python -m braillebyte.cli explain "the cow goes moo"
python -m unittest discover -s tests -v
```

Open `web/index.html` in a browser for the local demo.

### Multilingual interpretation

```bash
python -m braillebyte.cli explain 'vaca 牛 bank quasar'
python -m braillebyte.cli encode 'بقرة agua'
```

The first command reports resolved concepts, ambiguous alternatives, and unknown forms. For example, `vaca` and `牛` both resolve to `SEM:ANIMAL:COW`; `bank` remains ambiguous; and `quasar` is retained literally.

## Universal semantic-language usage

BrailleByte is universal at the **semantic transport** layer. It is not a new spoken language and it does not assume that source-language word order is the target-language word order.

```text
source text, speech, or glyphs
        ↓
candidate concepts and senses
        ↓
canonical role-labelled semantic graph
        ↓
BrailleByte 8-bit cell stream
        ↓
target-language text, speech, tactile cells, visual glyphs, or machine action
```

For example, these surface forms can share one immutable concept:

```text
cow / vaca / vache / корова / بقرة / गाय / 牛
                         ↓
                   SEM:ANIMAL:COW
```

Semantic roles are explicit, so a receiver can realize the graph in its own language:

```text
ACTION: GIVE
AGENT: PERSON_A
PATIENT: OBJECT
DESTINATION: PERSON_B
```

The receiver may choose a language-appropriate order without changing the semantic graph. When a form is unknown or has multiple senses, the protocol preserves the original form and candidate alternatives rather than inventing a universal meaning. `SemanticGraphCodec` now provides deterministic role-graph frames, and `realize()` supplies controlled `en`, `es`, and `zh-Hans` output for registered graphs. A production implementation still needs independently evaluated language recognition, word-sense disambiguation, cultural context, vocabulary governance, and broad target-language generation.

### Spoken protocol layer

Every cell now has a deterministic spoken form. A message is framed as `braillebyte <cell> / <cell> end`; dots are named by their fixed syllables, and `/` is a mandatory cell boundary. This is byte-recoverable speech, not a claim that every listener already understands the semantics.

```bash
python -m braillebyte.cli speak '⠁⡁⢀⠂'
python -m braillebyte.cli hear 'braillebyte ka / ka-ri / va / ta end'
```

### BrailleByte Spoken

BrailleByte Spoken is the project’s constructed semantic spoken language. Its first grammar is canonical agent-action-patient: `maku nari savi` means `cow eats food` and parses to the same `SemanticGraph` used by the byte protocol. See [`docs/BRAILLEBYTE_SPOKEN.md`](docs/BRAILLEBYTE_SPOKEN.md). It is a functioning designed language layer, not a claim of universal natural-language adoption.

## Encoding model

A BrailleByte message is a sequence of one-byte tokens displayed with Unicode Braille patterns U+2800..U+28FF.

Namespaces:

- `0x00-0x1F`: framing and control
- `0x20-0x3F`: grammar and relations
- `0x40-0x7F`: semantic primitives
- `0x80-0xBF`: entities and actions
- `0xC0-0xDF`: attributes, sound and modifiers
- `0xE0-0xEF`: dictionary references
- `0xF0-0xFF`: extension and literal escape

This repository ships with a versioned multilingual concept registry. Surface forms such as `cow`, `vaca`, `vache`, `корова`, `بقرة`, `गाय`, and `牛` resolve to `SEM:ANIMAL:COW` and therefore emit the same semantic bytes. The included registry covers English, Spanish, French, German, Italian, Dutch, Russian, Arabic, Hindi, Bengali, and Japanese/Chinese forms for its current concepts. It also segments registered unspaced CJK forms, so `牛吃水` resolves as cow → eat → water. An ambiguous form, such as `bank`, stays ambiguous and is encoded with `UNKNOWN` plus a reversible UTF-8 literal; it is never silently assigned a meaning. Unknown forms receive the same lossless fallback.

`data/concepts.json` is the governed concept registry. Each concept has an immutable identity, optional compact byte representation, and multilingual surface forms. Concepts without compact bytes use `EXTENSION` followed by a varuint concept ID.

## Trained 8-bit system-stream compression

`scripts/train_system_compression.py` trains a lossless phrase dictionary on complete-system semantic sentences, then validates it on held-out systems. The training corpus encodes each sentence as canonical 8-bit BrailleByte cells: role bytes, extension concept IDs, component clauses, and framing—not UTF-8 prose. The learned model replaces only repeated byte phrases and every compressed stream must expand byte-for-byte to its canonical semantic stream.

```bash
python scripts/train_system_compression.py
python -m unittest discover -s tests -v
```

Training writes these reproducible artifacts:

- `data/system_sentence_corpus.jsonl`: canonical complete-system byte streams.
- `data/system_compression_model.json`: learned byte-phrase dictionary.
- `data/system_compression_report.json`: held-out compression and exact-round-trip result.

## Four-graph chunk retrieval

`braillebyte/glyph_index.py` implements the enhancement layer for a future chunked model store. It uses four 8-bit BrailleByte graph-section markers:

```text
vocabulary graph   token ID → embedding/output shard
architecture graph model layer + component → tensor chunk(s)
chunk graph        chunk ID → URI + byte offset + length + codec
integrity graph    chunk ID → SHA-256 verification
```

Use it to create or inspect a manifest:

```bash
python scripts/build_chunk_graph_example.py
python -m unittest discover -s tests -v
```

The generated `data/four_graph_chunk_index.example.json` is a validated layout example only; it contains no model weights. A real 30B deployment must generate chunk hashes from the actual GGUF or safetensors payload and provide the matching model architecture and tokenizer metadata.

## Project layout

```text
braillebyte/codec.py          text, concepts, Braille cells, and protocol bytes
braillebyte/semantic.py       versioned concept registry and multilingual segmentation
braillebyte/compression.py    lossless trained byte-phrase compressor
braillebyte/glyph_index.py    four-graph token and tensor chunk routing
data/concepts.json            governed concept records and multilingual forms
scripts/train_system_compression.py
tests/                        codec and exact-round-trip validation
```
