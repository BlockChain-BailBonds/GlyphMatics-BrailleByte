# BrailleByte Spoken v1

BrailleByte Spoken is a compact constructed language for speaking supported BrailleByte semantic graphs. It is designed, not naturally evolved.

## Sound system

Open syllables use consonant-vowel pairs. `maku`, `nari`, and `savi` have regular stress on the first syllable. The byte-level protocol pronunciation remains available for exact transmission; this language is the semantic spoken layer.

## Grammar

Canonical order is **agent action patient**. Roles remain explicit in the semantic graph, so other languages can reorder the same meaning.

```text
maku nari savi
cow  eat  food
```

This parses to:

```text
AGENT:   SEM:ANIMAL:COW
ACTION:  SEM:ACTION:EAT
PATIENT: SEM:SUBSTANCE:FOOD
```

## Rules

- One lexical word maps to one immutable concept plus grammatical role.
- Unknown words and unsupported graphs are rejected, not guessed.
- Byte-level speech uses `braillebyte <cell> / <cell> end` for lossless transport.
- Semantic speech uses BrailleByte Spoken and round-trips through `SemanticGraph`.
