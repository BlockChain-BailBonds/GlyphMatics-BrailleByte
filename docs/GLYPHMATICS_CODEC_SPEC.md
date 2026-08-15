# GlyphMatics BrailleByte Codec Spec

## Layers

1. Semantic layer
- Inputs are normalized into concept graphs, role pairs, and explicit alternatives.
- Supported meanings use canonical concept IDs from the registry.

2. Prior layer
- Shared AI priors predict the likely concept graph or semantic structure.
- The codec stores the prediction header and exact residual, never the prediction alone.

3. Residual layer
- Residuals preserve anything not covered by the shared prior.
- Unknown or unsupported data falls back to exact literal storage.

4. Bytecode layer
- Semantic blocks are serialized into deterministic opcode streams.
- `BEGIN` / `END` blocks mark causes, conditions, goals, alternatives, and other nested scopes.

5. Braille transport layer
- Every stored byte is mapped to a single 8-dot Braille cell.
- Braille cells are the transport alphabet, not a lossy visualization.

## Losslessness

- Exact round-trip is required for supported graphs, text, and raw byte payloads.
- Literal escape is mandatory for any unsupported form.
- The codec must never guess a meaning when the registry cannot resolve it.

## Compression contract

- The compressor may use shared priors, shared registry terms, and repeated phrases.
- Compression ratios are corpus dependent.
- There is no blanket guarantee of high compression on arbitrary high-entropy inputs.

## Reference path

text
-> semantic graph
-> prior prediction
-> residual
-> bytecode
-> Braille cells

