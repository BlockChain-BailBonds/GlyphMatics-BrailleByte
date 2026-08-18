# SEMZIP v5 Shortcut Engine

This branch contains the SEMZIP v5 corpus-relative semantic delta compression engine developed from the GlyphMatics/BrailleByte compression work.

## Included artifact

`semzip-v5/semzip_v5_source_clean.tar.gz`

SHA-256:

`b43e7b2c9d15a5aba4344aed654f418f104cd19b89d04749ce4bcd07e37817bd`

The archive contains the complete v5 source-controlled package:

- `semzip/` engine source
- `tests/`
- `README.md`
- `V5_AUDIT.md`
- `pyproject.toml`
- `benchmark_sweep.py`
- `benchmark_results.json`
- `LICENSE`

Generated caches, compiled Python bytecode, temporary files, and sample `.semz` artifacts are excluded.

## v5 architecture

The common update path uses compact binary headers, RAW-body bypass for tiny deltas, same-layout `PATCH_ONLY` absolute-offset events, identity mode, prefix/suffix resynchronization, XOR residual reinjection, and deterministic GlyphMatics/BrailleByte color-lane dispersion.

## Validation

The local v5 validation run completed with 15 tests passing plus 6 subtests. The reported large compression ratios are corpus-relative transmitted-delta ratios: the decoder is assumed to already possess the referenced corpus/base object. They are not standalone arbitrary-file compression ratios.

A real `/bin/bash` ELF update test with three changed bytes produced a 110-byte transmitted container for a 1,298,416-byte target, approximately 11,803.78× corpus-relative transmitted ratio, with SHA-256 exact reconstruction.
