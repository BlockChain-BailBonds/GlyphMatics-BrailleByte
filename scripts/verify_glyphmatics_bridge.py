"""Verify lossless GlyphMatics transport alongside BrailleByte semantic transport."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from braillebyte import BrailleByteCodec

def main() -> int:
 parser = argparse.ArgumentParser()
 parser.add_argument('--glyphmatics-root', required=True)
 parser.add_argument('--vocabulary', required=True)
 parser.add_argument('--text', required=True)
 args = parser.parse_args()
 sys.path.insert(0, str(Path(args.glyphmatics_root) / 'src'))
 from glyphmatics.semantic_codec import SemanticVocabulary
 vocabulary = SemanticVocabulary.load(args.vocabulary)
 glyphs = vocabulary.encode_glyphs(args.text)
 binary = vocabulary.encode_binary(args.text)
 if vocabulary.decode_glyphs(glyphs) != args.text or vocabulary.decode_binary(binary) != args.text:
  raise RuntimeError('GlyphMatics lossless round trip failed')
 braille = BrailleByteCodec().encode(args.text)
 print(json.dumps({'glyphmatics': {'glyph_round_trip': True, 'binary_round_trip': True, 'vocabulary_sha256': vocabulary.digest, 'binary_bytes': len(binary)}, 'braillebyte': {'semantic_bytes': list(braille.bytes_), 'cells': braille.braille, 'interpretations': [item.status for item in braille.interpretations]}}, ensure_ascii=False, indent=2))
 return 0
if __name__ == '__main__': raise SystemExit(main())
