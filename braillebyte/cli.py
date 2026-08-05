from __future__ import annotations

import argparse
import json
from .codec import BrailleByteCodec
from .spoken import SpokenBrailleByte


def main() -> None:
    parser = argparse.ArgumentParser(prog="braillebyte", description="GlyphMatics BrailleByte semantic codec")
    sub = parser.add_subparsers(dest="command", required=True)

    encode_parser = sub.add_parser("encode", help="encode text into BrailleByte")
    encode_parser.add_argument("text")

    decode_parser = sub.add_parser("decode", help="decode Braille cells into token metadata")
    decode_parser.add_argument("braille")

    explain_parser = sub.add_parser("explain", help="show the complete encoding trace")
    explain_parser.add_argument("text")

    dictionary_parser = sub.add_parser("dictionary", help="print the token dictionary")
    dictionary_parser.add_argument("--kind", default=None)
    speak_parser = sub.add_parser('speak', help='render Braille cells as framed deterministic speech')
    speak_parser.add_argument('braille')
    hear_parser = sub.add_parser('hear', help='parse framed deterministic speech into Braille cells')
    hear_parser.add_argument('utterance')

    args = parser.parse_args()
    codec = BrailleByteCodec()
    spoken = SpokenBrailleByte(codec.dot_syllables)

    if args.command == "encode":
        result = codec.encode(args.text)
        print(result.braille)
        print("bytes:", " ".join(f"{value:02X}" for value in result.bytes_))
        print("spoken:", " | ".join(result.spoken))
    elif args.command == "decode":
        values = codec.braille_to_bytes(args.braille)
        rows = [
            {
                "byte": value,
                "hex": f"0x{value:02X}",
                "braille": codec.byte_to_braille(value),
                "spoken": codec.speak_byte(value),
                "token": codec.token_for(value).__dict__,
            }
            for value in values
        ]
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    elif args.command == "explain":
        result = codec.encode(args.text)
        print(json.dumps({
            'encoding': codec.explain(args.text),
            'interpretations': [
                {
                    'surface': item.surface,
                    'status': item.status,
                    'candidates': [
                        {'id': concept.id, 'identity': concept.identity, 'gloss': concept.gloss}
                        for concept in item.candidates
                    ],
                }
                for item in result.interpretations
            ],
        }, indent=2, ensure_ascii=False))
    elif args.command == "dictionary":
        rows = [token.__dict__ for token in codec.tokens_by_byte.values() if args.kind is None or token.kind == args.kind]
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    elif args.command == 'speak':
        print(spoken.speak(codec.braille_to_bytes(args.braille)))
    elif args.command == 'hear':
        print(codec.bytes_to_braille(spoken.hear(args.utterance)))


if __name__ == "__main__":
    main()
