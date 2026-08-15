from __future__ import annotations

import argparse
import json
import sys

from .codec import BrailleByteCodec
from .cube import GlyphCube, GlyphCubeFace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="braillebyte")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode")
    enc.add_argument("text")

    dec = sub.add_parser("decode")
    dec.add_argument("cells")

    exp = sub.add_parser("explain")
    exp.add_argument("text")

    sp = sub.add_parser("speak")
    sp.add_argument("cells")

    hr = sub.add_parser("hear")
    hr.add_argument("speech")

    cu = sub.add_parser("cube-encode")
    cu.add_argument("spec", help="JSON object with face payloads and optional semantic frames")

    cd = sub.add_parser("cube-decode")
    cd.add_argument("cells")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    codec = BrailleByteCodec()
    if args.cmd == "encode":
        print(codec.encode(args.text))
    elif args.cmd == "decode":
        print(codec.decode(args.cells))
    elif args.cmd == "explain":
        print(json.dumps(codec.explain(args.text), indent=2, ensure_ascii=False))
    elif args.cmd == "speak":
        print("braillebyte " + " / ".join(args.cells) + " end")
    elif args.cmd == "hear":
        print(args.speech.replace("braillebyte ", "").replace(" end", "").replace(" / ", ""))
    elif args.cmd == "cube-encode":
        spec = json.loads(args.spec)
        faces = {}
        for name, meta in spec["faces"].items():
            faces[name] = GlyphCubeFace(
                name=name,
                payload=bytes.fromhex(meta.get("payload_hex", "")),
                semantic_frame=meta.get("semantic_frame", {}),
            )
        cube = GlyphCube(faces=faces)
        print(codec.cube_to_bra8lle(cube))
    elif args.cmd == "cube-decode":
        cube = codec.bra8lle_to_cube(args.cells)
        print(json.dumps(cube.semantic_summary(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
