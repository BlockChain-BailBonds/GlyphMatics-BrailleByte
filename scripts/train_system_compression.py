from __future__ import annotations

from braillebyte.compression import BrailleByteCompressor
from braillebyte.codec import BrailleByteCodec


def main() -> None:
    corpus = ["controller online", "controller online", "storage nominal"]
    compressor = BrailleByteCompressor(BrailleByteCodec())
    print(compressor.model_json(corpus))


if __name__ == "__main__":
    main()
