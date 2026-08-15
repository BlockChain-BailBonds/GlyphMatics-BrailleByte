from braillebyte.codec import BrailleByteCodec


def test_round_trip():
    codec = BrailleByteCodec()
    text = "the cow goes moo"
    assert codec.decode(codec.encode(text)) == text
