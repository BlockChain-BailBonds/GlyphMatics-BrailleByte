from braillebyte.codec import BrailleByteCodec


def test_compound_encoding_uses_pairs_and_nested_blocks():
    codec = BrailleByteCodec()
    payload = codec.encode_compound(
        [("AGENT", "SEM:ENTITY:ROBOT"), ("PATIENT", "SEM:ENTITY:CUBE")],
        [{"op": "BEGIN", "kind": "CAUSE"}, {"op": "END", "kind": "CAUSE"}],
    )
    assert payload["provisional_registry"] is True
    assert payload["opcode_table"]["BEGIN"] == 0x01
    assert payload["compound"][0]["op"] == "PAIR"
    recovered = codec.decode_compound(payload["braille"])
    assert recovered["compound"] == payload["compound"]
