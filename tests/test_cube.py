from braillebyte import BrailleByteCodec, GlyphCube, GlyphCubeFace, FACE_ORDER, RubiksGlyphCube


def test_cube_round_trip_bytes():
    codec = BrailleByteCodec()
    faces = {}
    for idx, face in enumerate(FACE_ORDER):
        faces[face] = GlyphCubeFace(face, bytes([idx, idx + 1]), {"role": face, "kind": "face", "confidence": 1.0})
    cube = GlyphCube(faces=faces)
    cells = codec.cube_to_bra8lle(cube)
    restored = codec.bra8lle_to_cube(cells)
    assert restored.as_bytes() == cube.as_bytes()
    assert restored.semantic_summary() == cube.semantic_summary()


def test_rubiks_inverse_restores_history():
    cube = RubiksGlyphCube.solved().apply(["R", "U", "F"])
    restored = cube.inverse()
    assert restored.history == []
    assert restored.as_bytes() == RubiksGlyphCube.solved().as_bytes()
