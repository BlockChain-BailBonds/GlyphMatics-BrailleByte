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


def test_all_turns_round_trip_with_inverse():
    turns = ("R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'")
    for turn in turns:
        solved = RubiksGlyphCube.solved()
        inverse = {"R": "R'", "R'": "R", "L": "L'", "L'": "L", "U": "U'", "U'": "U", "D": "D'", "D'": "D", "F": "F'", "F'": "F", "B": "B'", "B'": "B"}[turn]
        cube = solved.rotate(turn).rotate(inverse)
        assert cube.as_bytes() == solved.as_bytes()
        assert cube.history == []


def test_all_turns_restore_solved_state_when_followed_by_inverse():
    turns = ("R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'")
    solved = RubiksGlyphCube.solved()
    for turn in turns:
        cube = solved.rotate(turn).rotate({"R": "R'", "R'": "R", "L": "L'", "L'": "L", "U": "U'", "U'": "U", "D": "D'", "D'": "D", "F": "F'", "F'": "F", "B": "B'", "B'": "B"}[turn])
        assert cube.as_bytes() == solved.as_bytes()
        assert cube.history == []


def test_sequence_inverse_restores_solved_state():
    solved = RubiksGlyphCube.solved()
    sequence = ["R", "U", "R'", "U'", "F", "B", "L", "D"]
    cube = solved.apply(sequence)
    restored = cube.inverse()
    assert restored.as_bytes() == solved.as_bytes()
    assert restored.history == []


def test_rubiks_orientation_partition_counts():
    cube = RubiksGlyphCube.solved()
    cube.validate()
    corners = [name for name in cube.facelets if name.split(":", 1)[1] in {"00", "02", "20", "22"}]
    edges = [name for name in cube.facelets if name.split(":", 1)[1] in {"01", "10", "12", "21"}]
    centers = [name for name in cube.facelets if name.split(":", 1)[1] == "11"]
    assert len(corners) == 24
    assert len(edges) == 24
    assert len(centers) == 6


def test_cubie_permutation_invariants_are_exposed():
    cube = RubiksGlyphCube.solved().apply(["R", "U", "F"])
    invariants = cube.cube_invariants()
    assert set(invariants) == {
        "corner_parity",
        "edge_parity",
        "corner_orientation_sum",
        "edge_orientation_sum",
        "is_solved",
    }
    permutation = cube.cubie_permutation()
    assert invariants["corner_parity"] == permutation.corner_parity()
    assert invariants["edge_parity"] == permutation.edge_parity()
    assert invariants["corner_orientation_sum"] == permutation.corner_orientation_sum()
    assert invariants["edge_orientation_sum"] == permutation.edge_orientation_sum()
    assert invariants["is_solved"] is False
