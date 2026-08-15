from braillebyte.codec import SemanticGraphCodec
from braillebyte.semantic import ConceptRegistry


def test_resolve_ambiguous_and_known():
    registry = ConceptRegistry()
    codec = SemanticGraphCodec(registry)
    graph = codec.parse("bank vaca")
    assert any("alternatives" in n for n in graph["nodes"])
    assert codec.realize(graph, "en")
