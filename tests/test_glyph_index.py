import unittest
from hashlib import sha256

from braillebyte.glyph_index import (
    ARCHITECTURE_GRAPH,
    CHUNK_GRAPH,
    INTEGRITY_GRAPH,
    VOCABULARY_GRAPH,
    ChunkRecord,
    GlyphChunkIndex,
    TensorRoute,
    VocabularyShard,
)


class GlyphChunkIndexTests(unittest.TestCase):
    def setUp(self):
        self.payload = b'chunk-a'
        self.index = GlyphChunkIndex(
            model_id='test-model',
            chunks=(
                ChunkRecord('embed', 'embed.zst', 0, 8, sha256(self.payload).hexdigest()),
                ChunkRecord('output', 'output.zst', 8, 8, sha256(b'chunk-b').hexdigest()),
                ChunkRecord('layer0', 'layer0.zst', 16, 8, sha256(b'chunk-c').hexdigest()),
            ),
            vocabulary=(VocabularyShard(0, 99, 'embed', 'output'),),
            tensors=(TensorRoute(0, 'attention', ('layer0',)),),
        )

    def test_token_routing_and_glyph_sections(self):
        self.assertEqual(self.index.route_tokens((4, 9))[0].chunk_id, 'embed')
        glyphs = self.index.token_route_glyphs((4, 9))
        for section in (VOCABULARY_GRAPH, ARCHITECTURE_GRAPH, CHUNK_GRAPH, INTEGRITY_GRAPH):
            self.assertIn(section, glyphs)

    def test_tensor_route_and_integrity_check(self):
        self.assertEqual(self.index.route_tensor(0, 'attention')[0].chunk_id, 'layer0')
        self.assertTrue(self.index.verify_chunk('embed', self.payload))
        self.assertFalse(self.index.verify_chunk('embed', b'altered'))


if __name__ == '__main__':
    unittest.main()
