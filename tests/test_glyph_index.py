import unittest
from hashlib import sha256

from braillebyte.glyph_index import (
    ARCHITECTURE_GRAPH,
    CHUNK_GRAPH,
    INTEGRITY_GRAPH,
    VOCABULARY_GRAPH,
    ChunkRecord,
    GlyphChunkIndex,
    RubiksCheckpointManifest,
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

    def test_manifest_cube_round_trip(self):
        manifest = RubiksCheckpointManifest(
            model_id='test-model',
            architecture_id='arch-v1',
            tokenizer_id='tok-v1',
            quantization_scheme='q4',
            chunk_index=self.index,
            reconstruction_order=('embed', 'output', 'layer0'),
        )
        cube = manifest.build_cube()
        restored = RubiksCheckpointManifest.from_cube(cube, self.index)
        self.assertEqual(restored.model_id, 'test-model')
        self.assertEqual(restored.architecture_id, 'arch-v1')
        self.assertEqual(restored.tokenizer_id, 'tok-v1')
        self.assertEqual(restored.quantization_scheme, 'q4')
        self.assertEqual(restored.reconstruction_order, ('embed', 'output', 'layer0'))
        self.assertTrue(manifest.verify({'embed': self.payload, 'output': b'chunk-b', 'layer0': b'chunk-c'}))
        self.assertFalse(manifest.verify({'embed': b'bad'}))
        layout = manifest.reconstruction_layout()
        self.assertEqual(layout['tensor_order'], ['attention'])
        self.assertEqual(layout['shard_groups']['embedding'], ['embed'])
        self.assertEqual(layout['shard_groups']['output'], ['output'])
        restored_layout = RubiksCheckpointManifest.from_layout(
            {
                'model_id': 'test-model',
                'architecture_id': 'arch-v1',
                'tokenizer_id': 'tok-v1',
                'quantization_scheme': 'q4',
                'chunk_order': ['embed', 'output', 'layer0'],
            },
            self.index,
        )
        self.assertEqual(restored_layout.reconstruction_order, ('embed', 'output', 'layer0'))
        payload = manifest.to_bytes()
        restored_bytes = RubiksCheckpointManifest.from_bytes(payload)
        self.assertEqual(restored_bytes.model_id, 'test-model')
        self.assertEqual(restored_bytes.chunk_index.model_id, 'test-model')


if __name__ == '__main__':
    unittest.main()
