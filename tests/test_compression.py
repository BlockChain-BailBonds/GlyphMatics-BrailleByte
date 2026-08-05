import unittest

from braillebyte.compression import BrailleByteCompressor, DICT_REF, RAW_ESCAPE


class BrailleByteCompressionTests(unittest.TestCase):
    def test_trained_phrases_round_trip(self):
        streams = [(1, 32, 255, 1, 2, 3, 34, 255, 2, 2)] * 8
        compressor = BrailleByteCompressor.train(streams, min_frequency=2)
        source = (1, 32, 255, 1, 2, 3, 34, 255, 2, 2)
        self.assertLess(len(compressor.compress(source)), len(source))
        self.assertEqual(compressor.decompress(compressor.compress(source)), source)

    def test_reserved_raw_bytes_are_lossless(self):
        compressor = BrailleByteCompressor()
        source = (1, DICT_REF, RAW_ESCAPE, 2)
        self.assertEqual(compressor.decompress(compressor.compress(source)), source)

    def test_compressed_transport_uses_only_braille_cells(self):
        source = (1, 32, 255, 1, 2, 3, 34, 255, 2, 2)
        compressor = BrailleByteCompressor.train([source] * 8, min_frequency=2)
        cells = compressor.compress_to_braille(source)
        self.assertTrue(all(0x2800 <= ord(cell) <= 0x28FF for cell in cells))
        self.assertEqual(compressor.decompress_braille(cells), source)


if __name__ == '__main__':
    unittest.main()
