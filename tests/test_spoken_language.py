import unittest
from braillebyte.spoken_language import parse, speak

class SpokenLanguageTests(unittest.TestCase):
 def test_semantic_round_trip(self):
  self.assertEqual(speak(parse('maku nari savi')), 'maku nari savi')
 def test_unknown_words_are_not_guessed(self):
  with self.assertRaises(ValueError): parse('maku zori savi')
