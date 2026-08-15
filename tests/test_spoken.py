import unittest
from braillebyte import BrailleByteCodec, SpokenBrailleByte

class SpokenTests(unittest.TestCase):
 def test_framed_speech_round_trip(self):
  codec = BrailleByteCodec(); spoken = SpokenBrailleByte(codec.dot_syllables)
  values = (1, 65, 128, 2)
  self.assertEqual(spoken.hear(spoken.speak(values)), values)
 def test_invalid_pronunciation_rejected(self):
  with self.assertRaises(ValueError): SpokenBrailleByte(BrailleByteCodec().dot_syllables).hear('braillebyte ka-ka end')
