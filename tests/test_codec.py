import unittest
from braillebyte import BrailleByteCodec


class BrailleByteCodecTests(unittest.TestCase):
    def setUp(self):
        self.codec = BrailleByteCodec()

    def test_byte_braille_round_trip(self):
        for value in range(256):
            cell = self.codec.byte_to_braille(value)
            self.assertEqual(self.codec.braille_to_byte(cell), value)

    def test_known_sentence(self):
        result = self.codec.encode("The cow goes moo.")
        self.assertEqual(result.bytes_, (1, 32, 41, 65, 128, 33, 38, 146, 35, 192, 2))
        self.assertEqual(len(result.braille), len(result.bytes_))

    def test_unknown_word(self):
        result = self.codec.encode("quasar")
        self.assertEqual(result.bytes_, (1, 4, 240, 6, 113, 117, 97, 115, 97, 114, 2))
        self.assertEqual(result.interpretations[0].status, 'unknown')

    def test_multilingual_forms_resolve_to_the_same_concept(self):
        expected = 'SEM:ANIMAL:COW'
        for form in ('cow', 'vaca', 'بقرة', '牛'):
            result = self.codec.encode(form)
            self.assertEqual(result.interpretations[0].resolved.identity, expected)
            self.assertEqual(result.bytes_, (1, 65, 128, 2))

    def test_expanded_registry_resolves_more_languages(self):
        cases = {
            'vache': 'SEM:ANIMAL:COW',
            'корова': 'SEM:ANIMAL:COW',
            'गाय': 'SEM:ANIMAL:COW',
            'wasser': 'SEM:SUBSTANCE:WATER',
            'вода': 'SEM:SUBSTANCE:WATER',
            'ماء': 'SEM:SUBSTANCE:WATER',
            '食べる': 'SEM:ACTION:EAT',
            '人': 'SEM:ENTITY:PERSON',
        }
        for form, expected in cases.items():
            self.assertEqual(self.codec.interpret(form)[0].resolved.identity, expected)

    def test_unspaced_cjk_concepts_are_segmented(self):
        result = self.codec.encode('牛吃水')
        self.assertEqual([item.resolved.identity for item in result.interpretations], [
            'SEM:ANIMAL:COW', 'SEM:ACTION:EAT', 'SEM:SUBSTANCE:WATER',
        ])
        self.assertEqual(result.bytes_, (1, 65, 128, 147, 67, 2))

    def test_ambiguous_forms_are_not_silently_resolved(self):
        result = self.codec.encode('bank')
        self.assertEqual(result.interpretations[0].status, 'ambiguous')
        self.assertEqual(result.bytes_[1:3], (4, 240))
        self.assertEqual(len(result.interpretations[0].candidates), 2)

    def test_explain_handles_expanded_literal_forms(self):
        trace = self.codec.explain('quasar')
        self.assertEqual(trace[2]['name'], 'LITERAL')

    def test_spoken_byte_is_deterministic(self):
        self.assertEqual(self.codec.speak_byte(0), "blank")
        self.assertEqual(self.codec.speak_byte(1), "ka")
        self.assertEqual(self.codec.speak_byte(255), "ka-ta-mi-no-se-lu-ri-va")


if __name__ == "__main__":
    unittest.main()
