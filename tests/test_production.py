import unittest
from braillebyte.production import identify_language, resolve_bank, context_envelope
class ProductionTests(unittest.TestCase):
 def test_services_preserve_uncertainty(self):
  self.assertEqual(identify_language('牛').language,'zh-Hans')
  self.assertEqual(resolve_bank('river bank')[0][0],'SEM:GEOGRAPHY:RIVER_BANK')
  self.assertFalse(context_envelope('en-US','formal',True)['literalize'])
