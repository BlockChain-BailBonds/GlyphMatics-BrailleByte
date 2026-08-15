import unittest
from braillebyte import BrailleByteCodec, SemanticGraph, SemanticGraphCodec, realize

class GraphTest(unittest.TestCase):
 def test_roundtrip_and_controlled_realization(self):
  graph = SemanticGraph({'AGENT':'SEM:ANIMAL:COW','ACTION':'SEM:ACTION:EAT','PATIENT':'SEM:SUBSTANCE:FOOD'})
  codec = SemanticGraphCodec(BrailleByteCodec().concepts)
  self.assertEqual(codec.decode(codec.encode(graph)).roles, graph.roles)
  self.assertEqual(realize(graph, 'zh-Hans'), '牛吃食物。')
