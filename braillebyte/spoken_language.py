"""BrailleByte Spoken: a compact constructed spoken language over semantic graphs."""
from __future__ import annotations
from .semantic_graph import SemanticGraph

LEXICON = {
 'maku': ('AGENT', 'SEM:ANIMAL:COW'), 'savi': ('PATIENT', 'SEM:SUBSTANCE:FOOD'),
 'nari': ('ACTION', 'SEM:ACTION:EAT'),
}

def parse(sentence: str) -> SemanticGraph:
 """Parse canonical SVO BrailleByte Spoken sentences, e.g. 'maku nari savi'."""
 words = sentence.casefold().strip().split()
 if len(words) != 3 or any(word not in LEXICON for word in words):
  raise ValueError('BrailleByte Spoken requires known canonical SVO words')
 roles = {}
 for word in words:
  role, concept = LEXICON[word]
  if role in roles: raise ValueError('duplicate grammatical role')
  roles[role] = concept
 graph = SemanticGraph(roles)
 graph.validate()
 return graph

def speak(graph: SemanticGraph) -> str:
 """Generate canonical SVO BrailleByte Spoken from a supported graph."""
 inverse = {value: word for word, value in LEXICON.items()}
 try:
  return ' '.join(inverse[(role, graph.roles[role])] for role in ('AGENT', 'ACTION', 'PATIENT'))
 except KeyError as error:
  raise ValueError('graph has no BrailleByte Spoken lexical realization') from error
