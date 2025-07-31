from dataclasses import dataclass
from typing import Dict, Tuple, DefaultDict

# models.py
@dataclass
class CorpusStats:
    tag_count: Dict[str, int]
    word_tag_count: Dict[Tuple[str, str], int]
    tag_transition_count: Dict[Tuple[str, str], int]
    total_words: int

    def __hash__(self):
        return hash((
            frozenset(self.tag_count.items()),
            frozenset(self.word_tag_count.items()),
            frozenset(self.tag_transition_count.items()),
            self.total_words
        ))