import math
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ViterbiTagger:
    def __init__(self):
        self._transition_cache: Dict[Tuple[str, str], float] = {}
        self._emission_cache: Dict[Tuple[str, str], float] = {}

    def get_transition_prob(self, prev_tag: str, curr_tag: str, stats: Any) -> float:
        """Manual caching for transition probabilities"""
        cache_key = (prev_tag, curr_tag)
        if cache_key not in self._transition_cache:
            count = stats.tag_transition_count.get(cache_key, 1)
            total = stats.tag_count.get(prev_tag, 1)
            self._transition_cache[cache_key] = math.log(count / total)
        return self._transition_cache[cache_key]

    def get_emission_prob(self,word: str, tag: str, stats: Any) -> float:
    # Beri probabilitas kecil untuk kata yang tidak dikenal
        if (word.lower(), tag) not in stats.word_tag_count:
            return math.log(1e-6)  # Nilai sangat kecil
        return math.log(stats.word_tag_count.get((word.lower(), tag), 1) / stats.tag_count.get(tag, 1))

    def viterbi(self, words: List[str], tag_set: Any, stats: Any) -> List[str]:
        """Optimized Viterbi algorithm with manual caching"""
        if not words:
            return []
        
        try:
            # Clear caches for new sentence
            self._transition_cache.clear()
            self._emission_cache.clear()
            
            # Initialization
            V = [{}]
            backpointer = [{}]
            
            # Precompute initial probabilities
            for tag in tag_set:
                P_tag = stats.tag_count.get(tag, 1) / stats.total_words
                P_emission = self.get_emission_prob(words[0], tag, stats)
                V[0][tag] = math.log(P_tag) + P_emission
                backpointer[0][tag] = None
            
            # Iteration
            for i in range(1, len(words)):
                V.append({})
                backpointer.append({})
                for curr_tag in tag_set:
                    max_prob = float('-inf')
                    best_prev = None
                    
                    for prev_tag in tag_set:
                        trans_prob = self.get_transition_prob(prev_tag, curr_tag, stats)
                        emit_prob = self.get_emission_prob(words[i], curr_tag, stats)
                        prob = V[i-1][prev_tag] + trans_prob + emit_prob
                        
                        if prob > max_prob:
                            max_prob = prob
                            best_prev = prev_tag
                    
                    V[i][curr_tag] = max_prob
                    backpointer[i][curr_tag] = best_prev
            
            # Termination
            last_tag = max(V[-1], key=lambda x: V[-1][x])
            best_path = [last_tag]
            
            # Backtracking
            for i in range(len(words)-1, 0, -1):
                best_path.insert(0, backpointer[i][best_path[0]])
            
            return best_path
        
        except Exception as e:
            logger.error(f"Viterbi algorithm failed: {e}", exc_info=True)
            # Fallback: return most common tag for each word
            return [max(tag_set, key=lambda t: stats.tag_count.get(t, 0)) for _ in words]