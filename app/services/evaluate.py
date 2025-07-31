from typing import List, Tuple, Callable, Any
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

def evaluate_model(
    predict_func: Callable[[List[str], Any, Any], List[str]],
    test_sentences: List[List[Tuple[str, str]]],
    stats: Any,
    tag_set: Any,
    workers: int = 4
) -> float:
    """Enhanced evaluation with parallel processing"""
    total = 0
    correct = 0
    
    def process_sentence(sentence):
        nonlocal total, correct
        try:
            words = [word for word, tag in sentence]
            true_tags = [tag for word, tag in sentence]
            predicted_tags = predict_func(words, tag_set, stats)
            
            sentence_correct = sum(1 for true, pred in zip(true_tags, predicted_tags) if true == pred )
            sentence_total = len(true_tags)
            
            return sentence_correct, sentence_total
        except Exception as e:
            logger.error(f"Error processing sentence: {e}")
            return 0, 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(process_sentence, test_sentences)
        for res_correct, res_total in results:
            correct += res_correct
            total += res_total
    
    return correct / total if total else 0