from collections import defaultdict
from app.models.models import CorpusStats
import json
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

def load_corpus(file_path: str) -> CorpusStats:
    """Enhanced corpus loading with better error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        
        tag_count = defaultdict(int)
        word_tag_count = defaultdict(int)
        tag_transition_count = defaultdict(int)
        total_words = 0
        
        for sentence in corpus_data:
            prev_tag = "<START>"
            for word, tag in sentence:
                tag_count[tag] += 1
                word_tag_count[(word.lower(), tag)] += 1  # Case insensitive
                tag_transition_count[(prev_tag, tag)] += 1
                total_words += 1
                prev_tag = tag
        
        return CorpusStats(
            tag_count=dict(tag_count),
            word_tag_count=dict(word_tag_count),
            tag_transition_count=dict(tag_transition_count),
            total_words=total_words
        )
    
    except FileNotFoundError:
        logger.error(f"Corpus file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in corpus file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading corpus: {e}")
        raise