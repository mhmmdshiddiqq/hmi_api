import language_tool_python
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
import logging
from functools import lru_cache

from scipy import stats

from models.models import CorpusStats
from services.viterby_tagger import ViterbiTagger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LanguageTool with pooling for better performance
tool = language_tool_python.LanguageTool('en-US', config={'maxCheckThreads': 4})

QUESTION_KEYWORDS = {
    "name": "My name is John",
    "from": "I am from Indonesia",
    "old": "I am 20 years old",
    "student": "Yes, I am a student",
    "like": "I like playing guitar and watching movie",
    "goodbye": "Thank you and goodbye",
    # Add other keywords as needed
}

@lru_cache(maxsize=1024)
def correct_grammar(text: str) -> str:
    """Cache grammar corrections to improve performance for repeated inputs"""
    try:
        matches = tool.check(text)
        return language_tool_python.utils.correct(text, matches)
    except Exception as e:
        logger.error(f"Grammar correction failed: {e}")
        return text  # Fallback to original text on error

def generate_suggestion(keyword: str, similarity: float) -> str:
    """Generate suggestion message based on keyword and similarity score"""
    
    # Mapping keyword ke pesan yang sesuai
    suggestion_messages = {
        "name": "Your name is different, but that's perfectly normal! Everyone has their own unique name.",
        "from": "Your city or country might be different , but that's okay! Everyone comes from different places.",
        "old": "Your age is different , but that's fine! Everyone has their own age.",
        "student": "Your student status might be different , but that's normal! People have different educational situations.",
        "like": "Your hobbies might be different , but that's fine! Everyone has their own interests.",
        "goodbye": "Your goodbye message might be different , but that's fine! Everyone has different goodbye messages."
    }
    
    # Dapatkan pesan dasar berdasarkan keyword
    base_message = suggestion_messages.get(keyword, "Your answer is different from the example, but personal information varies for everyone!")
    
    # Tambahkan pesan berdasarkan similarity score
    if similarity >= 80:
        return f"{base_message} Your answer structure is very good!"
    elif similarity >= 60:
        return f"{base_message} Try to use similar sentence structure as the example."
    elif similarity >= 40:
        return f"{base_message} Consider using a sentence structure closer to the example format."
    else:
        return f"{base_message} Try to follow the example sentence pattern more closely."

def similarity_score(user_answer: str, expected_answer: str) -> float:
    """Calculate similarity score with case insensitivity and caching"""
    user_answer = user_answer.lower().strip()
    expected_answer = expected_answer.lower().strip()
    return round(SequenceMatcher(None, user_answer, expected_answer).ratio() * 100, 2)

def extract_keyword(question: str) -> Optional[str]:
    """Cari keyword dalam pertanyaan (case-insensitive)"""
    question_lower = question.lower()
    print(f"DEBUG: Searching keyword in question: '{question_lower}'")  # Debug line
    
    for keyword in QUESTION_KEYWORDS.keys():
        if keyword in question_lower:
            print(f"DEBUG: Found keyword: '{keyword}'")  # Debug line
            return keyword
    
    print(f"DEBUG: No keyword found in available keywords: {list(QUESTION_KEYWORDS.keys())}")  # Debug line
    return None

def is_question(text: str) -> bool:
    """Validasi apakah text berupa pertanyaan"""
    question_words = ["what", "where", "when", "who", "why", "how", "which", "whose", "whom"]
    question_patterns = ["?", "are you", "do you", "can you", "will you", "have you"]
    
    text_lower = text.lower().strip()
    
    # Cek apakah dimulai dengan question words
    for word in question_words:
        if text_lower.startswith(word):
            return True
    
    # Cek apakah mengandung pattern pertanyaan
    for pattern in question_patterns:
        if pattern in text_lower:
            return True
            
    return False

def speaking_ability_score(question: str, user_answer: str, stats: CorpusStats, tag_set: set) -> Dict[str, Any]:
    try:
        print(f"DEBUG: Function called with:")  # Debug line
        print(f"  question: '{question}'")  # Debug line
        print(f"  user_answer: '{user_answer}'")  # Debug line
        
        # Validasi input - question dan user_answer wajib
        if not question or not user_answer:
            return {"error": "Question and user_answer are required", "final_score": 0.0}
        
        # Validasi apakah question benar-benar pertanyaan
        if not is_question(question):
            return {
                "error": f"The 'question' parameter should be a question, but got: '{question}'",
                "suggestion": "Make sure the first parameter is a question (e.g., 'Where are you from?')",
                "final_score": 0.0
            }
        
        # Ekstrak keyword
        keyword = extract_keyword(question)
        if not keyword:
            return {
                "error": f"Keyword not found in question: '{question}'", 
                "available_keywords": list(QUESTION_KEYWORDS.keys()),
                "final_score": 0.0
            }
        
        # Tentukan expected_answer: selalu ambil dari QUESTION_KEYWORDS berdasarkan keyword
        expected_answer = QUESTION_KEYWORDS[keyword]
        print(f"DEBUG: Using expected_answer from QUESTION_KEYWORDS: '{expected_answer}'")
        
        # Hitung similarity
        similarity = similarity_score(user_answer, expected_answer)
        user_words = user_answer.split()
        predicted_tags = ViterbiTagger().viterbi(user_words, tag_set, stats)  # Gunakan Viterbi
        print(f"DEBUG: Predicted tags: {predicted_tags}")
        
        grammar_rules = {
            "PRP_VBP": ("Personal pronoun should be followed by verb", ["PRP", "VBP"]),
            "DT_NN": ("Determiner should be followed by noun", ["DT", "NN"])
        }
        
        grammar_errors = []
        for i in range(len(predicted_tags)-1):
            for rule_name, (desc, pattern) in grammar_rules.items():
                if predicted_tags[i:i+2] == pattern:
                    grammar_errors.append(f"{rule_name}: {desc}")
        
        # Generate suggestion
        suggestion = generate_suggestion(keyword, similarity)
        
        # Return response yang benar
        return {
            "question": question,
            "user_answer": user_answer,
            "similarity": similarity,
            "grammar_score": 100 - (len(grammar_errors) * 10),
            "grammar_errors": grammar_errors,
            "suggestion": suggestion,
            "pos_tags": list(zip(user_words, predicted_tags)),  # Kata + tagnya
        }
        
        # print(f"DEBUG: Result: {result}")  # Debug line
        # return result
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return {"error": str(e)}