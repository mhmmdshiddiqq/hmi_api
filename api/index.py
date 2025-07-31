from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.corpus_repo import load_corpus
from services.viterby_tagger import ViterbiTagger
from services.evaluate import evaluate_model
from services.language_check import speaking_ability_score
import logging
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    corpus_file = "corpus.json"
    stats = load_corpus(corpus_file)
    tag_set = set(stats.tag_count.keys())
except Exception as e:
    logger.error(f"Failed to initialize corpus: {e}")
    stats = None
    tag_set = set()

app = FastAPI(title="NLP Backend API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentenceInput(BaseModel):
    words: list[str]

class SpeakingInput(BaseModel):
    user_answer: str
    question: str
    
class ConversationInput(BaseModel):
    messages: List[Dict[str, str]]

@app.get("/")
async def home():
    return {"status": "OK"}

@app.post("/tag")
def tag_sentence(input_data: SentenceInput):
    try:
        if not stats or not tag_set:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        tags = ViterbiTagger().viterbi(input_data.words, tag_set, stats)
        return {"words": input_data.words, "tags": tags}
    except Exception as e:
        logger.error(f"Tagging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {
        "status": "ok" if stats and tag_set else "degraded",
        "details": {
            "corpus_loaded": bool(stats),
            "tags_available": len(tag_set) if tag_set else 0
        }
    }

from services.viterby_tagger import ViterbiTagger

@app.get("/accuracy")
def accuracy():
    try:
        import json
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        viterbi_tagger = ViterbiTagger()
        acc = evaluate_model(viterbi_tagger.viterbi, corpus_data, stats, tag_set)
        return {"accuracy": acc}
    except Exception as e:
        logger.error(f"Accuracy calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate-speaking")
def evaluate_speaking(input_data: SpeakingInput):
    try:
        if not stats or not tag_set:
            raise HTTPException(status_code=503, detail="Service not initialized")
            
        return speaking_ability_score(
            question=input_data.question,
            user_answer=input_data.user_answer,
            stats =stats,          # Ditambahkan
            tag_set=tag_set       # Ditambahkan
        )
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/evaluate-conversation")
def evaluate_conversation(input_data: ConversationInput):
    try:
        if not stats or not tag_set:
            raise HTTPException(status_code=503, detail="Service not initialized")
            
        # Pisahkan pesan bot dan user
        bot_messages = [m['message'] for m in input_data.messages if m['role'] == 'bot']
        user_messages = [m['message'] for m in input_data.messages if m['role'] == 'user']
        
        if len(bot_messages) != len(user_messages):
            raise HTTPException(status_code=400, detail="Jumlah pertanyaan dan jawaban tidak sama")
        
        results = []
        total_similarity = 0
        total_grammar = 0
        
        # Evaluasi setiap pasangan pertanyaan-jawaban
        for question, answer in zip(bot_messages, user_messages):
            result = speaking_ability_score(
                question=question,
                user_answer=answer,
                stats=stats,
                tag_set=tag_set
            )
            
            # Pastikan result memiliki nilai default jika None
            similarity = result.get('similarity', 0)
            grammar = result.get('grammar_score', 0)
            
            results.append({
                'question': question,
                'user_answer': answer,
                'similarity': similarity,
                'grammar_score': grammar,
                'grammar_errors': result.get('grammar_errors', []),
                'suggestion': result.get('suggestion', ''),
                'pos_tags': result.get('pos_tags', [])
            })
            
            total_similarity += similarity
            total_grammar += grammar
        
        # Hitung rata-rata
        pair_count = max(1, len(results))  # Hindari division by zero
        avg_similarity = total_similarity / pair_count
        avg_grammar = total_grammar / pair_count
        final_score = (avg_similarity * 0.4 + avg_grammar * 0.6)  # Weighted average
        
        return {
            "success": True,
            "average_similarity": round(avg_similarity, 2),
            "average_grammar": round(avg_grammar, 2),
            "final_score": round(final_score, 2),
            "detailed_results": results,
            "total_pairs": pair_count
        }
        
    except Exception as e:
        logger.error(f"Conversation evaluation error: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "average_similarity": 0,
            "average_grammar": 0,
            "final_score": 0
        }
    
@app.on_event("startup")
async def startup_event():
    """Reload corpus on startup"""
    global stats, tag_set
    try:
        stats = load_corpus(corpus_file)
        tag_set = set(stats.tag_count.keys())
        logger.info("Corpus loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load corpus on startup: {e}")