import json
import re
from pathlib import Path
from collections import Counter
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__name__).parent.parent))
from backend.utils.logger import get_logger

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

chunks_path = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
output_path = PROJECT_ROOT / "backend" / "data" / "extracted" / "extracted_chunks.json"

# Technical terms and concepts that should be extracted
TECHNICAL_KEYWORDS = {
    # Machine Learning concepts
    "machine learning", "deep learning", "neural network", "algorithm", "model",
    "supervised learning", "unsupervised learning", "reinforcement learning",
    "classification", "regression", "clustering", "decision tree", "random forest",
    "support vector machine", "svm", "gradient descent", "backpropagation",
    "convolutional neural network", "cnn", "recurrent neural network", "rnn",
    "lstm", "transformer", "attention mechanism", "embedding",
    "feature engineering", "dimensionality reduction", "pca", "t-sne",
    
    # Data concepts
    "big data", "data analytics", "data science", "dataset", "training data",
    "test data", "validation", "cross-validation", "overfitting", "underfitting",
    "bias", "variance", "regularization", "normalization", "standardization",
    "feature extraction", "feature selection", "preprocessing",
    
    # AI/ML techniques
    "artificial intelligence", "ai", "computer vision", "natural language processing",
    "nlp", "text mining", "sentiment analysis", "image recognition", "voice recognition",
    "recommendation system", "recommender system", "anomaly detection",
    
    # Evaluation metrics
    "accuracy", "precision", "recall", "f1 score", "auc", "roc curve",
    "confusion matrix", "loss function", "optimization", "convergence",
    
    # Architecture/Implementation
    "training", "inference", "prediction", "classification model", "regression model",
    "ensemble method", "boosting", "bagging", "hyperparameter", "hyperparameter tuning",
    "cross validation", "grid search", "random search",
    
    # Domain applications
    "healthcare", "medical diagnosis", "autonomous vehicle", "robotics",
    "recommendation engine", "fraud detection", "predictive analysis",
    "time series", "sequence modeling", "natural language understanding",
    
    # Advanced topics
    "transfer learning", "fine-tuning", "few-shot learning", "zero-shot learning",
    "meta-learning", "curriculum learning", "active learning", "semi-supervised",
    "self-supervised", "unsupervised learning", "generative model", "gan",
    "adversarial", "explainability", "interpretability", "fairness",
    "privacy", "differential privacy", "federated learning",
    
    # Statistics/Math
    "probability", "bayesian", "inference", "hypothesis testing", "confidence interval",
    "distribution", "gaussian", "normal distribution", "correlation", "covariance",
    "eigenvector", "eigenvalue", "matrix factorization", "decomposition",
    
    # Others
    "data mining", "knowledge extraction", "pattern recognition",
    "system learning", "adaptive system", "intelligent system",
}

# Patterns to identify technical concepts (regex patterns)
CONCEPT_PATTERNS = [
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:algorithm|model|method|technique|network|system)\b',
    r'\b(?:algorithm|framework|library|method|technique|approach|model)\s+(?:for\s+)?[A-Za-z\s]+',
]

def extract_concepts_nlp(chunk_text: str) -> list[dict]:
    """
    Extract concepts from chunk using NLP-based approach
    without requiring LLM calls.
    """
    concepts = []
    concept_set = set()
    
    # Normalize text
    text_lower = chunk_text.lower()
    
    # Extract technical keywords
    for keyword in TECHNICAL_KEYWORDS:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text_lower):
            # Capitalize properly for display
            formatted_concept = ' '.join(word.capitalize() for word in keyword.split())
            if formatted_concept not in concept_set:
                concept_set.add(formatted_concept)
                concepts.append({
                    "name": formatted_concept,
                    "prerequisites": []
                })
    
    # Extract capitalized phrases (likely important concepts)
    capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', chunk_text)
    for phrase in capitalized_phrases:
        # Filter out common non-technical phrases
        if len(phrase) > 3 and phrase not in concept_set and not is_common_phrase(phrase):
            concept_set.add(phrase)
            concepts.append({
                "name": phrase,
                "prerequisites": []
            })
    
    # Remove non-technical noise and filter
    concepts = filter_concepts(concepts, chunk_text)
    
    return concepts

def is_common_phrase(phrase: str) -> bool:
    """Check if phrase is a common non-technical term"""
    common_phrases = {
        'The', 'This', 'That', 'These', 'Those', 'Chapter', 'Section',
        'Page', 'Figure', 'Table', 'Appendix', 'Introduction', 'Conclusion',
        'Summary', 'Reference', 'Abstract', 'Author', 'Publisher', 'Edition',
        'Year', 'Date', 'Version', 'Contents', 'Index', 'Glossary',
    }
    return phrase in common_phrases

def filter_concepts(concepts: list[dict], chunk_text: str) -> list[dict]:
    """
    Filter concepts to keep only meaningful technical/educational ones.
    Remove generic terms, duplicates, and non-educational content.
    """
    filtered = []
    seen = set()
    
    # Terms to exclude
    exclude_terms = {
        'page', 'number', 'document', 'text', 'content', 'information',
        'example', 'figure', 'table', 'chapter', 'section', 'paragraph',
        'reference', 'citation', 'footnote', 'index', 'glossary',
        'introduction', 'conclusion', 'summary', 'abstract', 'background',
        'overview', 'definition', 'explanation', 'description', 'discussion',
        'acknowledgment', 'appendix', 'bibliography', 'author', 'publisher',
        'year', 'date', 'edition', 'version', 'isbn', 'issn', 'url', 'doi',
    }
    
    for concept in concepts:
        name_lower = concept["name"].lower()
        
        # Skip if already seen (duplicate)
        if name_lower in seen:
            continue
            
        # Skip common non-technical terms
        if name_lower in exclude_terms:
            continue
        
        # Skip generic single words (unless technical)
        if len(name_lower.split()) == 1 and name_lower not in TECHNICAL_KEYWORDS:
            continue
        
        # Skip very vague terms
        if is_too_vague(concept["name"]):
            continue
        
        # Keep if it's meaningful
        seen.add(name_lower)
        filtered.append(concept)
    
    # Remove duplicates with different cases
    final = []
    seen_normalized = set()
    for concept in filtered:
        normalized = concept["name"].lower()
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            final.append(concept)
    
    return final

def is_too_vague(concept_name: str) -> bool:
    """Check if concept is too broad or vague"""
    vague_terms = {
        'thing', 'stuff', 'concept', 'idea', 'topic', 'subject', 'area',
        'type', 'kind', 'form', 'way', 'method', 'process', 'system',
        'model', 'framework', 'approach', 'technique', 'tool',
    }
    return concept_name.lower() in vague_terms

def load_chunks(path: str | Path) -> list[dict]:
    """Load chunks from JSON file"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def extract_all_chunks(chunks: list[dict]) -> list[dict]:
    """Extract concepts from all chunks"""
    results = []
    
    for index, chunk in enumerate(chunks):
        if (index + 1) % 100 == 0:
            logger.info(f"Processing chunk {index + 1}/{len(chunks)}: {chunk['id']}")
        
        try:
            concepts = extract_concepts_nlp(chunk["text"])
            results.append({
                "chunk_id": chunk["id"],
                "concepts": concepts
            })
        except Exception as e:
            logger.error(f"Failed to extract chunk {chunk['id']}: {e}")
            results.append({
                "chunk_id": chunk["id"],
                "concepts": []
            })
    
    return results

def save_results(results: list[dict], output_path: str | Path):
    """Save extraction results to JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info("Starting NLP-based concept extraction...")
    
    # Load chunks
    chunks = load_chunks(chunks_path)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Extract concepts
    logger.info("Extracting concepts from all chunks...")
    results = extract_all_chunks(chunks)
    
    # Save results
    save_results(results, output_path)
    
    # Summary
    elapsed = datetime.now() - start_time
    total_concepts = sum(len(r["concepts"]) for r in results)
    
    print(f"\n{'='*80}")
    print(f"Extraction Complete (NLP-based)")
    print(f"{'='*80}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total concepts extracted: {total_concepts}")
    print(f"Average concepts per chunk: {total_concepts/len(chunks):.2f}")
    print(f"Time elapsed: {elapsed}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")
