from pathlib import Path
import os
import re
import json
import sys
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, str(Path(__name__).parent.parent))
from backend.utils.logger import get_logger

logger = get_logger(__name__)
INPUT_PATH = Path("data/extracted/extracted_chunks.json")
OUTPUT_PATH = Path("data/extracted/normalized_chunks.json")

def normalize_names(name:str):
    """Normalizing the names in chunk.json

    Args:
        name (str): Chunk name
    """
    
    if not name:
        logger.error("Chunk-name not found")
        raise ValueError("Chunk-name not found")
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = name.replace("-", " ")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    
    return name.lower().strip()

def load_json():
    "load extacted_json file"
    with open(INPUT_PATH, "r",encoding="utf-8") as f:
        return json.load(f)

def normalize_chunks(chunks):
    normalized = []

    for chunk in chunks:
        concepts = []

        for concept in chunk.get("concepts", []):
            name = concept.get("name", "")
            prerequisites = concept.get("prerequisites", [])

            normalized_name = normalize_names(name)

            if not normalized_name:
                continue

            normalized_prerequisites = []

            for prerequisite in prerequisites:
                normalized_prerequisite = normalize_names(prerequisite)

                if (
                    normalized_prerequisite
                    and normalized_prerequisite not in normalized_prerequisites
                ):
                    normalized_prerequisites.append(normalized_prerequisite)

            concepts.append(
                {
                    "name": normalized_name,
                    "prerequisites": normalized_prerequisites,
                }
            )

        normalized.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "concepts": concepts,
            }
        )

    return normalized

def save_normalized_chunks(chunks):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

def get_unique_concepts(chunks):
    "Getting unique chunks"
    concepts = set()
    for chunk in chunks:
        for concept in chunk.get("concepts", []):
            name = concept.get("name","")
            if name:
                concepts.add(name)
    return sorted(concepts)

def build_semantic_clusters(concepts,
                            model="sentence-transformers/all-MiniLM-L6-v2",
                            SIMILARITY_THRESHOLD=0.93):
    hf_token = os.getenv("HF_API_KEY")
    logger.info(f'''
                Embedding model using: {model}
                ''')
    if hf_token:
        logger.info("HuggingFace Token has been loaded")
    embedding_model = InferenceClient(
        token=hf_token,
        model=model)
    embeddings = embedding_model.feature_extraction(
        concepts,
        normalize=True
    )
    similarities = cosine_similarity(embeddings)
    
    canonical_map = {}
    
    for i, concept in enumerate(concepts):
        if concept in canonical_map:
            continue
        canonical_map[concept] = concept
        
        for j in range(i+1, len(concepts)):
            
            if concepts[j] in canonical_map:
                continue
            
            similarity = similarities[i][j]
            if similarity >= SIMILARITY_THRESHOLD:
                canonical_map[concepts[j]] = concept

    return canonical_map

def apply_canonical_mapping(chunks, canonical_map):
    normalized = []
    
    for chunk in chunks:
        concepts = []
        for concept in chunk.get("concepts",[]):
            name = concept.get("name"," ")
            prerequisites = concept.get("prerequisites",[])
            canonical_name = canonical_map.get(
                name,
                name
            )
            canonical_prerequisites = []
            
            for prerequisite in prerequisites:
                
                canonical_prerequisite = canonical_map.get(
                    prerequisite,
                    prerequisite
                )
                if (
                    canonical_prerequisite
                    and canonical_prerequisite not in canonical_prerequisites
                ):
                    canonical_prerequisites.append(
                        canonical_prerequisite
                    )

            concepts.append(
                {
                    "name": canonical_name,
                    "prerequisites": canonical_prerequisites,
                }
            )

        normalized.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "concepts": concepts,
            }
        )

    return normalized

if __name__ == "__main__":
    chunks = load_json()
    normalized_chunks = normalize_chunks(chunks)

    unique_concepts = get_unique_concepts(normalized_chunks)

    print(f"Found {len(unique_concepts)} unique concepts")

    canonical_map = build_semantic_clusters(
        unique_concepts
    )

    normalized_chunks = apply_canonical_mapping(
        normalized_chunks,
        canonical_map
    )

    save_normalized_chunks(normalized_chunks)
                

        
