from pathlib import Path
import re
import json
import sys
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


def main():
    print(f"Loading extracted chunks from: {INPUT_PATH}")

    chunks = load_json()
    print(f"Loaded {len(chunks)} chunks")

    normalized_chunks = normalize_chunks(chunks)

    save_normalized_chunks(normalized_chunks)

    print(f"Saved normalized chunks to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

