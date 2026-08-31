from pydantic import BaseModel, Field
from langchain_openrouter import ChatOpenRouter
import sys
from pathlib import Path
from dotenv import load_dotenv
import os
sys.path.insert(0, str(Path(__name__).parent.parent))
from backend.utils.logger import get_logger
import json

logger = get_logger(__name__)
load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[2]

json_path = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
class Concept(BaseModel):
    name: str = Field(
        description="Name of the educational concept"
    )
    
    prerequisites: list[str] = Field(
        default_factory=list,
        description="Concepts that should be understood before learning this concept"
    )
    
class LLM_extraction(BaseModel):
    concepts: list[Concept] = Field(
        default_factory=list,
        description="Concept extracted from that chunk"
    )

class Extracted_results(BaseModel):
    chunk_id : str
    concepts : list[Concept]

OPENROUTER_API = os.getenv("OPENROUTER_API_KEY")

if OPENROUTER_API:
    logger.info("API_KEY loaded successfully")
else:
    logger.error("API_KEY not found")

llm = ChatOpenRouter(
    model="openrouter/free",
    temperature=0
)

structured_llm = llm.with_structured_output(
    LLM_extraction,
    method="json_schema"
)

def load_test_chunks(path: str | Path, limit: int = 5) -> list[dict]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks[:limit]

def extract_chunks(chunks: dict) ->LLM_extraction:
    prompt = (f"""
    You are an educational knowledge extraction system.
    
    Your task is to extract meaningful educational concepts
    from the provided text chunk.
    
    Chunk ID:
    {chunks["id"]}
    
    Chunk text:
    {chunks["text"]}
    
    Extraction rules:
    
    1. Extract only concepts that are explicitly supported by
       the provided text.
    
    2. Prefer meaningful educational concepts such as:
       - algorithms
       - methods
       - theories
       - mathematical concepts
       - models
       - architectures
       - techniques
       - important domain-specific concepts
    
    3. Do NOT extract ordinary words, individual variables,
       descriptive words, or trivial phrases as separate concepts.
    
    4. Do not create concepts using information that is not
       present in the chunk.
    
    5. Identify prerequisites only when the chunk provides
       evidence that one concept should be understood before
       another concept.
    
    6. Do not assume prerequisites from general knowledge.
    
    7. A concept may have an empty prerequisite list.
    
    8. If the chunk does not contain meaningful educational
       concepts, return an empty concepts list.
    
    9. Keep concept names concise and canonical.
    
    10. Do not duplicate concepts.
    """
    )
    extracted = structured_llm.invoke(prompt)
    return Extracted_results(
        chunk_id=chunks["id"],
        concepts=extracted.concepts
    )

if __name__ == "__main__":

    chunks = load_test_chunks(
        json_path,
        limit=5
    )

    logger.info(f"Loaded {len(chunks)} chunks for testing")

    for index, chunk in enumerate(chunks, start=1):

        logger.info(
            f"Processing test chunk {index}/5: {chunk['id']}"
        )

        try:
            result = extract_chunks(chunk)

            print("\n" + "=" * 80)
            print(f"CHUNK {index}")
            print(f"ID: {chunk['id']}")
            print("-" * 80)
            print(result)

        except Exception as e:
            logger.error(
                f"Failed to extract chunk {chunk['id']}: {e}"
            )