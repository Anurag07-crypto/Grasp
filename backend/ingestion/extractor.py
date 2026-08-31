from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
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

GROQ_API = os.getenv("GROQ_API_KEY")

if GROQ_API:
    logger.info("API_KEY loaded successfully")
else:
    logger.error("API_KEY not found")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
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
    prompt = f"""
You are extracting educational concepts from a single document chunk.

Chunk ID:
{chunk["id"]}

Chunk text:
{chunk["text"]}

Your task:
Extract meaningful technical or educational concepts that are
explicitly supported by the provided chunk.

Rules:

1. Extract only concepts supported by the chunk.
2. Do not use outside knowledge.
3. Prefer concepts that a learner could reasonably study or understand.
4. Prefer technical concepts, methods, algorithms, models, theories,
   principles, techniques, or important domain concepts.
5. Do not extract:
   - page numbers
   - document metadata
   - copyright information
   - dates
   - section headings
   - table-of-contents entries
   - organizations
   - locations
   - generic words or phrases
6. Do not extract a concept merely because it is related to another
   concept mentioned in the text.
7. Application examples should only be extracted when they represent
   a meaningful concept in the context of the text.
8. Do not invent prerequisites.
9. Add a prerequisite only when the chunk provides enough evidence
   that the prerequisite should be understood before the concept.
10. If the chunk contains no meaningful educational concepts,
    return an empty concepts list.
11. Avoid overly broad or vague concepts.
12. Avoid duplicate concepts within the same chunk.
13. Treat table-of-contents pages and navigation lists as non-content.
    Do not extract concepts from section titles or chapter listings
    when they appear as a table of contents.
14. A concept should normally be supported by explanatory prose,
    definitions, descriptions, examples, relationships, or claims
    in the chunk—not merely by appearing as a heading or title.
15. If the chunk consists primarily of headings, titles, metadata,
    or a table of contents, return an empty concepts list.
16. Do not require a concept to have a formal definition. A concept
    may be extracted when the prose meaningfully discusses it.

Return the concepts supported by this chunk.
"""
    extracted = structured_llm.invoke(prompt)
    return Extracted_results(
        chunk_id=chunks["id"],
        concepts=extracted.concepts
    )
    
def save_extraction(
    extracted_results: list[Extracted_results],
    output_path:str
):
    
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    data = [
        extracted_result.model_dump()
        for extracted_result in extracted_results
    ]
    
    with open(output_path, "w") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )
    logger.info(
        f"Extraction results saved successfully: {output_path}"
    )
    
if __name__ == "__main__":

    chunk_path = Path("data/chunks/chunks.json")
    output_path = Path("data/extracted/extracted_test.json")

    chunks = load_test_chunks(
        chunk_path,
        limit=10
    )

    logger.info(
        f"Loaded {len(chunks)} chunks for testing"
    )

    results = []

    for index, chunk in enumerate(chunks, start=1):

        logger.info(
            f"Processing test chunk {index}/5: {chunk['id']}"
        )

        try:
            result = extract_chunks(chunk)

            results.append(result)

            print("\n" + "=" * 80)
            print(f"CHUNK {index}")
            print(f"ID: {chunk['id']}")
            print("-" * 80)
            print(result)

        except Exception as e:

            logger.error(
                f"Failed to extract chunk {chunk['id']}: {e}"
            )

    save_extraction(
        results,
        output_path,
    )