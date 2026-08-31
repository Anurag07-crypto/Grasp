from pathlib import Path
import json
import sys
sys.path.insert(0 , str(Path(__name__).parent.parent))
from backend.utils.logger import get_logger

logger = get_logger(__name__)
def load_extractions(path:str):
    json_path = Path(path)
    
    with open(json_path,"r",encoding="utf-8") as file:
        return json.load(file)

def validate_extractions(results:list[dict]):
    
    total_chunks = len(results)
    successful_chunks = 0
    empty_chunks = 0
    total_concepts = 0
    
    for result in results:
        concepts = result.get("concepts",[])
        if concepts:
            successful_chunks += 1
            total_concepts += len(concepts)
        else:
            empty_chunks += 1

    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Chunks with concepts: {successful_chunks}")
    logger.info(f"Empty chunks: {empty_chunks}")
    logger.info(f"Total concepts: {total_concepts}")

    if total_chunks > 0:
        average = total_concepts / total_chunks
        logger.info(
            f"Average concepts per chunk: {average:.2f}"
        )
if __name__ == "__main__":

    extraction_path = Path(
        "data/extracted/extracted_test.json"
    )

    results = load_extractions(extraction_path)

    validate_extractions(results)