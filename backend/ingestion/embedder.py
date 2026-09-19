import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__name__).parent.parent))
from backend.utils.logger import get_logger
from backend.vector.embedder import HFEMBEDDER

logger = get_logger(__name__)
embedder = HFEMBEDDER()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "extracted" / "extracted_chunks.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "extracted" / "concept_embeddings.json"

class CHUNK_EMBEDDER:
    def __init__(
        self,
        input_path:str,
        output_path:str):
        
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
    
    def extract_chunks(self):
        """
        Extracting Chunks

        Returns:
            _type_: json file
        """
        
        with open(self.input_path, "r", encoding="utf-8") as f:
            logger.info("Json file Loaded")
            return json.load(f)
    
    def extract_concepts(self):
        """
        Extracting_concepts from json file
        """
        
        concepts = set()
        try:
            chunks = self.extract_chunks()
            for chunk in chunks:
                for concept in chunk.get("concepts", []):
                    name = concept.get("name")
                    if name:
                        concepts.add(name)
                    
                    for prerequisite in concept.get("prerequisites", []):
                        if isinstance(prerequisite, str):
                            if prerequisite:
                                concepts.add(prerequisite)
                        elif isinstance(prerequisite, dict):
                            name = prerequisite.get("name")
                            if name:
                                concepts.add(name)
                                
            logger.info("concepts extracted from chunks")
            return sorted(concepts)
        except Exception as e:
            logger.debug("chunks not exracted successfully",e)
            raise Exception("chunks not extracted successfully") from e
        
    def generate_embeddings(self):
        embedder = HFEMBEDDER()
        embedder.load_model()

        concepts = self.extract_concepts()
        vectors = embedder.generate_embeddings(concepts)
        embeddings = {
            name: vector.tolist()
            for name, vector in zip(concepts, vectors)
        }

        return embeddings


    def save_embeddings(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.generate_embeddings(), f, ensure_ascii=False, indent=2)
        logger.info(
            "Embeddings saved to %s",
            self.output_path
        )
        
def main():
    embedder = CHUNK_EMBEDDER(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH
    )

    embedder.save_embeddings()
if __name__ == "__main__":
    main()