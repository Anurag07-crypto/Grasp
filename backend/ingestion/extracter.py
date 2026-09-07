from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
import os  
from pathlib import Path
from dotenv import load_dotenv
import sys
import json
sys.path.insert(0, (str(Path(__name__).parent.parent)))
from backend.utils.logger import get_logger
from prompt import PROMPT
logger = get_logger(__name__)
load_dotenv()

class Concept(BaseModel):
    name: str = Field(
        description="Name of a Kubernetes concept explicitly discussed in the text."
    )

    prerequisites: list[str] = Field(
        description="Kubernetes concepts that are prerequisites for this concept, based only on relationships supported by the text."
    )


class ExtractInformation(BaseModel):
    concepts: list[Concept] = Field(
        description="All meaningful Kubernetes concepts explicitly discussed in the chunk."
    )
    
class Extracter:
    """
    Extracting concepts and important informations for the graph
    """
    def __init__(self,
                 output_path:str,
                 input_path:str,
                 ):
        self.output_path = output_path
        self.input_path = input_path
    
    def load_json(self):
        with open(self.input_path, "r",encoding="utf-8") as f:
            return json.load(f)
    
    def llm(self, model:str="openai/gpt-oss-20b"):
        API_KEY = os.getenv("GROQ_API_KEY")
        if API_KEY:
            logger.info("API KEY loaded successfully")
        else:
            logger.error("Issue with the API KEY")
            raise RuntimeError("Issue with the API KEY")
        
        try:
            llm = ChatGroq(api_key=API_KEY,
                            model=model)
            structured_llm = llm.with_structured_output(schema=ExtractInformation,
                                                        method="json_schema")
            chunks = self.load_json()
            extracted = []
            for chunk in chunks:

                chunk_id = chunk.get("id")
                text = chunk.get("text", "")

                if not text.strip():
                    continue

                llm_response = structured_llm.invoke(
                    PROMPT(text)
                )

                extracted.append(
                    {
                        "chunk_id": chunk_id,
                        "concepts": [
                            {
                                "name": concept.name,
                                "prerequisites": concept.prerequisites
                            }
                            for concept in llm_response.concepts
                        ]
                    }
                )

                logger.info(
                    f"Extracted concepts and prerequisites from chunk: {chunk_id}"
                )
        
        except RuntimeError as e:
            logger.error(f"Chunks not loaded successfully:{e}")
            raise RuntimeError("Chunks not loaded successfully") from e
                
                    