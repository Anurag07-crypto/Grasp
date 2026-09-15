from huggingface_hub import InferenceClient
from typing import List
import os
from dotenv import load_dotenv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__name__).parent.parent))
from backend.utils.logger import get_logger
load_dotenv()

logger = get_logger(__name__)
class HFEMBEDDER:
    def __init__(
        self,
        model:str="sentence-transformers/all-MiniLM-L6-v2"):
        
        API_KEY = os.getenv("HF_API_KEY")
        if API_KEY:
            logger.info("API Key Loaded")
        else:
            logger.error("Unable to load HuggingFace API key")
        
        self.model = model
        self.client = InferenceClient(
            model=self.model,
            provider="hf-inference",
            api_key=API_KEY
        )
        
    def embed(self, text:str):
        """
        Loading HuggingFace model and embed

        Args:
            text (str): text to get embed
        """
        try:
            result = self.client.feature_extraction(
                text
            )
            logger.info("Results created successfully")
            return result.tolist()
        except RuntimeError as e:
            logger.error(f"Unfortunatly results are not created: {e}")
            raise RuntimeError("Unfortunatly results are not created") from e
        
    def generate_embeddings(self, texts:List[str]):
        """
        Generating Embeddings

        Args:
            texts (List[str]): List of Texts
        """
        try:
            embeddings = [self.embed(text) for text in texts]
            logger.info("Embeddings Generated")
            return embeddings
        except Exception as e:
            logger.error(f"Got some unexpected error: {e}")
            raise Exception("Got some unexpected error") from e

embedder = HFEMBEDDER()

text = "Kubernetes Pod"

vector = embedder.embed(text)

print("Embedding generated!")
print("Dimensions:", len(vector))
print("First 5 values:", vector[:5])
        