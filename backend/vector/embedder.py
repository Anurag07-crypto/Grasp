from huggingface_hub import InferenceClient
from typing import List
import os
from dotenv import load_dotenv
from pathlib import Path
import sys
from sentence_transformers import SentenceTransformer
sys.path.insert(0, str(Path(__name__).parent.parent))
from backend.utils.logger import get_logger
load_dotenv()

logger = get_logger(__name__)
class HFEMBEDDER:
    def __init__(
        self,
        model_name:str="sentence-transformers/all-MiniLM-L6-v2"):
        # Removing API calls and using local models for long calls
        # API_KEY = os.getenv("HF_API_KEY")
        # if API_KEY:
        #     logger.info("API Key Loaded")
        # else:
        #     logger.error("Unable to load HuggingFace API key")
        
        # self.model = model
        # self.client = InferenceClient(
        #     model=self.model,
        #     provider="hf-inference",
        #     api_key=API_KEY
        # )
        
        self.model_name = model_name
        self.model = None
        
    def load_model(self):
        """
        Loading HuggingFace model and embed

        Args:
            text (str): text to get embed
        """
        try:
            self.model = SentenceTransformer(
                model_name_or_path=self.model_name
            )
            logger.info("Model loaded successfully")
        except OSError as e:
            logger.error(f"Unfortunatly Model is not loaded: {self.model_name}: {e}")
            raise OSError("Unfortunatly Model is not loaded") from e
        
        except Exception as e:
            logger.error(f"Unfortunatly results are not created: {e}")
            raise RuntimeError("Unfortunatly results are not created") from e
        
    def generate_embeddings(self, texts:List[str]):
        """
        Generating Embeddings

        Args:
            texts (List[str]): List of Texts
        """
        try:
            embeddings = self.model.encode(texts)
            logger.info("Embeddings Generated")
            return embeddings
        except Exception as e:
            logger.error(f"Got some unexpected error: {e}")
            raise Exception("Got some unexpected error") from e

