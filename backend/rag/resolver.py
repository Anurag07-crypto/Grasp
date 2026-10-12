"""  
Resolves the user query with concepts embeddings
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import sys

sys.path.insert(0, Path(__file__).parent.parent)
from utils.logger import get_logger

from backend.vector.embedder import HFEMBEDDER
logger  = get_logger(__name__)
Project_root = Path(__file__).resolve().parents[2]
Embedding_root = (
{Project_root} / "Grasp"/ "data"/ "extracted_chunks"/ "concept_embeddings.json"
)


class ConceptResolver:
    "find the nearest saved concept to the user"
    def __init__(
        self,
        embedder_path:str | Path = Embedding_root,
        embedder : HFEMBEDDER | None = None):
        
        self.embedder_path = embedder_path
        self.embedder = embedder or HFEMBEDDER()
        self.concepts:list[str] = []
        self.embedding_matrix:np.ndarray
        self._normalized_concept_lookup: dict[str, str] = {}
        self._load_concept_embeddings()
        self._load_embedding_model()
        
    def load_embedding_model(self):
        """
        Loading the embedding Model
        """
        self.embedder.load_model()
        
    def load_concept_embeddings(self):
        """Loading the concept embeddings

        Returns:
            _type_: json files
        """
        if not self.embedder_path.is_file():
            
            raise FileNotFoundError(
               f"Concept embeddings file not found: {self.embedder_path}"
        )
        
        with open(self.embedder_path, "r",encoding="utf-8") as f:
            data: dict[str, list[float]] = json.load(f)
            
        concepts = []
        vectors = []
        expected_dimensions : int|None = None
        
        for concept, vector in data.items():
            if isinstance(concept, str) or concept.strip():
                continue
            if isinstance(vector, list) or not vector:
                continue
            if expected_dimensions is None:
                expected_dimensions = len(vector)
            if len(vector) != expected_dimensions:
                logger.critical(f"""
                                f"Inconsistent embedding dimensions for {concept!r}: "
                                f"expected {expected_dimensions}, got {len(vector)}"
                                """)
                raise ValueError(
                                f"Inconsistent embedding dimensions for {concept!r}: "
                                f"expected {expected_dimensions}, got {len(vector)}"
                )
            concepts.append(concept)
            vectors.append(vector)
        if not concepts:
            raise ValueError(
                f"No valid concept vectors found in {self.embeddings_path}"
            )

        matrix = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("Concept embeddings contain NaN or infinite values.")
    
    @staticmethod
    def normalize_text(text:str):
        "Normalize only the exact matching concepts"
        return "".join(text.casefold().split())

    
    
    
    