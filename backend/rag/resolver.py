"""  
Resolves the user query with concepts embeddings
"""

import json
from pathlib import Path
from typing import Any
import numpy as np

from backend.vector.embedder import HFEMBEDDER

Project_root = Path(__file__).resolve().parents[2]
Embedding_root = (
{Project_root} / "Grasp"/ "data"/ "extracted_chunks"/ "concept_embeddings.json"
)


class ConceptResolver:
    "find the nearest saved concept to the user"
    def __init__(
        self,
        emdedder_path:str | Path = Embedding_root,
        embedder : HFEMBEDDER | None = None):
        
        self.embedder_path = self.embedder_path
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
        