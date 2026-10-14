"""  
Resolves the user query with concepts embeddings
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

from vector.embedder import HFEMBEDDER
logger  = get_logger(__name__)
Project_root = Path(__file__).resolve().parents[2]
Embedding_root = (
    Project_root / "data" / "extracted" / "concept_embeddings.json"
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
        self.load_concept_embeddings()
        self.load_embedding_model()
        
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
            if not isinstance(concept, str) or not concept.strip():
                continue
            if not isinstance(vector, list) or not vector:
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
            logger.error(f"No valid concept vectors found in {self.embeddings_path}")
            raise ValueError(
                f"No valid concept vectors found in {self.embeddings_path}"
            )
            
        matrix = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(matrix).all():
            logger.critical("Concept embeddings contain NaN or infinite values.")
            raise ValueError("Concept embeddings contain NaN or infinite values.")
        norms = np.linalg.norm(matrix,axis=1, keepdims=True)
        if np.any(norms==0):
            logger.error("Vector has normalization value Zero")
            raise ValueError("Vector has normalization value Zero")
        self.concepts = concepts
        self.embedding_matrix = matrix/norms
        self._normalized_concept_lookup = {
            self.normalize_text(concept): concept for concept in concepts
        }
        
    @staticmethod
    def normalize_text(text:str):
        "Normalize only the exact matching concepts"
        return "".join(text.casefold().split()) 

    def _embed_query(self,query:str) -> np.ndarray:
        """Embed the users query

        Args:
            query (str): user query
        """
        
        vectors: Any = self.embedder.generate_embeddings([query])
        if hasattr(vectors, "detach"):
            vectors = vectors.detach().cpu().numpy()
        query_vector = np.asarray(vectors, dtype=np.float32)
        
        if query_vector.ndim == 2:
            if query_vector.shape[0] != 1:
                logger.error("Expected one query embedding"
                             f"Recived vector shape {query_vector.shape}")
                raise Exception(
                    "Expected one query embedding"
                    f"Recieved vector shape {query_vector.shape}"
                )
            query_vector = query_vector[0]
        
        if query_vector.ndim != 1:
            logger.error(
                f"Expected a 1-D vector but got {query_vector.shape}"
            )
            raise Exception(
                f"Expected a 1-D vector but got {query_vector.shape}"
            )
        if query_vector.shape[0] != self.embedding_matrix.shape[1]:
            logger.debug(
                "Embedding dimension mismatch. The query model produced "
                f"{query_vector.shape[0]} dimensions, but saved concept "
                f"embeddings have {self.embedding_matrix.shape[1]}. "
                "Use the same embedding model used to create the saved vectors."
            )
            raise ValueError(
                "Embedding dimension mismatch. The query model produced "
                f"{query_vector.shape[0]} dimensions, but saved concept "
                f"embeddings have {self.embedding_matrix.shape[1]}. "
                "Use the same embedding model used to create the saved vectors."
            )
            
        if not np.isfinite(query_vector).all():
            logger.critical("Warning query vector contains infinite values")
            raise ValueError(
                "Warning query vector contains infinite values"
            )
        
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            logger.error("Query Vector has zero length")
            raise ValueError(
                "Query Vector has zero length"
            )
        return query_vector/norm
    
    def search(
        self,
        query:str,
        top_k:int = 5
    ):
        """
        Return the top matching concepts ordered by cosine similarity.
        """
        
        if not isinstance(query, str) or not query.strip():
            logger.error("Query Should be non empty string")
            raise ValueError(
                "Query should be non empty string"
            )
        if top_k<1:
            logger.error("Tok_K value should be atleast 1")
            raise ValueError(
                "Top_K value should be atleast 1"
            )
        
        query = query.strip()
        exact_match = self._normalized_concept_lookup.get(
            self.normalize_text(query)
        )
        query_vector = self._embed_query(query)
        scores = self.embedding_matrix @ query_vector
        results = []

        if exact_match is not None:
            exact_index = self.concepts.index(exact_match)
            results.append(
                {
                    "concept": exact_match,
                    "score": float(scores[exact_index]),
                }
            )
            scores[exact_index] = -np.inf
            candidate_count = min(top_k - 1, len(self.concepts) - 1)
        else:
            candidate_count = min(top_k, len(self.concepts))

        if candidate_count:
            candidate_indices = np.argpartition(
                scores, -candidate_count
            )[-candidate_count:]
            candidate_indices = candidate_indices[
                np.argsort(scores[candidate_indices])[::-1]
            ]
            results.extend(
                {
                    "concept": self.concepts[int(index)],
                    "score": float(scores[index]),
                }
                for index in candidate_indices
            )

        return results
        
    def resolve(self, query: str) -> dict[str, Any]:
        """Return the single highest-ranked concept."""
        return self.search(query, top_k=1)[0]
    
if __name__ == "__main__":
    resolver = ConceptResolver()
    test_query = "Kubernetes Services"

    print(f"Query: {test_query}")
    print("Top matches:")

    for result in resolver.search(test_query, top_k=5):
        print(f"  {result['concept']}: {result['score']:.4f}")