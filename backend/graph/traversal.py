import networkx as nx
from typing import Hashable
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

graph_path = PROJECT_ROOT / "data" / "graph" / "graph.pkl"

class GraphTraversal:
    def __init__(
        self,
        graph_path:Path=graph_path,
        ):
        self.graph_path = graph_path
        self.graph = self._load_graph()
        
    def load_graph(self) -> nx.DiGraph:
        """
        Loading the existing graph

        Args:
            graph_path (_type_, optional):Prebuild graph Path. Defaults to graph_path.

        Returns:
            nx.digraph: graph
        """
        
        if not self.graph_path.exists():
            logger.error("Graph Not Found")
            raise FileNotFoundError(
                "Graph file not found"
            )
        with open(self.graph_path,"rb") as g:
            graph = pickle.load(g)
        
        if not isinstance(self.graph, nx.DiGraph):
            logger.debug("Saved object is not network Digraph")
            raise TypeError(
                "Saved object is not network Digraph"
            )
        return graph
    
    def get_direct_prerequisites(self, concept: str):
        """
        Getting Prerequisites directly from the concepts
        """
        
        if concept not in self.graph:
            logger.error("Concepts not found in graph")
            raise ValueError("Concepts not found in graph")
        
        return list(self.graph.predecessors(concept))
        
