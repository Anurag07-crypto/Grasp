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
        
    def _load_graph(self) -> nx.DiGraph:
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
        
        if not isinstance(graph, nx.DiGraph):
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
    
    def get_all_prerequisites(self, concept: str):
        """ 
        getting prerequisities at very depth level
        """
        if concept not in self.graph:
            logger.error("Concept not found in graph")
            raise ValueError("Concept not found in graph")
        
        logger.info("Getting prerequisites from the anccestors")
        return nx.ancestors(self.graph, concept)
    
    def get_learning_order(self, concept:str):
        """
        Getting the learning order to make graph more efficient

        Args:
            concept (str): concept 
        """
        if concept not in self.graph:
            logger.error("Concept not found in graph")
            raise ValueError("Concept not found in graph")
        
        ancestors = nx.ancestors(self.graph, concept)
        relevant_nodes = ancestors | {concept}
        
        subgraph = self.graph.subgraph(relevant_nodes)

        # A cycle means those concepts cannot be given a strict order relative
        # to one another. Collapse each strongly connected component into one
        # unit, then order those units by their prerequisite edges. This keeps
        # every relevant concept in the result and preserves all acyclic
        # prerequisite relationships.
        condensed = nx.condensation(subgraph)
        components = condensed.graph["mapping"]
        ordered_components = []
        for generation in nx.topological_generations(condensed):
            ordered_components.extend(sorted(generation, key=lambda node: min(condensed.nodes[node]["members"])))

        learning_order = []
        for component in ordered_components:
            members = sorted(condensed.nodes[component]["members"])
            # Keep the requested concept at the end when it shares a cycle
            # with other concepts.
            if concept in members:
                members.remove(concept)
                members.append(concept)
            learning_order.extend(members)

        return learning_order
        
if __name__ == "__main__":
    traversal = GraphTraversal()

    print("Graph loaded successfully!")
    print("Nodes:", traversal.graph.number_of_nodes())
    print("Edges:", traversal.graph.number_of_edges())

    target = "Pods"
    prerequisites = traversal.get_direct_prerequisites(target)
    all_prerequisites = traversal.get_all_prerequisites("Pods")

    print("Total prerequisite ancestors:", len(all_prerequisites))
    print("Sample ancestors:", sorted(all_prerequisites)[:20])
    print("Target:", target)
    print("Direct prerequisites:", prerequisites)
    learning_order = traversal.get_learning_order("Pods")

    print("Learning path length:", len(learning_order))
    print("First 20 concepts:", learning_order[:20])
    print("Last concept:", learning_order[-1])
