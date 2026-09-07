import networkx as nx 
import json
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
INPUT_PATH = Path("data/extracted/normalized_chunks.json")
OUTPUT_PATH = Path("data/graph/graph.pkl")

def load_json():
    with open(INPUT_PATH, "r",encoding="utf-8") as f:
        return json.load(f)
    
def build_graph(chunks):
    graph = nx.DiGraph()

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        for concept in chunk.get("concepts",[]):
            name = concept.get("name")
            prerequisites = concept.get("prerequisites", [])

            if not name:
                continue
            graph.add_node(
                name,
                source_chunk_ids = [chunk_id]
            )

            for prerequisite in prerequisites:
                if not prerequisite:
                    continue
                graph.add_node(prerequisite)
                graph.add_edge(
                    prerequisite,
                    name
                )
    return graph

if __name__ == "__main__":
    chunks = load_json()

    graph = build_graph(chunks)

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    
    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(graph, seed=42)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        arrows=True,
        font_size=10
    )

    plt.title("Skill Dependency Graph")
    plt.show()
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(graph, f)
        print(f"Graph saved to {OUTPUT_PATH}")
