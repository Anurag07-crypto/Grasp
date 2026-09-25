import networkx as nx 
import json
from pathlib import Path
import pickle
INPUT_PATH = Path("data/extracted/extracted_chunks.json")
OUTPUT_PATH = Path("data/graph/graph.pkl")

def load_json():
    with open(INPUT_PATH, "r",encoding="utf-8") as f:
        return json.load(f)
    
def build_graph(chunks):
    graph = nx.DiGraph()

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")

        for concept in chunk.get("concepts", []):
            name = concept.get("name")

            if not isinstance(name, str) or not name.strip():
                continue

            name = name.strip()

            if not graph.has_node(name):
                graph.add_node(name, source_chunk_ids=[])

            graph.nodes[name].setdefault("source_chunk_ids", [])

            if chunk_id and chunk_id not in graph.nodes[name]["source_chunk_ids"]:
                graph.nodes[name]["source_chunk_ids"].append(chunk_id)

            prerequisites = concept.get("prerequisites", [])

            if not isinstance(prerequisites, list):
                continue

            for prerequisite in prerequisites:
                if not isinstance(prerequisite, str) or not prerequisite.strip():
                    continue

                prerequisite = prerequisite.strip()

                if not graph.has_node(prerequisite):
                    graph.add_node(prerequisite, source_chunk_ids=[])

                graph.add_edge(prerequisite, name)

    return graph

from collections import Counter
if __name__ == "__main__":

    chunks = load_json()
    print("Chunks:", len(chunks))

    prereq_types = Counter()
    examples = []

    for chunk in chunks:
        for concept in chunk.get("concepts", []):
            prereqs = concept.get("prerequisites", [])

            prereq_types[type(prereqs).__name__] += 1

            if prereqs and len(examples) < 5:
                examples.append({
                    "concept": concept.get("name"),
                    "prerequisites": prereqs,
                })

    print("Prerequisite field types:", prereq_types)
    print("Examples:")
    for item in examples:
        print(item)
    graph = build_graph(chunks)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(graph, f)

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    
