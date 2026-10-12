# Grasp

Grasp is a Python project for turning technical documentation into a structured learning graph. It ingests source material, splits it into chunks, extracts key concepts and prerequisite relationships, builds a graph of those concepts, and prepares embeddings for retrieval-based learning workflows.

This repository is currently centered on the backend/data pipeline rather than a fully shipped end-user application.

## Current project status

The codebase currently includes:

- PDF corpus ingestion and chunking
- Concept extraction from text chunks
- Graph construction using NetworkX
- Embedding generation with sentence-transformers
- Data artifacts for chunks, extracted concepts, graph storage, and vector storage

This repo is still in an active development stage. Some earlier project-planning documents mention a more complete frontend and service architecture, but the current checked-in repository mainly contains the Python backend pipeline and supporting data directories.

## Repository structure

```text
Grasp/
├── backend/
│   ├── config.py
│   ├── main.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── extracter.py
│   │   └── prompt.py
│   ├── graph/
│   │   ├── builder.py
│   │   └── store.py
│   ├── utils/
│   │   └── logger.py
│   └── vector/
│       └── embedder.py
├── corpora/
│   ├── Concepts.pdf
│   ├── Reference.pdf
│   ├── Setup.pdf
│   ├── Tasks.pdf
│   ├── Tutorials.pdf
│   └── kubectl-commands.pdf
├── data/
│   ├── chroma/
│   ├── chunks/
│   ├── extracted/
│   ├── extracted_chunks/
│   └── graph/
├── .gitignore
├── LICENSE
├── README.md
├── render.yaml
├── requirements.txt
└── project_structure.txt
```

## Tech stack

The current implementation uses:

- Python
- FastAPI
- NetworkX
- ChromaDB
- PyMuPDF / PyPDF
- LangChain
- sentence-transformers
- Hugging Face ecosystem
- Groq-based concept extraction

## Core workflow

The project is designed around this flow:

1. Load PDFs from the `corpora/` directory
2. Split them into chunks using a recursive text splitter
3. Extract concept names and prerequisite relationships using an LLM prompt
4. Build a dependency graph between concepts
5. Generate embeddings for concepts or text
6. Store the processed outputs in `data/`

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment variables

The repo expects environment configuration for model access. Create a `.env` file in the project root with values such as:

```env
GROQ_API_KEY=your_groq_key
HF_API_KEY=your_huggingface_key
```

The project uses `python-dotenv` via `backend/config.py` to load these values.

## Example usage

The main logic is organized in the backend modules. For example:

- `backend/ingestion/chunker.py` handles corpus ingestion and chunk creation
- `backend/ingestion/extracter.py` extracts concept and prerequisite data
- `backend/graph/builder.py` builds the graph model from extracted chunks
- `backend/vector/embedder.py` handles embeddings with sentence-transformers

A typical working flow is:

```python
from backend.ingestion.chunker import Ingestion_pipeline

ingest = Ingestion_pipeline()
chunks = ingest.load_corpus_chunking("corpora")
```

Then concept extraction and graph generation can be run from the corresponding backend modules.

## Data folders

- `corpora/` contains source PDFs used for training/document ingestion
- `data/chunks/` stores normalized chunk outputs
- `data/extracted/` and `data/extracted_chunks/` store extracted concept data
- `data/graph/` stores graph artifacts
- `data/chroma/` is used for vector-storage-related runtime data

## Notes

- The repo is currently Python-only and does not include a complete frontend app in the checked-in code.
- Some earlier design documents describe a richer UI and API layout, but the actual repository content at this point is primarily the pipeline and data layer.
- The project is best understood as a knowledge-graph + retrieval prototype rather than a production-ready application.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
