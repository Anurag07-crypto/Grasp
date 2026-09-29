import chromadb
import json
import sys
from pathlib import Path
from typing import List, Any
import os
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)
class VECTORDB:
    def __init__(
        self,
        persistant_path:str,
        collection_name:str="concepts_texts"
    ):
        """
        Storing the Extracted_chunks Data into VectorStore

        Args:
            persistant_path (str): path where data is being saved
            collection_name (str, optional): Defaults to "concepts_texts".
        """
        self.persistant_path = persistant_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.load_initialization()
        
    def load_initialization(
        self
    ):
        """
        Initializing the vector Store
        """
        try:
            os.makedirs(self.persistant_path, exist_ok=True)
            self.client = chromadb.PersistentClient(
                self.persistant_path
            )
            self.collection = self.client.get_or_create_collection(
                self.collection_name,
                metadata={"description":"Original source text chunks"}
            )
            
            logger.info("The Store Initialized")
        except RuntimeError as e:
            logger.error(f"Store not initialized because of some Runtime error: {e}")
            raise RuntimeError("Store not initialized because of some Runtime error") from e
        except Exception as e:
            logger.error(f"Store got some unexpected error: {e}")
            raise Exception("Store got some unexpected error") from e
        
    def add_documents(
        self,
        documents:List[Any]
    ):
        """
        Adding documents to the vector store

        Args:
            documents (List[Any]): list of documents 
            embeddings (np.ndarray): embeddings
        """
        
        if not documents:
            logger.critical("Documents are not in comfortable conditions")
        
        ids = []
        document_texts = []
        metadatas = []
        
        for doc in documents:
            chunk_id = str(doc["id"])
            text = doc["text"]

            ids.append(str(chunk_id))
            document_texts.append(text)
            metadatas.append({"id": str(chunk_id)})
        logger.info(f"Starting to store {len(ids)} chunks...")
        self.collection.upsert(
            ids=ids,
            documents=document_texts,
            metadatas=metadatas
        )

        logger.info(f"Finished storing chunks. Collection count: {self.collection.count()}")

with open("data/chunks/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

vectordb = VECTORDB(
    persistant_path="data/chroma",
    collection_name="source_chunks"
)

vectordb.add_documents(chunks)
result = vectordb.collection.get(
    ids=["Concepts_p0_c0000"],
    include=["documents", "metadatas"]
)

print(result)