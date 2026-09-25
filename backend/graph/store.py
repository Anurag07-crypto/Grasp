import chromadb
import uuid
import sys
from pathlib import Path
from typing import List, Any
import os
import numpy as np
sys.path.insert(0, str(Path(__name__).parent.parent))
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
        self._load_initialization()
        
    def load_initialization(
        self
    ):
        """
        Initializing the vector Store
        """
        try:
            os.makedirs(self.persistant_path, exist_ok=True)
            self.client = chromadb.PersistentClient(
                self.persistant_path,
                metadata={"description":"Concepts extraction texts"}
            )
            self.collection = self.client.get_or_create_collection(
                self.collection_name
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
        documents:List[Any],
        embeddings:np.ndarray
    ):
        """
        Adding documents to the vector store

        Args:
            documents (List[Any]): list of documents 
            embeddings (np.ndarray): embeddings
        """
        
        if len(documents) != len(embeddings):
            logger.critical("Length of documents and embeddings should be same")
        
        ids = []
        document_texts = []
        metadatas = []
        embedding_lists = []
        
        