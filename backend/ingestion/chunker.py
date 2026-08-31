from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__name__).parent.parent))
from Grasp.backend.utils.logger import get_logger
import json
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

output_path = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
class Ingestion_pipeline:
    def __init__(self):
        self.chunks = None
        self.doc_load = None
    def load_corpus_chunking(self,corpus_path:str):
        """_summary_

        Args:
            corpus_path (str): _description_
        """
        
        try:
            loader = DirectoryLoader(
                path=corpus_path,
                glob="**/*.pdf",
                loader_cls=PyMuPDFLoader,
                show_progress=True
            )
            self.doc_load = loader.load()
            logger.info(f"Files Chunked Successfully of this DIR - {corpus_path}")
            
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["/n/n","/n",""],
                    chunk_size=2000,
                    chunk_overlap=200
                )
                self.chunks = text_splitter.split_documents(self.doc_load)
                logger.info("Chunks created successfully")
                normalized_chunks = self.normalize()
                self.save_chunks(normalized_chunks,
                                 output_path)
                return normalized_chunks
            except RuntimeError as e:
                logger.error(f"Chunks are not created successfully: {e}")
                raise RuntimeError("Chunks are not created successfully") from e
        except Exception as e:
            logger.error(f"Directory not Loaded Successfully: {e}")
            raise Exception("Directory not Loaded Successfully") from e
            
    def normalize(self):
        """_summary_

        Args:
            docs (_type_): _description_
        """

        try:
            chunks = []
            for index, document in enumerate(self.chunks):
                source = document.metadata.get("source"," ")
                page = document.metadata.get("page")
                source_name = Path(source).stem
                chunk = {
                "id": f"{source_name}_p{page}_c{index:04d}",
                "text": document.page_content.strip(),
                "source": source,
                "page": page,
                }

                chunks.append(chunk)
            logger.info("Chunk Normalized")
            return chunks
        except Exception as e:
            logger.error(f"Chunk not normalized properly: {e}")
            raise Exception("Chunks not normalized properly") from e
        
    def save_chunks(
        self,
        chunks: list[dict],
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(
    f"Chunks saved successfully: {output_path}"
)
        

ingest = Ingestion_pipeline()
chunks = ingest.load_corpus_chunking("C:/Users/Lenovo/Desktop/Skill-graph/Grasp/corpora")