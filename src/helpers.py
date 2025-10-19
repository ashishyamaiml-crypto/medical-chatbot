from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any
from pathlib import Path
import os

# Try to import Document from langchain; fall back to a lightweight dataclass
try:
    from langchain.schema import Document  # type: ignore
except Exception:
    from dataclasses import dataclass, field

    @dataclass
    class Document:
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)


def load_pdf_file(data):
    """Load PDF files from `data` path. Accepts a path string (absolute or
    relative). If the path doesn't exist, try a few repository-relative
    candidates before raising a helpful FileNotFoundError.
    """
    p = Path(data)
    if not p.exists():
        repo_root = Path(__file__).resolve().parent.parent
        candidates = [
            repo_root / data,
            repo_root / "data",
            repo_root / "data" / "",
        ]
        found = None
        for c in candidates:
            if c.exists() and c.is_dir():
                found = c
                break
        if found is None:
            tried = [str(p)] + [str(c) for c in candidates]
            raise FileNotFoundError(
                f"Directory not found: '{data}'. Tried: {tried}.\n"
                "Run the script from the project root or pass an absolute path."
            )
        p = found

    loader = DirectoryLoader(str(p), glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Return Documents containing only page_content and a metadata.source."""
    minimal_docs: List[Document] = []
    for doc in docs:
        src = None
        if isinstance(doc.metadata, dict):
            src = doc.metadata.get("source")
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
    return minimal_docs


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_hugging_face_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return embeddings
    except Exception:
        # Let the caller see the traceback; print a helpful message first
        print(f"Failed to load embeddings model: {model_name}")
        raise