"""Document ingestion pipeline for loading, splitting, embedding, and indexing in ChromaDB."""

import os
import glob
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from rag.embeddings import get_embeddings

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "docs")


def load_file(file_path: str) -> List[Document]:
    """Loads a single PDF, TXT, or MD file into a list of Documents."""
    ext = os.path.splitext(file_path)[1].lower()
    docs: List[Document] = []
    
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif ext in [".txt", ".md", ".markdown"]:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        else:
            print(f"[SKIP] Unsupported file format: {file_path}")
            return []
    except Exception as e:
        print(f"[ERROR] Error loading {file_path}: {e}")
        return []

    # Ensure source metadata is the basename for clean citation
    for doc in docs:
        doc.metadata["source"] = os.path.basename(file_path)

    return docs


def load_directory(docs_dir: str = DOCS_DIR) -> List[Document]:
    """Loads all supported documents (.pdf, .txt, .md) from a directory."""
    all_docs: List[Document] = []
    supported_extensions = ["*.pdf", "*.txt", "*.md"]
    
    for ext in supported_extensions:
        pattern = os.path.join(docs_dir, "**", ext)
        for file_path in glob.glob(pattern, recursive=True):
            loaded = load_file(file_path)
            all_docs.extend(loaded)

    return all_docs


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Splits documents into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_documents(documents)


def get_vectorstore(persist_dir: str = CHROMA_DIR) -> Chroma:
    """Returns the persistent Chroma vector store instance."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="knowledge_base",
    )


def ingest_documents(
    docs_dir: str = DOCS_DIR,
    persist_dir: str = CHROMA_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """Loads, splits, embeds, and indexes all documents from docs_dir into Chroma."""
    docs = load_directory(docs_dir)
    if not docs:
        print(f"[INFO] No documents found in {docs_dir}")
        return 0

    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[INFO] Split {len(docs)} document pages into {len(chunks)} chunks.")

    vectorstore = get_vectorstore(persist_dir=persist_dir)
    vectorstore.add_documents(chunks)
    print(f"[SUCCESS] Successfully indexed {len(chunks)} chunks into ChromaDB at '{persist_dir}'.")
    return len(chunks)


def index_single_document(file_path: str, persist_dir: str = CHROMA_DIR) -> int:
    """Loads and indexes a single uploaded file."""
    docs = load_file(file_path)
    if not docs:
        return 0
    chunks = split_documents(docs)
    vectorstore = get_vectorstore(persist_dir=persist_dir)
    vectorstore.add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else DOCS_DIR
    print(f"Ingesting documents from: {target_dir}")
    count = ingest_documents(docs_dir=target_dir)
    print(f"Total chunks indexed: {count}")
