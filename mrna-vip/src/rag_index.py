"""Module 3 - retrieval index over the regulatory guideline document.

Pipeline: load text -> split into overlapping chunks (LangChain splitter) -> embed each chunk
-> persist in a local ChromaDB collection -> answer questions by similarity search.

Two embedding back-ends share one interface:
  * "minilm"  - sentence-transformers all-MiniLM-L6-v2 (the PRD's model; needs a one-time
                download from huggingface.co).
  * "tfidf"   - fully offline TF-IDF vectors; used automatically when MiniLM cannot be loaded
                (no internet / package missing) so the pipeline still runs. The back-end that
                was really used is stored in outputs/rag_meta.json and printed every run.
EMBEDDING_BACKEND=minilm|tfidf|auto (default auto) selects it.
"""
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import chromadb
import joblib
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR, CHUNK_OVERLAP_TOKENS, CHUNK_SEPARATORS, CHUNK_SIZE_TOKENS, COLLECTION_NAME,
    DEFAULT_K, EMBEDDING_MODEL, GUIDELINES_TXT, RAG_LOG, RAG_META,
)


def count_tokens(text: str) -> int:
    """Approximate token count: ~4 characters per token (OpenAI's English rule of thumb).

    Deterministic and offline. Real tokenizers differ by roughly +/-20%.
    """
    return math.ceil(len(text) / 4)


def split_document(text: str, chunk_size: int = CHUNK_SIZE_TOKENS,
                   overlap: int = CHUNK_OVERLAP_TOKENS, separators=None) -> list[str]:
    """Overlapping chunks measured in TOKENS (not characters) - FR-3.1."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, length_function=count_tokens,
        separators=separators or CHUNK_SEPARATORS,
    )
    return splitter.split_text(text)


# ----------------------------------------------------------------------------- embedders
class TfidfEmbedder:
    name = "tfidf"

    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer

    def fit(self, corpus_text: str):
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Fit the vocabulary/IDF on SENTENCES of the whole document so rare words get weight.
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", corpus_text) if s.strip()]
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
        self.vectorizer.fit(sentences)
        return self

    def embed(self, texts: list[str]) -> list[list[float]]:
        m = self.vectorizer.transform(texts).toarray()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (m / norms).tolist()

    def save(self, directory: Path):
        joblib.dump(self.vectorizer, Path(directory) / "tfidf_vectorizer.joblib")

    @classmethod
    def load(cls, directory: Path):
        return cls(joblib.load(Path(directory) / "tfidf_vectorizer.joblib"))


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        # The default window is 256 word-pieces; our chunks can be longer, so raise it to the
        # model's architectural maximum (512) to avoid silently ignoring the end of each chunk.
        self.model.max_seq_length = 512

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def save(self, directory: Path):  # nothing to save: weights come from the HF cache
        pass


def _make_embedder(backend: str, corpus_text: str | None = None, directory: Path | None = None):
    backend = (backend or os.environ.get("EMBEDDING_BACKEND", "auto")).lower()
    if backend in ("minilm", "auto"):
        try:
            return MiniLMEmbedder()
        except Exception as exc:  # no package / no internet
            if backend == "minilm":
                raise
            print(f"WARNING: MiniLM unavailable ({type(exc).__name__}); falling back to offline TF-IDF "
                  "embeddings. Install sentence-transformers + internet for the PRD model.")
    if directory is not None and (Path(directory) / "tfidf_vectorizer.joblib").exists():
        return TfidfEmbedder.load(directory)
    return TfidfEmbedder().fit(corpus_text)


# ----------------------------------------------------------------------------- index
@dataclass
class GuidelineIndex:
    collection: "chromadb.Collection"
    embedder: object
    persist_dir: Path

    @property
    def backend(self) -> str:
        return self.embedder.name

    def query(self, question: str, k: int = DEFAULT_K, log: bool = True):
        """Return (list_of_chunk_texts, latency_seconds). Chunks are ordered most-similar first."""
        start = time.perf_counter()
        q = self.embedder.embed([question])
        res = self.collection.query(query_embeddings=q, n_results=k)
        latency = max(time.perf_counter() - start, 1e-6)
        chunks = res["documents"][0]
        if log:
            RAG_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(RAG_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "question": question, "k": k, "latency_s": round(latency, 5),
                    "backend": self.backend, "top_chunk_id": res["ids"][0][0],
                    "distances": [round(d, 4) for d in res["distances"][0]],
                }) + "\n")
        return chunks, latency


def build_index(doc_path=GUIDELINES_TXT, persist_dir=CHROMA_DIR, backend: str | None = None,
                chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS,
                collection_name: str = COLLECTION_NAME, verbose: bool = True) -> GuidelineIndex:
    text = Path(doc_path).read_text(encoding="utf-8")
    chunks = split_document(text, chunk_size, overlap)
    embedder = _make_embedder(backend, corpus_text=text, directory=None)
    embeddings = embedder.embed(chunks)
    assert len(embeddings) == len(chunks)                      # FR-3.2

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name, embedding_function=None,
                                   configuration={"hnsw": {"space": "cosine"}})
    col.add(ids=[f"chunk_{i}" for i in range(len(chunks))], documents=chunks, embeddings=embeddings,
            metadatas=[{"tokens": count_tokens(c), "position": i} for i, c in enumerate(chunks)])
    embedder.save(persist_dir)
    RAG_META.write_text(json.dumps({
        "backend": embedder.name, "embedding_model": EMBEDDING_MODEL if embedder.name == "minilm" else "tfidf (offline fallback)",
        "n_chunks": len(chunks), "chunk_tokens": [count_tokens(c) for c in chunks],
        "chunk_size": chunk_size, "overlap": overlap}, indent=2))
    if verbose:
        print(f"Split guideline document into {len(chunks)} chunks; token lengths: "
              f"{[count_tokens(c) for c in chunks]}  (embedding backend: {embedder.name})")
    return GuidelineIndex(col, embedder, persist_dir)


def load_index(persist_dir=CHROMA_DIR, backend: str | None = None,
               collection_name: str = COLLECTION_NAME) -> GuidelineIndex:
    """Re-open a persisted index WITHOUT re-embedding anything (FR-3.3)."""
    persist_dir = Path(persist_dir)
    meta = json.loads(RAG_META.read_text()) if RAG_META.exists() else {}
    used = backend or meta.get("backend")
    if used == "tfidf":
        embedder = TfidfEmbedder.load(persist_dir)
    else:
        embedder = _make_embedder(used or "auto", directory=persist_dir)
    col = chromadb.PersistentClient(path=str(persist_dir)).get_collection(collection_name, embedding_function=None)
    return GuidelineIndex(col, embedder, persist_dir)


def query_guidelines(index: GuidelineIndex, question: str, k: int = DEFAULT_K):
    """Top-k most similar chunks + latency in seconds (FR-3.4, FR-3.5)."""
    chunks, latency = index.query(question, k)
    print(f"Retrieved {len(chunks)} chunks in {latency:.4f}s")
    return chunks, latency


if __name__ == "__main__":
    idx = build_index()
    for i, c in enumerate(query_guidelines(idx, "What are the storage requirements below -20C?")[0], 1):
        print(f"--- Result {i} ---\n{c}\n")
