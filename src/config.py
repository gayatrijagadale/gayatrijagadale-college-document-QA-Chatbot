"""Central configuration. Edit values here, nothing else needs changing."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "raw": ROOT / "data" / "raw_pdfs",
    "processed": ROOT / "data" / "processed",
    "sources": ROOT / "data" / "sources.csv",
    "chunks": ROOT / "data" / "processed" / "chunks.csv",
    "index_dir": ROOT / "index",
    "tfidf": ROOT / "index" / "tfidf.pkl",
    "faiss": ROOT / "index" / "faiss.index",
    "test_set": ROOT / "eval" / "test_set.csv",
    "eval_dir": ROOT / "eval",
    "figures": ROOT / "eval" / "figures",
}

CONFIG = {
    "chunk_words": 200,        # words per chunk
    "chunk_overlap": 40,       # overlapping words between neighbouring chunks
    "min_chunk_words": 15,     # tiny chunks (headers, page numbers) are dropped
    "top_k": 5,                # passages retrieved by BOTH models
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "google/flan-t5-base",
    "sim_threshold": 0.20,     # Model 2: min cosine similarity of top-1 passage, else "not found"
    "tfidf_threshold": 0.05,   # Model 1: min TF-IDF cosine of top-1 passage, else "not found"
    "max_input_tokens": 512,   # FLAN-T5 input limit
    "max_new_tokens": 128,
}

NOT_FOUND_MSG = "This information was not found in the provided documents."

for _p in (PATHS["raw"], PATHS["processed"], PATHS["index_dir"], PATHS["eval_dir"], PATHS["figures"]):
    _p.mkdir(parents=True, exist_ok=True)
