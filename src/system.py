"""Build / load the indexes (Sections 6-9, 18)."""
import pandas as pd
from .config import PATHS
from .ingest import build_chunks
from .retrievers import TfidfRetriever, FaissRetriever
from .qa_models import BaselineQA, AdvancedQA


def build_all(verbose=True):
    """Re-extract PDFs, rebuild chunks, TF-IDF and FAISS index, and save them."""
    chunks = build_chunks(verbose=verbose)
    tfidf = TfidfRetriever(chunks); tfidf.save()
    faiss_r = FaissRetriever(chunks); faiss_r.save()
    if verbose:
        print(f"Indexes saved in {PATHS['index_dir']}")
    return chunks, tfidf, faiss_r


def indexes_exist():
    return PATHS["chunks"].exists() and PATHS["tfidf"].exists() and PATHS["faiss"].exists()


def load_all():
    chunks = pd.read_csv(PATHS["chunks"])
    return chunks, TfidfRetriever.load(chunks), FaissRetriever.load(chunks)


def get_models(load_llm=True, rebuild=False):
    """Return (BaselineQA, AdvancedQA)."""
    if rebuild or not indexes_exist():
        _, tfidf, faiss_r = build_all()
    else:
        _, tfidf, faiss_r = load_all()
    return BaselineQA(tfidf), AdvancedQA(faiss_r, load_llm=load_llm)
