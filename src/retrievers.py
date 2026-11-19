"""Sections 7-10: the two retrievers. Retrieval only - no answer generation here."""
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import PATHS, CONFIG


class TfidfRetriever:
    """MODEL 1 retriever: TF-IDF vectors + cosine similarity."""
    def __init__(self, chunks_df):
        self.chunks = chunks_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english",
                                          ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.chunks["text"])

    def search(self, query, k=None):
        k = k or CONFIG["top_k"]
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]

    def save(self):
        with open(PATHS["tfidf"], "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.matrix}, f)

    @classmethod
    def load(cls, chunks_df):
        obj = cls.__new__(cls)
        obj.chunks = chunks_df.reset_index(drop=True)
        with open(PATHS["tfidf"], "rb") as f:
            d = pickle.load(f)
        obj.vectorizer, obj.matrix = d["vectorizer"], d["matrix"]
        return obj


class FaissRetriever:
    """MODEL 2 retriever: MiniLM sentence embeddings + FAISS inner-product index.
    Vectors are L2-normalised, so inner product == cosine similarity."""
    def __init__(self, chunks_df, index=None):
        import faiss  # imported here so Model 1 works even without faiss
        from sentence_transformers import SentenceTransformer
        self.faiss = faiss
        self.chunks = chunks_df.reset_index(drop=True)
        self.encoder = SentenceTransformer(CONFIG["embed_model"])
        if index is None:
            emb = self.encoder.encode(self.chunks["text"].tolist(), batch_size=64,
                                      show_progress_bar=True, normalize_embeddings=True)
            emb = np.asarray(emb, dtype="float32")
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
        self.index = index

    def search(self, query, k=None):
        k = min(k or CONFIG["top_k"], self.index.ntotal)
        q = self.encoder.encode([query], normalize_embeddings=True)
        sims, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i >= 0]

    def save(self):
        self.faiss.write_index(self.index, str(PATHS["faiss"]))

    @classmethod
    def load(cls, chunks_df):
        import faiss
        index = faiss.read_index(str(PATHS["faiss"]))
        if index.ntotal != len(chunks_df):
            raise ValueError("FAISS index and chunks.csv are out of sync. Rebuild the index.")
        return cls(chunks_df, index=index)
