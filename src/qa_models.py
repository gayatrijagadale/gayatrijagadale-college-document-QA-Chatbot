# """Sections 7, 11, 12: the two complete QA systems. Both return the same result format."""
# import time
# import re
# from .config import CONFIG, NOT_FOUND_MSG
# from .retrievers import TfidfRetriever, FaissRetriever


# def _pack(retriever, hits):
#     """Turn (chunk_idx, score) pairs into passage dicts with citation fields."""
#     out = []
#     for idx, score in hits:
#         row = retriever.chunks.iloc[idx]
#         out.append({"doc_name": row["doc_name"], "page_no": int(row["page_no"]),
#                     "text": row["text"], "score": float(score)})
#     return out


# def format_citations(passages):
#     seen, cites = set(), []
#     for p in passages:
#         key = (p["doc_name"], p["page_no"])
#         if key not in seen:
#             seen.add(key)
#             cites.append(f"[{p['doc_name']}, PDF page {p['page_no']}]")
#     return cites


# class BaselineQA:
#     """MODEL 1: TF-IDF retrieval + extractive answer (the best passage itself)."""
#     name = "Baseline (TF-IDF)"

#     def __init__(self, retriever: TfidfRetriever):
#         self.retriever = retriever

#     def retrieve(self, question):
#         return _pack(self.retriever, self.retriever.search(question, CONFIG["top_k"]))

#     def answer(self, question):
#         t0 = time.perf_counter()
#         passages = self.retrieve(question)
#         if not passages or passages[0]["score"] < CONFIG["tfidf_threshold"]:
#             return {"answer": NOT_FOUND_MSG, "found": False, "citations": [],
#                     "passages": passages, "time": time.perf_counter() - t0}
#         best = passages[0]
#         return {"answer": best["text"], "found": True,
#                 "citations": [f"[{best['doc_name']}, PDF page {best['page_no']}]"],
#                 "passages": passages, "time": time.perf_counter() - t0}


# class AdvancedQA:
#     """MODEL 2: FAISS semantic retrieval + FLAN-T5 generation from retrieved context only."""
#     name = "Advanced (MiniLM + FAISS + FLAN-T5)"

#     def __init__(self, retriever: FaissRetriever, load_llm=True):
#         self.retriever = retriever
#         self.tokenizer = self.model = None
#         if load_llm:
#             self._load_llm()

#     def _load_llm(self):
#         import torch
#         from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#         self.torch = torch
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_model"])
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["llm_model"]).to(self.device).eval()

#     def retrieve(self, question):
#         return _pack(self.retriever, self.retriever.search(question, CONFIG["top_k"]))

#     def _generate(self, question, passages):
#         # Question first, context after: if the 512-token limit truncates anything,
#         # it cuts the end of the context, never the question.
#         context = " ".join(f"({i}) {p['text']}" for i, p in enumerate(passages, 1))
#         prompt = ("Answer the question using only the context below. "
#                   "If the context does not contain the answer, reply exactly: not found.\n"
#                   f"Question: {question}\nContext: {context}\nAnswer:")
#         enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
#                              max_length=CONFIG["max_input_tokens"]).to(self.device)
#         with self.torch.no_grad():
#             out = self.model.generate(**enc, max_new_tokens=CONFIG["max_new_tokens"],
#                                       num_beams=4, early_stopping=True)
#         return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

#     def answer(self, question):
#         t0 = time.perf_counter()
#         passages = self.retrieve(question)
#         # Fallback 1: retrieval confidence too low -> do not even call the LLM
#         if not passages or passages[0]["score"] < CONFIG["sim_threshold"]:
#             return {"answer": NOT_FOUND_MSG, "found": False, "citations": [],
#                     "passages": passages, "time": time.perf_counter() - t0}
#         used = [p for p in passages if p["score"] >= CONFIG["sim_threshold"]]
#         text = self._generate(question, used)
#         # Fallback 2: the LLM itself says the context has no answer
#         if (not text) or re.sub(r"[^a-z ]", "", text.lower()).strip() in ("not found", "unknown", "no answer"):
#             return {"answer": NOT_FOUND_MSG, "found": False, "citations": [],
#                     "passages": passages, "time": time.perf_counter() - t0}
#         return {"answer": text, "found": True, "citations": format_citations(used),
#                 "passages": passages, "time": time.perf_counter() - t0}


"""The two complete QA systems. Both return the same result format.

Add these keys to CONFIG in src/config.py:
    "use_reranker": True,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "candidates": 15,          # passages fetched before reranking
    "context_passages": 3,     # passages given to FLAN-T5
and change "llm_model" to "google/flan-t5-large" for noticeably better answers (needs ~3 GB RAM).
"""
import time
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .config import CONFIG, NOT_FOUND_MSG
from .retrievers import TfidfRetriever, FaissRetriever


def _pack(retriever, hits, docs=None):
    out = []
    for idx, score in hits:
        row = retriever.chunks.iloc[idx]
        if docs and row["doc_name"] not in docs:
            continue
        out.append({"doc_name": row["doc_name"], "page_no": int(row["page_no"]),
                    "text": row["text"], "score": float(score)})
    return out


def format_citations(passages):
    seen, cites = set(), []
    for p in passages:
        key = (p["doc_name"], p["page_no"])
        if key not in seen:
            seen.add(key)
            cites.append(f"[{p['doc_name']}, PDF page {p['page_no']}]")
    return cites


def _search_depth(docs, wanted):
    """When a document filter is active we must search deeper, then filter."""
    return 300 if docs else wanted


def _best_sentences(question, text, n=2):
    """Extractive answer: the n sentences sharing most content words with the question."""
    q = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in ENGLISH_STOP_WORDS}
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]
    if not sents:
        return text
    scored = [(len(q & set(re.findall(r"[a-z0-9]+", s.lower()))), i, s) for i, s in enumerate(sents)]
    top = sorted(scored, key=lambda t: (-t[0], t[1]))[:n]
    return " ".join(s for _, _, s in sorted(top, key=lambda t: t[1]))


class BaselineQA:
    """MODEL 1: TF-IDF retrieval + extractive answer (best sentences of the best passage)."""
    name = "Baseline (TF-IDF)"

    def __init__(self, retriever: TfidfRetriever):
        self.retriever = retriever

    def retrieve(self, question, docs=None):
        k = CONFIG["top_k"]
        hits = self.retriever.search(question, _search_depth(docs, k))
        return _pack(self.retriever, hits, docs)[:k]

    def answer(self, question, docs=None):
        t0 = time.perf_counter()
        passages = self.retrieve(question, docs)
        if not passages or passages[0]["score"] < CONFIG["tfidf_threshold"]:
            return {"answer": NOT_FOUND_MSG, "found": False, "citations": [],
                    "passages": passages, "top_score": passages[0]["score"] if passages else 0.0,
                    "time": time.perf_counter() - t0}
        best = passages[0]
        return {"answer": _best_sentences(question, best["text"]), "found": True,
                "citations": [f"[{best['doc_name']}, PDF page {best['page_no']}]"],
                "passages": passages, "top_score": best["score"], "time": time.perf_counter() - t0}


class AdvancedQA:
    """MODEL 2: FAISS retrieval -> cross-encoder rerank -> FLAN-T5 generation."""
    name = "Advanced (MiniLM + FAISS + FLAN-T5)"

    def __init__(self, retriever: FaissRetriever, load_llm=True):
        self.retriever = retriever
        self.tokenizer = self.model = self.reranker = None
        if load_llm:
            self._load_llm()

    def _load_llm(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_model"])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["llm_model"]).to(self.device).eval()
        if CONFIG.get("use_reranker"):
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(CONFIG["reranker_model"], device=self.device)

    def retrieve(self, question, docs=None):
        n = CONFIG.get("candidates", CONFIG["top_k"])
        hits = self.retriever.search(question, _search_depth(docs, n))
        return _pack(self.retriever, hits, docs)[:n]

    def _rerank(self, question, passages):
        if self.reranker is None or len(passages) < 2:
            return passages
        scores = self.reranker.predict([(question, p["text"]) for p in passages])
        for p, s in zip(passages, scores):
            p["rerank"] = float(s)
        return sorted(passages, key=lambda p: -p["rerank"])

    def _generate(self, question, passages):
        context = " ".join(f"({i}) {p['text']}" for i, p in enumerate(passages, 1))
        prompt = ("Answer the question using only the context below, in one or two complete sentences. "
                  "If the context does not contain the answer, reply exactly: not found.\n"
                  f"Question: {question}\nContext: {context}\nAnswer:")
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=CONFIG["max_input_tokens"]).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=CONFIG["max_new_tokens"],
                                      num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def answer(self, question, docs=None):
        t0 = time.perf_counter()
        cands = self.retrieve(question, docs)
        top = cands[0]["score"] if cands else 0.0
        if not cands or top < CONFIG["sim_threshold"]:           # Fallback 1: weak retrieval
            return {"answer": NOT_FOUND_MSG, "found": False, "citations": [], "passages": cands,
                    "top_score": top, "time": time.perf_counter() - t0}
        used = [p for p in cands if p["score"] >= CONFIG["sim_threshold"]]
        used = self._rerank(question, used)[:CONFIG.get("context_passages", 3)]
        text = self._generate(question, used)
        if (not text) or re.sub(r"[^a-z ]", "", text.lower()).strip() in ("not found", "unknown", "no answer"):
            return {"answer": NOT_FOUND_MSG, "found": False, "citations": [], "passages": used,
                    "top_score": top, "time": time.perf_counter() - t0}   # Fallback 2: LLM abstains
        return {"answer": text, "found": True, "citations": format_citations(used),
                "passages": used, "top_score": top, "time": time.perf_counter() - t0}