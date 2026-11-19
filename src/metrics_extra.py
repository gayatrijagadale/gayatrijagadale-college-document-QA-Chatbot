"""Extra metrics (put in src/metrics_extra.py). Plain Python/numpy so each formula is explainable.

Usage inside evaluate.py is shown at the bottom of this file."""
import math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .metrics import normalize


# ---------- retrieval ----------
def _flags(retrieved, gold_doc, gold_pages, k):
    gold = {(gold_doc.lower(), int(p)) for p in gold_pages}
    return [(d.lower(), int(p)) in gold for d, p in retrieved[:k]], gold


def hit_at_k(retrieved, gold_doc, gold_pages, k=3):
    """1 if at least one gold page is in the top-k, else 0 (best single retrieval number)."""
    flags, _ = _flags(retrieved, gold_doc, gold_pages, k)
    return float(any(flags))


def ndcg_at_k(retrieved, gold_doc, gold_pages, k=3):
    """Rank-aware: a gold page at rank 1 is worth more than at rank 3."""
    flags, gold = _flags(retrieved, gold_doc, gold_pages, k)
    dcg = sum(f / math.log2(i + 2) for i, f in enumerate(flags))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


# ---------- answer quality ----------
def token_recall(pred, ref):
    """Share of reference tokens found in the prediction. Fair to long extractive answers."""
    p, r = Counter(normalize(pred).split()), Counter(normalize(ref).split())
    return sum((p & r).values()) / max(sum(r.values()), 1)


class SemanticScorer:
    """Cosine similarity of MiniLM embeddings: rewards correct meaning even if wording differs."""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.enc = SentenceTransformer(model_name)

    def score(self, pred, ref):
        a, b = self.enc.encode([str(pred), str(ref)], normalize_embeddings=True)
        return float(np.dot(a, b))


def faithfulness(answer, passages):
    """Groundedness: share of the answer's content words that appear in the retrieved passages.
    Low value = the model may be inventing facts (hallucination proxy)."""
    ans = [w for w in normalize(answer).split() if w not in ENGLISH_STOP_WORDS]
    if not ans:
        return 0.0
    ctx = set(normalize(" ".join(p["text"] for p in passages)).split())
    return sum(w in ctx for w in ans) / len(ans)


# ---------- abstention ("not found") ----------
def abstention_prf(df):
    """Treat 'not found' as the positive class.
    Precision: of all refusals, how many were correct. Recall: of unanswerable questions, how many refused."""
    refused = df.found == 0
    unans = df.answerable == 0
    tp, fp, fn = (refused & unans).sum(), (refused & ~unans).sum(), (~refused & unans).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"Abstain precision": p, "Abstain recall": r, "Abstain F1": f}


# ---------- statistics ----------
def paired_bootstrap(base_scores, adv_scores, n=5000, seed=0):
    """95% CI and p-value for (advanced - baseline) on the same questions."""
    d = np.asarray(adv_scores, float) - np.asarray(base_scores, float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p)}


# ------------------------------------------------------------------
# HOW TO PLUG INTO evaluate.py
# 1) imports:
#    from .metrics_extra import hit_at_k, ndcg_at_k, token_recall, SemanticScorer, faithfulness, abstention_prf, paired_bootstrap
# 2) in main(), before evaluating:   sem = SemanticScorer()      (pass `sem` into evaluate_model)
# 3) in evaluate_model(), inside the `if t["answerable"] == 1:` block, add to row.update(...):
#        hit=hit_at_k(pairs, t["gold_doc"], gold_pages, 3),
#        ndcg=ndcg_at_k(pairs, t["gold_doc"], gold_pages, 3),
#        tok_recall=token_recall(res["answer"], t["reference_answer"]),
#        sem_sim=sem.score(res["answer"], t["reference_answer"]),
#        faith=faithfulness(res["answer"], res["passages"]) if res["found"] else 0.0
#    (where pairs = [(x["doc_name"], x["page_no"]) for x in passages])
#    and change  "top1_score": passages[0]["score"] ...   to   "top1_score": res["top_score"]
# 4) in summarize(), add:
#        "Hit@3": a["hit"].mean(), "nDCG@3": a["ndcg"].mean(), "Token Recall": a["tok_recall"].mean(),
#        "Semantic similarity": a["sem_sim"].mean(), "Faithfulness": a["faith"].mean()
#    and  s.update(abstention_prf(df))
# 5) significance, after both DataFrames exist:
#        a_b, a_a = base_df[base_df.answerable == 1], adv_df[adv_df.answerable == 1]
#        print(paired_bootstrap(a_b["sem_sim"], a_a["sem_sim"]))