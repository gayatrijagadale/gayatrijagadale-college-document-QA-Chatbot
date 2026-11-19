"""Metric functions (Section 15). Plain Python so every formula is easy to explain."""
import re
import string
from collections import Counter


def normalize(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred, ref):
    """EM = 1 if normalised prediction == normalised reference, else 0."""
    return float(normalize(pred) == normalize(ref))


def token_f1(pred, ref):
    """F1 = 2PR/(P+R), P = overlap/len(pred tokens), R = overlap/len(ref tokens)."""
    p, r = normalize(pred).split(), normalize(ref).split()
    common = Counter(p) & Counter(r)
    overlap = sum(common.values())
    if not p or not r or overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def _lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(pred, ref):
    """ROUGE-L F = 2PR/(P+R) with P = LCS/len(pred), R = LCS/len(ref)."""
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return 0.0
    lcs = _lcs(p, r)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


def retrieval_metrics(retrieved, gold_doc, gold_pages, k=3):
    """retrieved: list of (doc_name, page_no) in rank order.
    Precision@k = relevant retrieved / k
    Recall@k    = distinct gold pages found in top-k / number of gold pages
    RR          = 1 / rank of first relevant passage (0 if none); MRR = mean of RR."""
    gold = {(gold_doc.lower(), int(pg)) for pg in gold_pages}
    top = [(d.lower(), int(p)) for d, p in retrieved[:k]]
    rel_flags = [t in gold for t in top]
    precision = sum(rel_flags) / k
    recall = len(gold & set(top)) / len(gold) if gold else 0.0
    rr = next((1.0 / (i + 1) for i, f in enumerate(rel_flags) if f), 0.0)
    return precision, recall, rr
