"""Sections 14-17: evaluate both models on eval/test_set.csv.   Run:  python -m src.evaluate

test_set.csv columns: question, reference_answer, gold_doc, gold_page, answerable
  gold_page may list several pages as  3;4   |  answerable = 1 or 0 (0 = answer is NOT in the documents)
All numbers below are computed from real predictions. Nothing is hard-coded."""
import time
import os
import numpy as np
import pandas as pd
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PATHS, CONFIG
from .metrics import exact_match, token_f1, rouge_l, retrieval_metrics
from .system import indexes_exist, build_all
from .retrievers import TfidfRetriever, FaissRetriever
from .qa_models import BaselineQA, AdvancedQA


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def load_test_set():
    if not PATHS["test_set"].exists():
        raise FileNotFoundError(f"Create {PATHS['test_set']} (see README, section 'Test set').")
    df = pd.read_csv(PATHS["test_set"], dtype=str).fillna("")
    need = {"question", "reference_answer", "gold_doc", "gold_page", "answerable"}
    if not need.issubset(df.columns):
        raise ValueError(f"test_set.csv must have columns: {sorted(need)}")
    df["answerable"] = df["answerable"].astype(float).astype(int)
    return df


def parse_pages(v):
    pages = []
    for x in str(v).replace(",", ";").split(";"):
        try:
            pages.append(int(float(x.strip())))
        except ValueError:
            pass
    return pages


def evaluate_model(model, test):
    rows = []
    for _, t in test.iterrows():
        res = model.answer(t["question"])
        passages = res["passages"]  # raw top-k, before any threshold decision
        row = {"question": t["question"], "answerable": int(t["answerable"]),
               "answer": res["answer"], "found": int(res["found"]), "time": res["time"],
               "top1_score": passages[0]["score"] if passages else 0.0,
               "citations": " ".join(res["citations"])}
        if t["answerable"] == 1:
            gold_pages = parse_pages(t["gold_page"])
            p, r, rr = retrieval_metrics([(x["doc_name"], x["page_no"]) for x in passages],
                                         t["gold_doc"], gold_pages, CONFIG["top_k"])
            row.update(precision=p, recall=r, rr=rr,
                       em=exact_match(res["answer"], t["reference_answer"]),
                       f1=token_f1(res["answer"], t["reference_answer"]),
                       rougeL=rouge_l(res["answer"], t["reference_answer"]))
            cited = {c.strip("[]") for c in res["citations"]}
            row["citation_ok"] = int(any(f"{t['gold_doc']}, PDF page {pg}".lower() in c.lower()
                                         for c in cited for pg in gold_pages))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df):
    a, u = df[df.answerable == 1], df[df.answerable == 0]
    s = {"Precision@3": a["precision"].mean(), "Recall@3": a["recall"].mean(), "MRR": a["rr"].mean(),
         "Exact Match": a["em"].mean(), "Token F1": a["f1"].mean(), "ROUGE-L": a["rougeL"].mean(),
         "Citation accuracy": a["citation_ok"].mean(),
         "Avg response time (s)": df["time"].mean()}
    if len(u):
        s["Correct 'not found' on unanswerable"] = 1 - u["found"].mean()
    s["False 'not found' on answerable"] = 1 - a["found"].mean()
    return s


def tune_threshold(df, grid):
    """Pick a threshold on top-1 retrieval score that best separates answerable/unanswerable."""
    if df.answerable.nunique() < 2:
        return None
    out = []
    for t in grid:
        pred = (df.top1_score >= t).astype(int)
        out.append({"threshold": round(float(t), 3), "accuracy": float((pred == df.answerable).mean())})
    return pd.DataFrame(out)


def human_scores(sheet_path, test, base_df, adv_df):
    """Create the sheet once; if you have filled in 0/1/2 scores, return their means."""
    if not sheet_path.exists():
        sheet = pd.DataFrame({"question": test["question"], "reference_answer": test["reference_answer"],
                              "baseline_answer": base_df["answer"], "advanced_answer": adv_df["answer"]})
        for c in ["baseline_correct", "baseline_relevant", "advanced_correct", "advanced_relevant"]:
            sheet[c] = ""
        sheet.to_csv(sheet_path, index=False)
        print(f"Human evaluation sheet created: {sheet_path}\n"
              "  Fill the 4 score columns (0=wrong/irrelevant, 1=partly, 2=fully) and re-run to include them.")
        return {}
    sh = pd.read_csv(sheet_path)
    res = {}
    for m in ("baseline", "advanced"):
        for k in ("correct", "relevant"):
            col = pd.to_numeric(sh[f"{m}_{k}"], errors="coerce")
            if col.notna().sum() > 0:
                res[(m, k)] = col.mean() / 2.0  # scaled to 0-1
    return res


def main():
    test = load_test_set()
    print(f"Test questions: {len(test)} ({int(test.answerable.sum())} answerable, "
          f"{int((1 - test.answerable).sum())} unanswerable)")
    if not indexes_exist():
        build_all()
    chunks = pd.read_csv(PATHS["chunks"])

    # ---- load models and measure memory (RSS increase, approximate) ----
    m0 = rss_mb()
    baseline = BaselineQA(TfidfRetriever.load(chunks))
    m1 = rss_mb()
    advanced = AdvancedQA(FaissRetriever.load(chunks))
    m2 = rss_mb()
    mem = {"base": max(m1 - m0, 0), "adv": max(m2 - m1, 0)}
    advanced.answer("warm-up question")  # exclude one-off start-up cost from timing

    print("Evaluating baseline ..."); base_df = evaluate_model(baseline, test)
    print("Evaluating advanced ..."); adv_df = evaluate_model(advanced, test)
    base_df.to_csv(PATHS["eval_dir"] / "results_baseline.csv", index=False)
    adv_df.to_csv(PATHS["eval_dir"] / "results_advanced.csv", index=False)

    sb, sa = summarize(base_df), summarize(adv_df)
    sb["Memory added by model (MB)"], sa["Memory added by model (MB)"] = mem["base"], mem["adv"]
    hs = human_scores(PATHS["eval_dir"] / "human_eval_sheet.csv", test, base_df, adv_df)
    for (m, k), v in hs.items():
        (sb if m == "baseline" else sa)[f"Human {k} (0-1)"] = v

    lower_better = {"Avg response time (s)", "Memory added by model (MB)", "False 'not found' on answerable"}
    rows = []
    for metric in sb:
        b, a = sb[metric], sa.get(metric, np.nan)
        imp = (b - a) if metric in lower_better else (a - b)  # positive = advanced is better
        rel = (imp / abs(b) * 100) if b not in (0, np.nan) and b == b and b != 0 else np.nan
        rows.append({"Metric": metric, "Baseline": round(b, 4), "Advanced": round(a, 4),
                     "Improvement (+ = advanced better)": round(imp, 4),
                     "Relative improvement %": round(rel, 1) if rel == rel else "n/a"})
    comp = pd.DataFrame(rows)
    comp.to_csv(PATHS["eval_dir"] / "comparison_table.csv", index=False)
    print("\n=== COMPARISON TABLE ===\n" + comp.to_string(index=False))

    # ---- threshold tuning suggestion ----
    for name, df, grid, key in (("Advanced", adv_df, np.arange(0.10, 0.65, 0.05), "sim_threshold"),
                                ("Baseline", base_df, np.arange(0.01, 0.31, 0.02), "tfidf_threshold")):
        t = tune_threshold(df, grid)
        if t is not None:
            best = t.loc[t.accuracy.idxmax()]
            t.to_csv(PATHS["eval_dir"] / f"threshold_sweep_{name.lower()}.csv", index=False)
            print(f"\n{name}: best '{key}' on this test set = {best.threshold} (accuracy {best.accuracy:.2f}); "
                  f"current = {CONFIG[key]}. Edit src/config.py if you want to use it.")

    # ---- graphs ----
    def bar(metrics, title, fname, ylabel):
        x = np.arange(len(metrics)); w = 0.38
        plt.figure(figsize=(8, 4.5))
        plt.bar(x - w / 2, [sb[m] for m in metrics], w, label="Baseline (TF-IDF)")
        plt.bar(x + w / 2, [sa[m] for m in metrics], w, label="Advanced (RAG)")
        plt.xticks(x, metrics, rotation=15, ha="right"); plt.ylabel(ylabel); plt.title(title)
        plt.legend(); plt.tight_layout()
        plt.savefig(PATHS["figures"] / fname, dpi=150); plt.close()

    bar(["Precision@3", "Recall@3", "MRR"], "Retrieval metrics", "retrieval_metrics.png", "Score")
    bar(["Exact Match", "Token F1", "ROUGE-L", "Citation accuracy"], "Answer quality", "answer_metrics.png", "Score")
    bar(["Avg response time (s)"], "Average response time", "response_time.png", "Seconds")
    print(f"\nGraphs saved in {PATHS['figures']}")


if __name__ == "__main__":
    main()
