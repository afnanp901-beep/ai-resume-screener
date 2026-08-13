import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from eval.run_eval import score_pairs


def precision_at_k(ranked_labels, k):
    top_k = ranked_labels[:k]
    if len(top_k) == 0:
        return 0.0
    return sum(top_k) / len(top_k)


def recall_at_k(ranked_labels, k, total_relevant):
    if total_relevant == 0:
        return 0.0
    top_k = ranked_labels[:k]
    return sum(top_k) / total_relevant


def evaluate_ranking(scored_df, k=3):
    p_scores, r_scores = [], []
    for jd_text, group in scored_df.groupby("jd_text"):
        ranked = group.sort_values("avg_score", ascending=False)
        labels = ranked["label"].tolist()
        total_relevant = sum(labels)

        p_scores.append(precision_at_k(labels, k))
        r_scores.append(recall_at_k(labels, k, total_relevant))

    return {
        f"precision@{k}": round(float(np.mean(p_scores)), 2),
        f"recall@{k}": round(float(np.mean(r_scores)), 2),
    }


if __name__ == "__main__":
    df = pd.read_csv("eval/ground_truth.csv")
    scored = score_pairs(df)

    for k in (1, 3, 5):
        result = evaluate_ranking(scored, k=k)
        print(result)