import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sentence_transformers import util
from app import sbert_model, calculate_doc2vec_similarities


def score_pairs(df):
    """Scores every JD/resume pair using the exact same logic as app.py's /predict route."""
    results = []
    for jd_text, group in df.groupby("jd_text"):
        resumes = group["resume_text"].tolist()
        labels = group["label"].tolist()

        jd_embedding = sbert_model.encode(jd_text, convert_to_tensor=True)
        resume_embeddings = sbert_model.encode(resumes, convert_to_tensor=True)
        cosine_results = util.cos_sim(jd_embedding, resume_embeddings)[0]
        sbert_scores = [max(0.0, min(100.0, float(s.item() * 100))) for s in cosine_results]

        doc2vec_scores = calculate_doc2vec_similarities(jd_text, resumes)

        for i in range(len(resumes)):
            avg_score = (sbert_scores[i] + doc2vec_scores[i]) / 2
            results.append({
                "jd_text": jd_text,
                "avg_score": avg_score,
                "label": labels[i]
            })
    return pd.DataFrame(results)


def evaluate_thresholds(scored_df, thresholds=(40, 50, 60, 65, 70, 80, 90)):
    rows = []
    for t in thresholds:
        y_pred = (scored_df["avg_score"] >= t).astype(int)
        y_true = scored_df["label"]
        rows.append({
            "threshold": t,
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 2),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 2),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 2),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Loading ground truth data...")
    df = pd.read_csv("eval/ground_truth.csv")

    print(f"Scoring {len(df)} JD/resume pairs using SBERT + Doc2Vec...")
    scored = score_pairs(df)
    scored.to_csv("eval/scored_pairs.csv", index=False)

    print("\nEvaluating thresholds...\n")
    results = evaluate_thresholds(scored)
    print(results.to_string(index=False))

    results.to_csv("eval/threshold_results.csv", index=False)
    print("\nSaved results to eval/threshold_results.csv")

    best = results.loc[results["f1"].idxmax()]
    print(f"\nBest threshold by F1 score: {int(best['threshold'])} "
          f"(Precision={best['precision']}, Recall={best['recall']}, F1={best['f1']})")