"""
Error analysis module.
Analiza grešaka modela: karakteristike lažno pozitivnih (FP) i lažno negativnih (FN) primjera.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    MODEL_PATH,
    REPORTS_DIR,
    ERROR_ANALYSIS_PATH,
    ERROR_ANALYSIS_SUMMARY_PATH,
)
from src.preprocessing import run_preprocessing_pipeline
from src.train import load_model
from src.features import extract_all_heuristics
from src.utils import setup_logger, ensure_directories, save_json

logger = setup_logger(__name__)

HEURISTIC_KEYS = [
    "url_count", "has_url", "exclamation_count", "question_count",
    "text_len", "word_count", "caps_ratio", "keyword_hits",
    "suspicious_tld", "uses_shortener",
]


def _avg_heuristics(df: pd.DataFrame) -> dict:
    """Compute mean of each heuristic over rows. urls column must be present."""
    if len(df) == 0:
        return {k: None for k in HEURISTIC_KEYS}
    rows = [
        extract_all_heuristics(row["text"], str(row.get("urls", "")))
        for _, row in df.iterrows()
    ]
    out = {}
    for k in HEURISTIC_KEYS:
        vals = [r[k] for r in rows]
        out[k] = float(np.mean(vals))
    return out


def run_error_analysis() -> dict:
    """
    Run error analysis: characterize FP (legit predicted as phishing) and FN (phishing predicted as legit).

    Returns:
        Dict with counts and average heuristics for TP, TN, FP, FN.
    """
    logger.info("Loading model and test data...")
    pipeline = load_model(MODEL_PATH)
    _, test_df = run_preprocessing_pipeline()

    X = test_df[["text", "urls"]].copy()
    y_true = test_df["label"].values
    y_pred = pipeline.predict(X)

    # Ensure we have urls column in test_df for heuristics
    if "urls" not in test_df.columns:
        test_df["urls"] = ""

    test_df = test_df.copy()
    test_df["y_true"] = y_true
    test_df["y_pred"] = y_pred

    tp_df = test_df[(y_true == 1) & (y_pred == 1)]
    tn_df = test_df[(y_true == 0) & (y_pred == 0)]
    fp_df = test_df[(y_true == 0) & (y_pred == 1)]  # legit → predicted phishing
    fn_df = test_df[(y_true == 1) & (y_pred == 0)]  # phishing → predicted legit

    out = {
        "counts": {
            "true_positives": int(len(tp_df)),
            "true_negatives": int(len(tn_df)),
            "false_positives": int(len(fp_df)),
            "false_negatives": int(len(fn_df)),
            "total_test": int(len(test_df)),
        },
        "false_positives_avg_heuristics": _avg_heuristics(fp_df),
        "false_negatives_avg_heuristics": _avg_heuristics(fn_df),
        "true_negatives_avg_heuristics": _avg_heuristics(tn_df),
        "true_positives_avg_heuristics": _avg_heuristics(tp_df),
    }
    return out


def write_summary_txt(results: dict, path: Path) -> None:
    """Write a short human-readable summary for the paper."""
    c = results["counts"]
    fp_avg = results["false_positives_avg_heuristics"]
    fn_avg = results["false_negatives_avg_heuristics"]
    tn_avg = results["true_negatives_avg_heuristics"]
    tp_avg = results["true_positives_avg_heuristics"]

    lines = [
        "ERROR ANALYSIS SUMMARY",
        "======================",
        "",
        f"Test set size: {c['total_test']}",
        f"  True positives:  {c['true_positives']}",
        f"  True negatives:  {c['true_negatives']}",
        f"  False positives: {c['false_positives']} (legitimate emails predicted as phishing)",
        f"  False negatives: {c['false_negatives']} (phishing emails predicted as legitimate)",
        "",
        "--- False Positives (legit → predicted phishing) ---",
        "Average heuristics (why the model may have been misled):",
        f"  keyword_hits:    {fp_avg.get('keyword_hits')}",
        f"  url_count:       {fp_avg.get('url_count')}",
        f"  suspicious_tld:  {fp_avg.get('suspicious_tld')}",
        f"  word_count:      {fp_avg.get('word_count')}",
        f"  text_len:        {fp_avg.get('text_len')}",
        "",
        "--- False Negatives (phishing → predicted legit) ---",
        "Average heuristics (model missed these signals):",
        f"  keyword_hits:    {fn_avg.get('keyword_hits')}",
        f"  url_count:       {fn_avg.get('url_count')}",
        f"  suspicious_tld:  {fn_avg.get('suspicious_tld')}",
        f"  word_count:      {fn_avg.get('word_count')}",
        "",
        "--- Comparison: TN vs FP (legit emails) ---",
        f"  TN avg keyword_hits:   {tn_avg.get('keyword_hits')}",
        f"  FP avg keyword_hits:   {fp_avg.get('keyword_hits')}",
        "",
        "--- Comparison: TP vs FN (phishing emails) ---",
        f"  TP avg keyword_hits:   {tp_avg.get('keyword_hits')}",
        f"  FN avg keyword_hits:   {fn_avg.get('keyword_hits')}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run error analysis and save results."""
    logger.info("=" * 60)
    logger.info("PHISH DETECTOR - ERROR ANALYSIS")
    logger.info("=" * 60)

    ensure_directories()
    try:
        results = run_error_analysis()
        save_json(results, ERROR_ANALYSIS_PATH)
        logger.info(f"Results saved to {ERROR_ANALYSIS_PATH}")

        write_summary_txt(results, ERROR_ANALYSIS_SUMMARY_PATH)
        logger.info(f"Summary saved to {ERROR_ANALYSIS_SUMMARY_PATH}")

        c = results["counts"]
        print("\n--- Error analysis ---")
        print(f"  False positives: {c['false_positives']}")
        print(f"  False negatives: {c['false_negatives']}")
        print()
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error analysis failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
