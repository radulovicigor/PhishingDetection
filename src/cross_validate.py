"""
Cross-validation module.
Statistička validacija modela: k-fold CV, rezultati kao mean ± std za pouzdanost procjene.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold

from src.config import (
    RAW_DATA_PATH,
    REPORTS_DIR,
    RANDOM_STATE,
    CROSS_VALIDATION_RESULTS_PATH,
    CV_FOLDS,
)
from src.preprocessing import load_raw_data, validate_columns, preprocess_dataframe
from src.train import create_pipeline
from src.utils import setup_logger, ensure_directories, save_json

logger = setup_logger(__name__)

SCORING = ["accuracy", "precision", "recall", "f1"]


def run_cross_validation(data_path: Path = None, n_folds: int = None) -> dict:
    """
    Run k-fold cross-validation and return mean ± std for each metric.

    Args:
        data_path: Path to CSV. If None, uses RAW_DATA_PATH.
        n_folds: Number of folds. If None, uses CV_FOLDS from config.

    Returns:
        Dict with keys like 'accuracy_mean', 'accuracy_std', etc.
    """
    data_path = data_path or RAW_DATA_PATH
    n_folds = n_folds or CV_FOLDS

    logger.info(f"Loading data from {data_path}")
    df = load_raw_data(data_path)
    validate_columns(df)
    df = preprocess_dataframe(df)

    X = df[["text", "urls"]].copy()
    y = df["label"].values

    logger.info(f"Running {n_folds}-fold cross-validation on {len(X)} samples...")
    pipeline = create_pipeline()

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
        return_train_score=False,
    )

    out = {
        "n_folds": n_folds,
        "n_samples": len(X),
        "random_state": RANDOM_STATE,
    }
    for name in SCORING:
        key = f"test_{name}"
        if key in cv_results:
            scores = cv_results[key]
            out[f"{name}_mean"] = float(np.mean(scores))
            out[f"{name}_std"] = float(np.std(scores))
            out[f"{name}_per_fold"] = [float(s) for s in scores]

    return out


def main():
    """Run cross-validation and save results to reports/."""
    logger.info("=" * 60)
    logger.info("PHISH DETECTOR - CROSS-VALIDATION (statistička validacija)")
    logger.info("=" * 60)

    ensure_directories()
    try:
        results = run_cross_validation()
        save_json(results, CROSS_VALIDATION_RESULTS_PATH)
        logger.info(f"Results saved to {CROSS_VALIDATION_RESULTS_PATH}")

        print("\n--- Cross-Validation Results (mean ± std) ---")
        for name in SCORING:
            m, s = results.get(f"{name}_mean"), results.get(f"{name}_std")
            if m is not None and s is not None:
                print(f"  {name}: {m:.4f} ± {s:.4f}")
        print()
        return 0
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
