"""
Evaluation module.
Computes metrics, confusion matrix, and threshold analysis.
"""
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

from src.config import (
    MODEL_PATH,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    THRESHOLD_METRICS_PATH,
    EVALUATION_THRESHOLDS,
)
from src.preprocessing import run_preprocessing_pipeline
from src.train import load_model
from src.utils import (
    setup_logger,
    ensure_directories,
    save_json,
    save_csv,
    print_metrics_summary,
    print_separator,
)

logger = setup_logger(__name__)


def compute_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Compute confusion matrix and derived metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with TP, FP, TN, FN
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # For binary classification: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
    }


def compute_false_positive_rate(fp: int, tn: int) -> float:
    """
    Compute False Positive Rate.
    
    FPR = FP / (FP + TN)
    
    Args:
        fp: False positives count
        tn: True negatives count
        
    Returns:
        False positive rate
    """
    if fp + tn == 0:
        return 0.0
    return fp / (fp + tn)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional, for ROC-AUC)
        
    Returns:
        Dictionary of all metrics
    """
    # Basic metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    # Confusion matrix metrics
    cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)
    metrics['confusion_matrix'] = cm_metrics
    
    # False Positive Rate
    fpr = compute_false_positive_rate(
        cm_metrics['false_positives'],
        cm_metrics['true_negatives']
    )
    metrics['false_positive_rate'] = float(fpr)
    
    # ROC-AUC (if probabilities available)
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            metrics['roc_auc'] = float(roc_auc)
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    
    # Sample counts
    metrics['total_samples'] = len(y_true)
    metrics['positive_samples'] = int(y_true.sum())
    metrics['negative_samples'] = int(len(y_true) - y_true.sum())
    
    return metrics


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Compute metrics at a specific probability threshold.
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics at threshold
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred)
    fpr = compute_false_positive_rate(
        cm_metrics['false_positives'],
        cm_metrics['true_negatives']
    )
    
    return {
        'threshold': threshold,
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'false_positive_rate': float(fpr),
        'true_positives': cm_metrics['true_positives'],
        'false_positives': cm_metrics['false_positives'],
        'true_negatives': cm_metrics['true_negatives'],
        'false_negatives': cm_metrics['false_negatives'],
    }


def threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: List[float] = EVALUATION_THRESHOLDS
) -> pd.DataFrame:
    """
    Perform threshold analysis.
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        thresholds: List of thresholds to evaluate
        
    Returns:
        DataFrame with metrics at each threshold
    """
    results = []
    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_proba, threshold)
        results.append(metrics)
    
    return pd.DataFrame(results)


def save_confusion_matrix(cm_metrics: Dict[str, int], path=CONFUSION_MATRIX_PATH) -> None:
    """
    Save confusion matrix as CSV.
    
    Args:
        cm_metrics: Dictionary with TP, FP, TN, FN
        path: Output path
    """
    # Create 2x2 matrix format
    df = pd.DataFrame([
        [cm_metrics['true_negatives'], cm_metrics['false_positives']],
        [cm_metrics['false_negatives'], cm_metrics['true_positives']],
    ], index=['Actual Negative', 'Actual Positive'],
       columns=['Predicted Negative', 'Predicted Positive'])
    
    save_csv(df, path, index=True)
    logger.info(f"Confusion matrix saved to {path}")


def print_threshold_analysis(df: pd.DataFrame) -> None:
    """Print threshold analysis results."""
    print_separator()
    print("THRESHOLD ANALYSIS")
    print_separator()
    
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10}")
    print("-" * 52)
    
    for _, row in df.iterrows():
        print(f"{row['threshold']:>10.2f} "
              f"{row['precision']:>10.2%} "
              f"{row['recall']:>10.2%} "
              f"{row['f1_score']:>10.2%} "
              f"{row['false_positive_rate']:>10.2%}")


def evaluate_model(pipeline, test_df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Run full evaluation on test set.
    
    Args:
        pipeline: Trained model pipeline
        test_df: Test DataFrame
        
    Returns:
        Tuple of (metrics dict, threshold analysis DataFrame)
    """
    logger.info("Running evaluation...")
    
    # Prepare test data
    X_test = test_df[['text', 'urls']].copy()
    y_true = test_df['label'].values
    
    # Get predictions
    y_pred = pipeline.predict(X_test)
    
    # Get probabilities if available
    y_proba = None
    if hasattr(pipeline, 'predict_proba'):
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception as e:
            logger.warning(f"Could not get probabilities: {e}")
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    
    # Threshold analysis
    threshold_df = None
    if y_proba is not None:
        threshold_df = threshold_analysis(y_true, y_proba)
    
    return metrics, threshold_df


def main():
    """Main evaluation entry point."""
    logger.info("=" * 60)
    logger.info("PHISH DETECTOR - EVALUATION")
    logger.info("=" * 60)
    
    ensure_directories()
    
    try:
        # Load model
        logger.info("\n--- LOADING MODEL ---")
        pipeline = load_model(MODEL_PATH)
        
        # Load and preprocess data
        logger.info("\n--- LOADING DATA ---")
        _, test_df = run_preprocessing_pipeline()
        
        # Run evaluation
        logger.info("\n--- EVALUATING ---")
        metrics, threshold_df = evaluate_model(pipeline, test_df)
        
        # Save results
        logger.info("\n--- SAVING RESULTS ---")
        save_json(metrics, METRICS_PATH)
        logger.info(f"Metrics saved to {METRICS_PATH}")
        
        save_confusion_matrix(metrics['confusion_matrix'], CONFUSION_MATRIX_PATH)
        
        if threshold_df is not None:
            save_csv(threshold_df, THRESHOLD_METRICS_PATH)
            logger.info(f"Threshold analysis saved to {THRESHOLD_METRICS_PATH}")
        
        # Print summary
        print("\n")
        print_metrics_summary(metrics)
        
        if threshold_df is not None:
            print("\n")
            print_threshold_analysis(threshold_df)
        
        # Highlight FPR
        print("\n")
        print_separator("*")
        fpr = metrics['false_positive_rate']
        print(f"  FALSE POSITIVE RATE (FPR): {fpr:.2%}")
        print(f"  This means {fpr:.2%} of legitimate emails are incorrectly flagged as phishing.")
        print_separator("*")
        
        logger.info("\nEvaluation complete!")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please run 'python -m src.train' first.")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
