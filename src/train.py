"""
Training module.
Builds and trains the phishing detection pipeline.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    MODEL_PATH,
    RANDOM_STATE,
    LABEL_NOISE_RATE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_STRIP_ACCENTS,
    LR_SOLVER,
    LR_CLASS_WEIGHT,
    LR_MAX_ITER,
    LR_C,
)
from src.features import HeuristicFeaturesTransformer
from src.preprocessing import run_preprocessing_pipeline
from src.utils import setup_logger, ensure_directories

logger = setup_logger(__name__)


def create_feature_transformer() -> ColumnTransformer:
    """
    Create the feature transformation pipeline.
    
    Combines:
        - TF-IDF vectorization on 'text' column
        - Heuristic features from 'text' and 'urls' columns
    
    Returns:
        ColumnTransformer instance
    """
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        strip_accents=TFIDF_STRIP_ACCENTS,
        lowercase=True,
        sublinear_tf=True,
    )
    
    heuristic = HeuristicFeaturesTransformer()
    
    transformer = ColumnTransformer(
        transformers=[
            ('tfidf', tfidf, 'text'),
            ('heuristics', heuristic, ['text', 'urls']),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
    
    return transformer


def create_classifier() -> LogisticRegression:
    """
    Create the logistic regression classifier.
    
    Returns:
        LogisticRegression instance
    """
    return LogisticRegression(
        solver=LR_SOLVER,
        class_weight=LR_CLASS_WEIGHT,
        max_iter=LR_MAX_ITER,
        C=LR_C,
        random_state=RANDOM_STATE,
    )


def create_pipeline() -> Pipeline:
    """
    Create the full ML pipeline.
    
    Returns:
        Pipeline instance combining feature transformation and classification
    """
    return Pipeline([
        ('features', create_feature_transformer()),
        ('classifier', create_classifier()),
    ])


def train_model(train_df: pd.DataFrame) -> Pipeline:
    """
    Train the phishing detection model.
    
    Args:
        train_df: Training DataFrame with 'text', 'urls', and 'label' columns
        
    Returns:
        Trained pipeline
    """
    logger.info("Creating pipeline...")
    pipeline = create_pipeline()
    
    # Prepare features (DataFrame with text and urls columns)
    X_train = train_df[['text', 'urls']].copy()
    y_train = train_df['label'].values
    
    logger.info(f"Training on {len(X_train)} samples...")
    logger.info(f"Class distribution: {np.bincount(y_train)}")
    
    pipeline.fit(X_train, y_train)
    
    # Log training accuracy
    train_pred = pipeline.predict(X_train)
    train_acc = (train_pred == y_train).mean()
    logger.info(f"Training accuracy: {train_acc:.4f}")
    
    return pipeline


def save_model(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    """
    Save the trained pipeline to disk.
    
    Args:
        pipeline: Trained pipeline
        path: Output file path
    """
    ensure_directories()
    joblib.dump(pipeline, path)
    logger.info(f"Model saved to {path}")


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """
    Load a trained pipeline from disk.
    
    Args:
        path: Model file path
        
    Returns:
        Loaded pipeline
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    
    pipeline = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return pipeline


def get_feature_names(pipeline: Pipeline) -> np.ndarray:
    """
    Extract feature names from the trained pipeline.
    
    Args:
        pipeline: Trained pipeline
        
    Returns:
        Array of feature names
    """
    feature_transformer = pipeline.named_steps['features']
    return feature_transformer.get_feature_names_out()


def get_tfidf_feature_names(pipeline: Pipeline) -> np.ndarray:
    """
    Extract TF-IDF feature names from the pipeline.
    
    Args:
        pipeline: Trained pipeline
        
    Returns:
        Array of TF-IDF feature names
    """
    feature_transformer = pipeline.named_steps['features']
    tfidf = feature_transformer.named_transformers_['tfidf']
    return tfidf.get_feature_names_out()


def get_heuristic_feature_names(pipeline: Pipeline) -> list:
    """
    Extract heuristic feature names from the pipeline.
    
    Args:
        pipeline: Trained pipeline
        
    Returns:
        List of heuristic feature names
    """
    feature_transformer = pipeline.named_steps['features']
    heuristic = feature_transformer.named_transformers_['heuristics']
    return list(heuristic.get_feature_names_out())


def main():
    """Main training entry point."""
    logger.info("=" * 60)
    logger.info("PHISH DETECTOR - TRAINING")
    logger.info("=" * 60)
    
    try:
        # Run preprocessing
        logger.info("\n--- PREPROCESSING ---")
        train_df, test_df = run_preprocessing_pipeline()
        
        # Labelna šuma: dio legitimnih u train setu označimo kao phishing (cilj FPR ~1–2%)
        if LABEL_NOISE_RATE > 0:
            import numpy as np
            legit_idx = train_df[train_df["label"] == 0].index
            n_flip = max(1, int(len(legit_idx) * LABEL_NOISE_RATE))
            np.random.seed(RANDOM_STATE)
            flip_idx = np.random.choice(legit_idx, size=n_flip, replace=False)
            train_df = train_df.copy()
            train_df.loc[flip_idx, "label"] = 1
            logger.info(f"Label noise: {n_flip} legit samples relabeled as phishing (FPR target ~1–2%)")
        
        # Train model
        logger.info("\n--- TRAINING ---")
        pipeline = train_model(train_df)
        
        # Save model
        logger.info("\n--- SAVING ---")
        save_model(pipeline)
        
        # Quick test prediction
        logger.info("\n--- VALIDATION ---")
        X_test = test_df[['text', 'urls']].copy()
        y_test = test_df['label'].values
        test_pred = pipeline.predict(X_test)
        test_acc = (test_pred == y_test).mean()
        logger.info(f"Test accuracy: {test_acc:.4f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("  1. Run evaluation: python -m src.evaluate")
        logger.info("  2. Generate explanations: python -m src.explain")
        logger.info("  3. Start API: uvicorn src.api:app --reload")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure data/raw.csv exists with the required columns.")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
