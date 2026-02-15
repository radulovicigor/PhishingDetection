"""
Explainability module.
Provides global feature importance and local explanations for predictions.
"""
import sys
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from src.config import (
    MODEL_PATH,
    TOP_FEATURES_PHISHING_PATH,
    TOP_FEATURES_LEGIT_PATH,
    TOP_FEATURES_COUNT,
    LOCAL_EXPLANATION_TOP_N,
)
from src.train import load_model, get_tfidf_feature_names, get_heuristic_feature_names
from src.features import extract_all_heuristics, get_heuristic_feature_names as get_heur_names
from src.utils import setup_logger, ensure_directories, save_csv, print_separator

logger = setup_logger(__name__)


def get_classifier_coefficients(pipeline) -> np.ndarray:
    """
    Extract coefficients from the logistic regression classifier.
    
    Args:
        pipeline: Trained pipeline
        
    Returns:
        Coefficient array
    """
    classifier = pipeline.named_steps['classifier']
    return classifier.coef_.flatten()


def get_all_feature_names(pipeline) -> List[str]:
    """
    Get all feature names (TF-IDF + heuristics) from pipeline.
    
    Args:
        pipeline: Trained pipeline
        
    Returns:
        List of feature names
    """
    tfidf_names = list(get_tfidf_feature_names(pipeline))
    heuristic_names = get_heuristic_feature_names(pipeline)
    return tfidf_names + heuristic_names


def compute_global_feature_importance(pipeline) -> pd.DataFrame:
    """
    Compute global feature importance from model coefficients.
    
    Args:
        pipeline: Trained pipeline
        
    Returns:
        DataFrame with feature names and coefficients
    """
    coefficients = get_classifier_coefficients(pipeline)
    feature_names = get_all_feature_names(pipeline)
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients),
    })
    
    return df.sort_values('abs_coefficient', ascending=False)


def get_top_phishing_features(importance_df: pd.DataFrame, n: int = TOP_FEATURES_COUNT) -> pd.DataFrame:
    """
    Get top features that indicate phishing (positive coefficients).
    
    Args:
        importance_df: Feature importance DataFrame
        n: Number of features to return
        
    Returns:
        DataFrame with top phishing-indicating features
    """
    phishing_features = importance_df[importance_df['coefficient'] > 0].copy()
    phishing_features = phishing_features.sort_values('coefficient', ascending=False)
    return phishing_features.head(n)[['feature', 'coefficient']]


def get_top_legit_features(importance_df: pd.DataFrame, n: int = TOP_FEATURES_COUNT) -> pd.DataFrame:
    """
    Get top features that indicate legitimate emails (negative coefficients).
    
    Args:
        importance_df: Feature importance DataFrame
        n: Number of features to return
        
    Returns:
        DataFrame with top legitimate-indicating features
    """
    legit_features = importance_df[importance_df['coefficient'] < 0].copy()
    legit_features = legit_features.sort_values('coefficient', ascending=True)
    return legit_features.head(n)[['feature', 'coefficient']]


def explain_email(
    pipeline,
    text: str,
    urls: str,
    top_n: int = LOCAL_EXPLANATION_TOP_N
) -> Dict[str, Any]:
    """
    Generate local explanation for a single email.
    
    Args:
        pipeline: Trained pipeline
        text: Email text (subject + body)
        urls: URLs string
        top_n: Number of top contributing features to return
        
    Returns:
        Dictionary with prediction and explanation
    """
    # Prepare input
    input_df = pd.DataFrame({'text': [text], 'urls': [urls]})
    
    # Get prediction
    prediction = pipeline.predict(input_df)[0]
    label = "PHISHING" if prediction == 1 else "LEGIT"
    
    # Get probability
    proba = None
    if hasattr(pipeline, 'predict_proba'):
        proba = float(pipeline.predict_proba(input_df)[0, 1])
    
    # Get coefficients and feature names
    coefficients = get_classifier_coefficients(pipeline)
    feature_names = get_all_feature_names(pipeline)
    
    # Transform input to get feature values
    feature_transformer = pipeline.named_steps['features']
    X_transformed = feature_transformer.transform(input_df)
    
    if issparse(X_transformed):
        X_transformed = X_transformed.toarray()
    
    feature_values = X_transformed.flatten()
    
    # Compute contributions (coefficient * value)
    contributions = []
    for i, (name, coef, value) in enumerate(zip(feature_names, coefficients, feature_values)):
        if value != 0:  # Only non-zero features
            contribution = coef * value
            contributions.append({
                'feature': name,
                'value': float(value),
                'coefficient': float(coef),
                'contribution': float(contribution),
            })
    
    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    # Get top contributing features
    top_contributions = contributions[:top_n]
    
    # Generate human-readable reasons
    reasons = generate_reasons(top_contributions, text, urls)
    
    # Get heuristic feature values
    heuristics = extract_all_heuristics(text, urls)
    
    return {
        'label': label,
        'score': proba,
        'prediction': int(prediction),
        'top_contributing_features': top_contributions,
        'heuristics': heuristics,
        'reasons': reasons,
    }


def generate_reasons(contributions: List[Dict], text: str, urls: str) -> List[str]:
    """
    Generate human-readable reasons from feature contributions.
    
    Args:
        contributions: List of contribution dictionaries
        text: Email text
        urls: URLs string
        
    Returns:
        List of reason strings
    """
    reasons = []
    heuristic_names = set(get_heur_names())
    
    for contrib in contributions:
        feature = contrib['feature']
        value = contrib['value']
        contribution = contrib['contribution']
        
        # Direction
        direction = "phishing" if contribution > 0 else "legitimate"
        strength = abs(contribution)
        
        if feature in heuristic_names:
            # Heuristic feature
            if feature == 'url_count' and value > 0:
                reasons.append(f"Contains {int(value)} URL(s) - indicates {direction}")
            elif feature == 'suspicious_tld' and value > 0:
                reasons.append(f"Contains URL with suspicious TLD - indicates {direction}")
            elif feature == 'uses_shortener' and value > 0:
                reasons.append(f"Uses URL shortener service - indicates {direction}")
            elif feature == 'keyword_hits' and value > 0:
                reasons.append(f"Contains {int(value)} phishing keyword(s) - indicates {direction}")
            elif feature == 'caps_ratio' and value > 0.1:
                reasons.append(f"High proportion of CAPS ({value:.1%}) - indicates {direction}")
            elif feature == 'exclamation_count' and value > 2:
                reasons.append(f"Multiple exclamation marks ({int(value)}) - indicates {direction}")
        else:
            # TF-IDF feature (word/ngram)
            if strength > 0.1:  # Only significant contributions
                if contribution > 0:
                    reasons.append(f"Contains term '{feature}' - indicates {direction}")
                else:
                    reasons.append(f"Contains term '{feature}' - indicates {direction}")
    
    # Limit to top 5 reasons
    return reasons[:5]


def save_feature_importance_reports(pipeline) -> None:
    """
    Generate and save feature importance reports.
    
    Args:
        pipeline: Trained pipeline
    """
    ensure_directories()
    
    # Compute importance
    importance_df = compute_global_feature_importance(pipeline)
    
    # Get top phishing features
    phishing_df = get_top_phishing_features(importance_df)
    save_csv(phishing_df, TOP_FEATURES_PHISHING_PATH)
    logger.info(f"Top phishing features saved to {TOP_FEATURES_PHISHING_PATH}")
    
    # Get top legit features
    legit_df = get_top_legit_features(importance_df)
    save_csv(legit_df, TOP_FEATURES_LEGIT_PATH)
    logger.info(f"Top legitimate features saved to {TOP_FEATURES_LEGIT_PATH}")


def print_feature_importance_summary(pipeline) -> None:
    """Print a summary of feature importance."""
    importance_df = compute_global_feature_importance(pipeline)
    
    print_separator()
    print("TOP FEATURES INDICATING PHISHING")
    print_separator()
    phishing_df = get_top_phishing_features(importance_df, n=15)
    for _, row in phishing_df.iterrows():
        print(f"  {row['feature']:<40} {row['coefficient']:>10.4f}")
    
    print("\n")
    print_separator()
    print("TOP FEATURES INDICATING LEGITIMATE")
    print_separator()
    legit_df = get_top_legit_features(importance_df, n=15)
    for _, row in legit_df.iterrows():
        print(f"  {row['feature']:<40} {row['coefficient']:>10.4f}")


def main():
    """Main explainability entry point."""
    logger.info("=" * 60)
    logger.info("PHISH DETECTOR - EXPLAINABILITY")
    logger.info("=" * 60)
    
    try:
        # Load model
        logger.info("\n--- LOADING MODEL ---")
        pipeline = load_model(MODEL_PATH)
        
        # Generate and save reports
        logger.info("\n--- GENERATING REPORTS ---")
        save_feature_importance_reports(pipeline)
        
        # Print summary
        print("\n")
        print_feature_importance_summary(pipeline)
        
        # Demo local explanation
        print("\n")
        print_separator()
        print("DEMO: LOCAL EXPLANATION")
        print_separator()
        
        demo_text = """
        URGENT: Your account has been suspended!
        
        Dear Customer,
        
        We have detected suspicious activity on your account. 
        Please click the link below to verify your identity immediately 
        or your account will be permanently closed.
        
        Click here to verify: http://secure-bank.xyz/verify
        
        Thank you,
        Security Team
        """
        demo_urls = "http://secure-bank.xyz/verify"
        
        explanation = explain_email(pipeline, demo_text, demo_urls)
        
        print(f"\nPrediction: {explanation['label']}")
        print(f"Score: {explanation['score']:.4f}")
        print("\nReasons:")
        for reason in explanation['reasons']:
            print(f"  • {reason}")
        
        print("\nHeuristic Features:")
        for k, v in explanation['heuristics'].items():
            print(f"  {k}: {v}")
        
        logger.info("\nExplainability analysis complete!")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please run 'python -m src.train' first.")
        return 1
    except Exception as e:
        logger.error(f"Explainability analysis failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
