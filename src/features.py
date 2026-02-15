"""
Feature engineering module.
Contains heuristic feature extractors and scikit-learn compatible transformers.
"""
import re
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    PHISHING_KEYWORDS,
    SUSPICIOUS_TLDS,
    URL_SHORTENERS,
)

# =============================================================================
# URL EXTRACTION UTILITIES
# =============================================================================

URL_REGEX = re.compile(
    r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*|'
    r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*',
    re.IGNORECASE
)


def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract URLs from text using regex.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted URLs
    """
    if not text:
        return []
    return URL_REGEX.findall(text)


def parse_urls_column(urls_str: str) -> List[str]:
    """
    Parse URLs from the urls column (may be comma-separated or space-separated).
    
    Args:
        urls_str: String containing URLs
        
    Returns:
        List of parsed URLs
    """
    if not urls_str or pd.isna(urls_str):
        return []
    
    # Try common separators
    urls = []
    for sep in [',', ';', ' ', '\n']:
        if sep in str(urls_str):
            urls = [u.strip() for u in str(urls_str).split(sep) if u.strip()]
            break
    
    if not urls:
        urls = [str(urls_str).strip()] if str(urls_str).strip() else []
    
    return [u for u in urls if u]


def get_all_urls(text: str, urls_column: str) -> List[str]:
    """
    Get all URLs from both text content and urls column.
    
    Args:
        text: Email text content
        urls_column: Content of urls column (may be binary 0/1 or actual URLs)
        
    Returns:
        Combined list of unique URLs
    """
    urls = set()
    
    # From urls column - check if it's actual URLs or just a binary indicator
    urls_str = str(urls_column).strip()
    if urls_str not in ('0', '1', '', 'nan', 'None'):
        # Might be actual URLs
        urls.update(parse_urls_column(urls_column))
    
    # From text via regex (primary source)
    urls.update(extract_urls_from_text(text))
    
    return list(urls)


def extract_domain(url: str) -> Optional[str]:
    """
    Extract domain from URL.
    
    Args:
        url: URL string
        
    Returns:
        Domain string or None
    """
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return None


def extract_tld(domain: str) -> Optional[str]:
    """
    Extract TLD from domain.
    
    Args:
        domain: Domain string
        
    Returns:
        TLD string or None
    """
    if not domain:
        return None
    parts = domain.split('.')
    if len(parts) >= 2:
        return parts[-1].lower()
    return None


# =============================================================================
# HEURISTIC FEATURE FUNCTIONS
# =============================================================================

def count_urls(text: str, urls_column: str) -> int:
    """Count total number of URLs."""
    return len(get_all_urls(text, urls_column))


def has_url(text: str, urls_column: str) -> int:
    """Check if any URL is present."""
    return 1 if count_urls(text, urls_column) > 0 else 0


def count_exclamations(text: str) -> int:
    """Count exclamation marks in text."""
    return text.count('!') if text else 0


def count_questions(text: str) -> int:
    """Count question marks in text."""
    return text.count('?') if text else 0


def get_text_length(text: str) -> int:
    """Get character count of text."""
    return len(text) if text else 0


def get_word_count(text: str) -> int:
    """Get word count of text."""
    if not text:
        return 0
    return len(text.split())


def get_caps_ratio(text: str) -> float:
    """
    Calculate ratio of ALL CAPS tokens to total tokens.
    
    Args:
        text: Input text
        
    Returns:
        Ratio (0.0 to 1.0)
    """
    if not text:
        return 0.0
    
    tokens = text.split()
    if not tokens:
        return 0.0
    
    # Count tokens that are all uppercase (at least 2 chars)
    caps_tokens = sum(1 for t in tokens if len(t) >= 2 and t.isupper())
    return caps_tokens / len(tokens)


def count_keyword_hits(text: str, keywords: List[str] = PHISHING_KEYWORDS) -> int:
    """
    Count occurrences of phishing-related keywords.
    
    Args:
        text: Input text
        keywords: List of keywords to search for
        
    Returns:
        Total count of keyword matches
    """
    if not text:
        return 0
    
    text_lower = text.lower()
    return sum(text_lower.count(kw.lower()) for kw in keywords)


def has_suspicious_tld(text: str, urls_column: str, suspicious_tlds: Set[str] = SUSPICIOUS_TLDS) -> int:
    """
    Check if any URL has a suspicious TLD.
    
    Args:
        text: Email text
        urls_column: URLs column content
        suspicious_tlds: Set of suspicious TLDs
        
    Returns:
        1 if suspicious TLD found, 0 otherwise
    """
    urls = get_all_urls(text, urls_column)
    
    for url in urls:
        domain = extract_domain(url)
        if domain:
            tld = extract_tld(domain)
            if tld and tld in suspicious_tlds:
                return 1
    return 0


def uses_url_shortener(text: str, urls_column: str, shorteners: Set[str] = URL_SHORTENERS) -> int:
    """
    Check if any URL uses a known URL shortener.
    
    Args:
        text: Email text
        urls_column: URLs column content
        shorteners: Set of shortener domains
        
    Returns:
        1 if shortener found, 0 otherwise
    """
    urls = get_all_urls(text, urls_column)
    
    for url in urls:
        domain = extract_domain(url)
        if domain:
            # Check if domain is or ends with a shortener
            for shortener in shorteners:
                if domain == shortener or domain.endswith('.' + shortener):
                    return 1
    return 0


def extract_all_heuristics(text: str, urls_column: str) -> Dict[str, float]:
    """
    Extract all heuristic features for a single sample.
    
    Args:
        text: Combined subject + body text
        urls_column: URLs column content
        
    Returns:
        Dictionary of feature name -> value
    """
    return {
        'url_count': count_urls(text, urls_column),
        'has_url': has_url(text, urls_column),
        'exclamation_count': count_exclamations(text),
        'question_count': count_questions(text),
        'text_len': get_text_length(text),
        'word_count': get_word_count(text),
        'caps_ratio': get_caps_ratio(text),
        'keyword_hits': count_keyword_hits(text),
        'suspicious_tld': has_suspicious_tld(text, urls_column),
        'uses_shortener': uses_url_shortener(text, urls_column),
    }


# =============================================================================
# SCIKIT-LEARN COMPATIBLE TRANSFORMER
# =============================================================================

class HeuristicFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer for extracting heuristic features.
    
    Expects input as a DataFrame with 'text' and 'urls' columns,
    or as a 2D array where column 0 is text and column 1 is urls.
    """
    
    FEATURE_NAMES = [
        'url_count',
        'has_url',
        'exclamation_count',
        'question_count',
        'text_len',
        'word_count',
        'caps_ratio',
        'keyword_hits',
        'suspicious_tld',
        'uses_shortener',
    ]
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """Fit method (no-op for this transformer)."""
        return self
    
    def transform(self, X) -> np.ndarray:
        """
        Transform input data to heuristic features.
        
        Args:
            X: DataFrame with 'text' and 'urls' columns, or 2D array
            
        Returns:
            NumPy array of shape (n_samples, n_features)
        """
        features = []
        
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            for _, row in X.iterrows():
                text = str(row.get('text', ''))
                urls = str(row.get('urls', ''))
                feat_dict = extract_all_heuristics(text, urls)
                features.append([feat_dict[name] for name in self.FEATURE_NAMES])
        else:
            # Handle array input (assume column 0 = text, column 1 = urls)
            X = np.atleast_2d(X)
            for row in X:
                text = str(row[0]) if len(row) > 0 else ''
                urls = str(row[1]) if len(row) > 1 else ''
                feat_dict = extract_all_heuristics(text, urls)
                features.append([feat_dict[name] for name in self.FEATURE_NAMES])
        
        return np.array(features, dtype=np.float64)
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return feature names for output."""
        return np.array(self.FEATURE_NAMES)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_heuristic_feature_names() -> List[str]:
    """Get list of heuristic feature names."""
    return HeuristicFeaturesTransformer.FEATURE_NAMES.copy()


def compute_heuristic_contributions(
    text: str,
    urls: str,
    coefficients: Dict[str, float]
) -> List[Tuple[str, float, float]]:
    """
    Compute contribution of each heuristic feature to prediction.
    
    Args:
        text: Email text
        urls: URLs string
        coefficients: Dict mapping feature name to coefficient
        
    Returns:
        List of (feature_name, value, contribution) tuples sorted by |contribution|
    """
    features = extract_all_heuristics(text, urls)
    contributions = []
    
    for name, value in features.items():
        if name in coefficients:
            contribution = value * coefficients[name]
            contributions.append((name, value, contribution))
    
    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x[2]), reverse=True)
    return contributions
