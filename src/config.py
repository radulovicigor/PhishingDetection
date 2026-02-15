"""
Configuration module for Phish Detector.
All hyperparameters, paths, and constants are centralized here.
"""
from pathlib import Path
from typing import List, Set

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = DATA_DIR / "raw_dataset.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.csv"
THRESHOLD_METRICS_PATH = REPORTS_DIR / "threshold_metrics.csv"
TOP_FEATURES_PHISHING_PATH = REPORTS_DIR / "top_features_phishing.csv"
TOP_FEATURES_LEGIT_PATH = REPORTS_DIR / "top_features_legit.csv"

# =============================================================================
# RANDOM STATE & REPRODUCIBILITY
# =============================================================================
RANDOM_STATE: int = 42

# Labelna šuma u treningu (isključeno – koristimo teške legit primjere u datasetu umjesto toga)
LABEL_NOISE_RATE: float = 0.0

# =============================================================================
# DATA SPLIT
# =============================================================================
TEST_SIZE: float = 0.2

# =============================================================================
# TF-IDF VECTORIZER
# =============================================================================
TFIDF_MAX_FEATURES: int = 30000
TFIDF_NGRAM_RANGE: tuple = (1, 2)
TFIDF_MIN_DF: int = 3
TFIDF_STRIP_ACCENTS: str = "unicode"

# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================
LR_SOLVER: str = "liblinear"
LR_CLASS_WEIGHT: str = "balanced"
LR_MAX_ITER: int = 1000
LR_C: float = 1.0

# =============================================================================
# HEURISTIC FEATURE KEYWORDS
# =============================================================================
PHISHING_KEYWORDS: List[str] = [
    "verify", "password", "urgent", "account", "login", "bank",
    "suspended", "update", "confirm", "click", "immediately",
    "expire", "security", "alert", "warning", "unauthorized",
    "credit", "card", "ssn", "social security", "paypal",
    "ebay", "amazon", "apple", "microsoft", "netflix"
]

# =============================================================================
# SUSPICIOUS TLDs
# =============================================================================
SUSPICIOUS_TLDS: Set[str] = {"tk", "ru", "cn", "top", "xyz", "pw", "cc", "ga", "ml", "cf"}

# =============================================================================
# URL SHORTENERS
# =============================================================================
URL_SHORTENERS: Set[str] = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "bit.do", "mcaf.ee"
}

# =============================================================================
# EVALUATION THRESHOLDS
# =============================================================================
EVALUATION_THRESHOLDS: List[float] = [0.3, 0.5, 0.7, 0.9]

# =============================================================================
# EXPLAINABILITY
# =============================================================================
TOP_FEATURES_COUNT: int = 30
LOCAL_EXPLANATION_TOP_N: int = 10

# =============================================================================
# API
# =============================================================================
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
MODEL_VERSION: str = "1.0.0"

# =============================================================================
# REQUIRED COLUMNS
# =============================================================================
REQUIRED_COLUMNS: List[str] = ["ender", "receiver", "date", "subject", "body", "label", "urls"]
