"""
Data preprocessing module.
Handles loading, validation, cleaning, and splitting of the dataset.
"""
import re
from typing import Optional, Tuple

import pandas as pd
from dateutil import parser as date_parser
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.utils import setup_logger

logger = setup_logger(__name__)


def load_raw_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw CSV data.
    
    Args:
        path: Path to CSV file. If None, uses default from config.
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    data_path = path or RAW_DATA_PATH
    logger.info(f"Loading data from {data_path}")
    
    if not data_path.exists() if hasattr(data_path, 'exists') else not pd.io.common.file_exists(str(data_path)):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path, encoding="utf-8", low_memory=False)
    logger.info(f"Loaded {len(df)} rows")
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validate that required columns exist in the DataFrame.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If required columns are missing
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("All required columns present")


def clean_text(text: Optional[str]) -> str:
    """
    Clean a text field by handling NaN and normalizing whitespace.
    
    Args:
        text: Input text (may be NaN)
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    text = str(text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_date_robust(date_str: Optional[str]) -> Optional[pd.Timestamp]:
    """
    Robustly parse a date string.
    
    Args:
        date_str: Date string to parse
        
    Returns:
        Parsed timestamp or None if parsing fails
    """
    if pd.isna(date_str):
        return None
    try:
        return pd.Timestamp(date_parser.parse(str(date_str)))
    except (ValueError, TypeError, OverflowError):
        return None


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps to the DataFrame.
    
    Steps:
        1. Handle NaN in text fields
        2. Cast and validate labels
        3. Create combined text column
        4. Parse dates (for analytics only)
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Starting preprocessing...")
    df = df.copy()
    
    # Clean text fields
    logger.info("Cleaning text fields...")
    df["subject"] = df["subject"].apply(clean_text)
    df["body"] = df["body"].apply(clean_text)
    # Handle urls column - may be binary (0/1) or actual URLs
    df["urls"] = df["urls"].astype(str).apply(clean_text)
    df["sender"] = df["ender"].apply(clean_text)  # Note: dataset has 'ender' typo
    df["receiver"] = df["receiver"].apply(clean_text)
    
    # Handle labels
    logger.info("Processing labels...")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    
    # Keep only rows with valid labels (0 or 1)
    initial_count = len(df)
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    dropped = initial_count - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with invalid labels")
    
    # Create combined text column
    logger.info("Creating combined text column...")
    df["text"] = df["subject"] + "\n" + df["body"]
    
    # Parse dates (for analytics, not used in model)
    logger.info("Parsing dates (for analytics only)...")
    df["parsed_date"] = df["date"].apply(parse_date_robust)
    
    # Log date statistics
    valid_dates = df["parsed_date"].notna().sum()
    logger.info(f"Valid dates parsed: {valid_dates}/{len(df)}")
    
    # Check for outlier dates
    if valid_dates > 0:
        try:
            valid_date_series = df["parsed_date"].dropna()
            if len(valid_date_series) > 0:
                min_date = valid_date_series.min()
                max_date = valid_date_series.max()
                logger.info(f"Date range: {min_date} to {max_date}")
                if hasattr(max_date, 'year') and max_date.year > 2030:
                    logger.warning("Dataset contains outlier dates (e.g., year > 2030). Dates used only for analytics.")
        except Exception as e:
            logger.warning(f"Could not compute date range: {e}")
    
    logger.info(f"Preprocessing complete. Final dataset: {len(df)} rows")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train/test split.
    
    Args:
        df: Preprocessed DataFrame
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting data (test_size={test_size}, random_state={random_state})")
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )
    
    logger.info(f"Train set: {len(train_df)} rows")
    logger.info(f"Test set: {len(test_df)} rows")
    
    # Log class distribution
    train_phishing = train_df["label"].sum()
    test_phishing = test_df["label"].sum()
    logger.info(f"Train phishing rate: {train_phishing/len(train_df):.2%}")
    logger.info(f"Test phishing rate: {test_phishing/len(test_df):.2%}")
    
    return train_df, test_df


def run_preprocessing_pipeline(data_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete preprocessing pipeline.
    
    Args:
        data_path: Optional path to CSV file
        
    Returns:
        Tuple of (train_df, test_df)
    """
    df = load_raw_data(data_path)
    validate_columns(df)
    df = preprocess_dataframe(df)
    train_df, test_df = split_data(df)
    return train_df, test_df


if __name__ == "__main__":
    # Quick test
    try:
        train_df, test_df = run_preprocessing_pipeline()
        print(f"\nTrain shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"\nSample text:\n{train_df['text'].iloc[0][:200]}...")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please place your dataset in data/raw.csv")
