"""
Utility functions for logging, file I/O, and common operations.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with console handler.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Output file path
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        path: Output file path
        index: Whether to include index
    """
    df.to_csv(path, index=index, encoding="utf-8")


def load_csv(path: Path) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        path: Input file path
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(path, encoding="utf-8")


def format_percentage(value: float) -> str:
    """Format float as percentage string."""
    return f"{value * 100:.2f}%"


def print_separator(char: str = "=", length: int = 60) -> None:
    """Print a separator line."""
    print(char * length)


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted summary of evaluation metrics.
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print_separator()
    print("EVALUATION METRICS SUMMARY")
    print_separator()
    
    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 42)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ["precision", "recall", "f1_score", "roc_auc", "false_positive_rate", "accuracy"]:
                print(f"{key:<25} {format_percentage(value):>15}")
            else:
                print(f"{key:<25} {value:>15.4f}")
        elif isinstance(value, int):
            print(f"{key:<25} {value:>15}")
        elif isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k:<23} {v:>15}")
    
    print_separator()
