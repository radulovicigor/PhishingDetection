"""
Visualization module.
Generates and saves all metrics charts and plots as images.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from src.config import (
    PROJECT_ROOT,
    MODEL_PATH,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    THRESHOLD_METRICS_PATH,
    TOP_FEATURES_PHISHING_PATH,
    TOP_FEATURES_LEGIT_PATH,
    RANDOM_STATE,
)
from src.preprocessing import run_preprocessing_pipeline
from src.train import load_model
from src.utils import setup_logger, load_json, load_csv

logger = setup_logger(__name__)

# Output directory for visualizations
VISUALIZATIONS_DIR = PROJECT_ROOT / "Metrike"

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'primary': '#06b6d4',      # Cyan
    'secondary': '#8b5cf6',    # Purple
    'success': '#22c55e',      # Green
    'danger': '#ef4444',       # Red
    'warning': '#f59e0b',      # Orange
    'background': '#1a2234',
    'text': '#e2e8f0',
    'muted': '#64748b',
}


def setup_plot_style():
    """Configure matplotlib style for dark theme."""
    plt.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'axes.facecolor': '#111827',
        'axes.edgecolor': '#334155',
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': '#334155',
        'grid.alpha': 0.5,
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Visualizations will be saved to: {VISUALIZATIONS_DIR}")


def save_figure(fig: plt.Figure, filename: str, dpi: int = 150):
    """Save figure to file."""
    path = VISUALIZATIONS_DIR / filename
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: {path}")


def create_metrics_summary(metrics: Dict[str, Any]) -> plt.Figure:
    """
    Create a visual summary of all key metrics.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Metrics to display
    metric_items = [
        ('Accuracy', metrics['accuracy'], COLORS['success']),
        ('Precision', metrics['precision'], COLORS['primary']),
        ('Recall', metrics['recall'], COLORS['primary']),
        ('F1 Score', metrics['f1_score'], COLORS['success']),
        ('ROC-AUC', metrics.get('roc_auc', 0) or 0, COLORS['secondary']),
        ('FPR', metrics['false_positive_rate'], COLORS['danger']),
    ]
    
    names = [m[0] for m in metric_items]
    values = [m[1] * 100 for m in metric_items]
    colors = [m[2] for m in metric_items]
    
    bars = ax.barh(names, values, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        label_x = width + 1 if width < 95 else width - 8
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}%', va='center', ha='left' if width < 95 else 'right',
                fontweight='bold', fontsize=12, color=COLORS['text'])
    
    ax.set_xlim(0, 105)
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.tick_params(left=False)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_confusion_matrix(metrics: Dict[str, Any]) -> plt.Figure:
    """
    Create confusion matrix heatmap.
    """
    cm = metrics['confusion_matrix']
    
    # Create matrix
    matrix = np.array([
        [cm['true_negatives'], cm['false_positives']],
        [cm['false_negatives'], cm['true_positives']]
    ])
    
    # Calculate percentages
    total = matrix.sum()
    matrix_pct = matrix / total * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Custom colormap
    cmap = sns.diverging_palette(10, 150, as_cmap=True)
    
    # Create heatmap
    sns.heatmap(matrix, annot=False, cmap=cmap, ax=ax, 
                cbar=True, square=True, linewidths=2, linecolor=COLORS['background'])
    
    # Add annotations with counts and percentages
    labels = [
        [f'TN\n{cm["true_negatives"]:,}\n({matrix_pct[0,0]:.1f}%)', 
         f'FP\n{cm["false_positives"]:,}\n({matrix_pct[0,1]:.1f}%)'],
        [f'FN\n{cm["false_negatives"]:,}\n({matrix_pct[1,0]:.1f}%)', 
         f'TP\n{cm["true_positives"]:,}\n({matrix_pct[1,1]:.1f}%)']
    ]
    
    for i in range(2):
        for j in range(2):
            color = 'white' if (i == j) else '#1a1a1a'
            ax.text(j + 0.5, i + 0.5, labels[i][j], 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color=color)
    
    ax.set_xlabel('Predicted Label', fontsize=13, labelpad=15)
    ax.set_ylabel('Actual Label', fontsize=13, labelpad=15)
    ax.set_xticklabels(['Legitimate', 'Phishing'], fontsize=12)
    ax.set_yticklabels(['Legitimate', 'Phishing'], fontsize=12, rotation=0)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    
    # Add FPR annotation
    fpr = metrics['false_positive_rate']
    ax.text(0.5, -0.15, f'False Positive Rate (FPR): {fpr:.2%}', 
            transform=ax.transAxes, ha='center', fontsize=12,
            color=COLORS['danger'], fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_roc_curve(pipeline, test_df: pd.DataFrame) -> plt.Figure:
    """
    Create ROC curve plot.
    """
    X_test = test_df[['text', 'urls']].copy()
    y_true = test_df['label'].values
    
    # Get probabilities
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor('#111827')
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color=COLORS['primary'], lw=3, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], color=COLORS['muted'], lw=2, linestyle='--',
            label='Random Classifier')
    
    # Fill under curve
    ax.fill_between(fpr, tpr, alpha=0.2, color=COLORS['primary'])
    
    # Mark optimal point (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color=COLORS['success'], 
               s=200, zorder=5, edgecolor='white', linewidth=2)
    ax.annotate(f'Optimal\nThreshold: {optimal_threshold:.2f}',
                xy=(fpr[optimal_idx], tpr[optimal_idx]),
                xytext=(fpr[optimal_idx] + 0.1, tpr[optimal_idx] - 0.1),
                fontsize=11, color=COLORS['text'],
                arrowprops=dict(arrowstyle='->', color=COLORS['text']))
    
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve - Receiver Operating Characteristic', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.8)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_threshold_analysis(threshold_df: pd.DataFrame) -> plt.Figure:
    """
    Create threshold analysis chart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(COLORS['background'])
    
    thresholds = threshold_df['threshold'].values
    
    # Left plot: Precision, Recall, F1
    ax1 = axes[0]
    ax1.set_facecolor('#111827')
    
    ax1.plot(thresholds, threshold_df['precision'] * 100, 'o-', 
             color=COLORS['primary'], lw=2.5, markersize=10, label='Precision')
    ax1.plot(thresholds, threshold_df['recall'] * 100, 's-', 
             color=COLORS['success'], lw=2.5, markersize=10, label='Recall')
    ax1.plot(thresholds, threshold_df['f1_score'] * 100, '^-', 
             color=COLORS['secondary'], lw=2.5, markersize=10, label='F1 Score')
    
    ax1.set_xlabel('Classification Threshold', fontsize=12)
    ax1.set_ylabel('Score (%)', fontsize=12)
    ax1.set_title('Precision / Recall / F1 vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=11)
    ax1.set_ylim([70, 102])
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(thresholds)
    
    # Right plot: FPR and FNR
    ax2 = axes[1]
    ax2.set_facecolor('#111827')
    
    fnr = 1 - threshold_df['recall']  # False Negative Rate
    fpr = threshold_df['false_positive_rate']
    
    ax2.bar(thresholds - 0.03, fpr * 100, width=0.06, 
            color=COLORS['danger'], label='False Positive Rate', alpha=0.9)
    ax2.bar(thresholds + 0.03, fnr * 100, width=0.06, 
            color=COLORS['warning'], label='False Negative Rate', alpha=0.9)
    
    # Add value labels
    for i, (t, f, n) in enumerate(zip(thresholds, fpr, fnr)):
        ax2.text(t - 0.03, f * 100 + 0.5, f'{f*100:.1f}%', ha='center', 
                 fontsize=9, color=COLORS['text'])
        ax2.text(t + 0.03, n * 100 + 0.5, f'{n*100:.1f}%', ha='center', 
                 fontsize=9, color=COLORS['text'])
    
    ax2.set_xlabel('Classification Threshold', fontsize=12)
    ax2.set_ylabel('Error Rate (%)', fontsize=12)
    ax2.set_title('Error Rates vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(thresholds)
    
    plt.tight_layout()
    return fig


def create_feature_importance(phishing_df: pd.DataFrame, legit_df: pd.DataFrame) -> plt.Figure:
    """
    Create feature importance bar chart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Top phishing features (left)
    ax1 = axes[0]
    ax1.set_facecolor('#111827')
    
    top_phishing = phishing_df.head(20)
    y_pos = range(len(top_phishing))
    
    bars1 = ax1.barh(y_pos, top_phishing['coefficient'], 
                     color=COLORS['danger'], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_phishing['feature'], fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('Coefficient Weight', fontsize=12)
    ax1.set_title('Top Phishing Indicators', fontsize=14, fontweight='bold', color=COLORS['danger'])
    ax1.grid(True, alpha=0.3, axis='x')
    
    for spine in ax1.spines.values():
        spine.set_color('#334155')
    
    # Top legitimate features (right)
    ax2 = axes[1]
    ax2.set_facecolor('#111827')
    
    top_legit = legit_df.head(20)
    y_pos = range(len(top_legit))
    
    bars2 = ax2.barh(y_pos, top_legit['coefficient'].abs(), 
                     color=COLORS['success'], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_legit['feature'], fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel('Coefficient Weight (absolute)', fontsize=12)
    ax2.set_title('Top Legitimate Indicators', fontsize=14, fontweight='bold', color=COLORS['success'])
    ax2.grid(True, alpha=0.3, axis='x')
    
    for spine in ax2.spines.values():
        spine.set_color('#334155')
    
    plt.tight_layout()
    return fig


def create_class_distribution(metrics: Dict[str, Any]) -> plt.Figure:
    """
    Create pie chart of class distribution.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor(COLORS['background'])
    
    sizes = [metrics['negative_samples'], metrics['positive_samples']]
    labels = ['Legitimate', 'Phishing']
    colors = [COLORS['success'], COLORS['danger']]
    explode = (0.02, 0.02)
    
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes)):,})',
        startangle=90, textprops={'fontsize': 12, 'color': COLORS['text']},
        wedgeprops={'edgecolor': COLORS['background'], 'linewidth': 2}
    )
    
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    ax.set_title('Test Set Class Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(wedges, [f'{l}: {s:,}' for l, s in zip(labels, sizes)],
              loc='lower center', bbox_to_anchor=(0.5, -0.05),
              fontsize=11, framealpha=0.8, ncol=2)
    
    plt.tight_layout()
    return fig


def create_metrics_table(metrics: Dict[str, Any]) -> plt.Figure:
    """
    Create a nice metrics table as an image.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    ax.axis('off')
    
    # Table data
    cm = metrics['confusion_matrix']
    data = [
        ['Accuracy', f"{metrics['accuracy']*100:.2f}%"],
        ['Precision', f"{metrics['precision']*100:.2f}%"],
        ['Recall (Sensitivity)', f"{metrics['recall']*100:.2f}%"],
        ['F1 Score', f"{metrics['f1_score']*100:.2f}%"],
        ['ROC-AUC', f"{(metrics.get('roc_auc') or 0)*100:.2f}%"],
        ['', ''],
        ['False Positive Rate', f"{metrics['false_positive_rate']*100:.2f}%"],
        ['False Negative Rate', f"{(1-metrics['recall'])*100:.2f}%"],
        ['', ''],
        ['True Positives (TP)', f"{cm['true_positives']:,}"],
        ['True Negatives (TN)', f"{cm['true_negatives']:,}"],
        ['False Positives (FP)', f"{cm['false_positives']:,}"],
        ['False Negatives (FN)', f"{cm['false_negatives']:,}"],
        ['', ''],
        ['Total Test Samples', f"{metrics['total_samples']:,}"],
        ['Phishing Samples', f"{metrics['positive_samples']:,}"],
        ['Legitimate Samples', f"{metrics['negative_samples']:,}"],
    ]
    
    # Create table
    table = ax.table(
        cellText=data,
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='left',
        colWidths=[0.5, 0.3]
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    for key, cell in table.get_celld().items():
        cell.set_facecolor('#1a2234')
        cell.set_edgecolor('#334155')
        cell.set_text_props(color=COLORS['text'])
        
        if key[0] == 0:  # Header row
            cell.set_facecolor(COLORS['primary'])
            cell.set_text_props(color='white', fontweight='bold')
        elif key[1] == 1:  # Value column
            cell.set_text_props(fontweight='bold', fontfamily='monospace')
    
    ax.set_title('Complete Metrics Summary', fontsize=18, fontweight='bold', 
                 pad=30, color=COLORS['text'])
    
    plt.tight_layout()
    return fig


def generate_all_visualizations():
    """
    Generate all visualizations and save them to the Metrike folder.
    """
    logger.info("=" * 60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 60)
    
    setup_plot_style()
    ensure_output_dir()
    
    # Load metrics
    logger.info("\nLoading metrics...")
    if not METRICS_PATH.exists():
        logger.error("Metrics not found. Run 'python -m src.evaluate' first.")
        return 1
    
    metrics = load_json(METRICS_PATH)
    logger.info("Metrics loaded successfully")
    
    # Load threshold analysis
    threshold_df = None
    if THRESHOLD_METRICS_PATH.exists():
        threshold_df = load_csv(THRESHOLD_METRICS_PATH)
        logger.info("Threshold analysis loaded")
    
    # Load feature importance
    phishing_features = None
    legit_features = None
    if TOP_FEATURES_PHISHING_PATH.exists():
        phishing_features = load_csv(TOP_FEATURES_PHISHING_PATH)
        logger.info("Phishing features loaded")
    if TOP_FEATURES_LEGIT_PATH.exists():
        legit_features = load_csv(TOP_FEATURES_LEGIT_PATH)
        logger.info("Legitimate features loaded")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # 1. Metrics Summary
    logger.info("  Creating metrics summary...")
    fig = create_metrics_summary(metrics)
    save_figure(fig, '01_metrics_summary.png')
    
    # 2. Metrics Table
    logger.info("  Creating metrics table...")
    fig = create_metrics_table(metrics)
    save_figure(fig, '02_metrics_table.png')
    
    # 3. Confusion Matrix
    logger.info("  Creating confusion matrix...")
    fig = create_confusion_matrix(metrics)
    save_figure(fig, '03_confusion_matrix.png')
    
    # 4. Class Distribution
    logger.info("  Creating class distribution...")
    fig = create_class_distribution(metrics)
    save_figure(fig, '04_class_distribution.png')
    
    # 5. ROC Curve (requires model and test data)
    logger.info("  Creating ROC curve...")
    try:
        pipeline = load_model(MODEL_PATH)
        _, test_df = run_preprocessing_pipeline()
        fig = create_roc_curve(pipeline, test_df)
        save_figure(fig, '05_roc_curve.png')
    except Exception as e:
        logger.warning(f"Could not create ROC curve: {e}")
    
    # 6. Threshold Analysis
    if threshold_df is not None:
        logger.info("  Creating threshold analysis...")
        fig = create_threshold_analysis(threshold_df)
        save_figure(fig, '06_threshold_analysis.png')
    
    # 7. Feature Importance
    if phishing_features is not None and legit_features is not None:
        logger.info("  Creating feature importance...")
        fig = create_feature_importance(phishing_features, legit_features)
        save_figure(fig, '07_feature_importance.png')
    
    logger.info("\n" + "=" * 60)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nAll images saved to: {VISUALIZATIONS_DIR}")
    logger.info("\nGenerated files:")
    for f in sorted(VISUALIZATIONS_DIR.glob("*.png")):
        logger.info(f"  - {f.name}")
    
    return 0


def main():
    """Main entry point."""
    return generate_all_visualizations()


if __name__ == "__main__":
    sys.exit(main())
