"""
Evaluation Module
=================
Metrics computation, visualization, and results reporting.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class (optional)

    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['avg_precision'] = average_precision_score(y_true, y_prob)
        except ValueError as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
            metrics['roc_auc'] = np.nan
            metrics['avg_precision'] = np.nan

    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = ['Non-Camera', 'Camera']
) -> str:
    """
    Generate detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names for each class

    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=target_names)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    class_names: List[str] = ['Non-Camera', 'Camera'],
    save_path: Optional[str] = None,
    normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        class_names: Names for each class
        save_path: Path to save the figure
        normalize: Whether to normalize by row (true labels)

    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2%'
    else:
        cm_display = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add raw counts as secondary annotation
    if normalize:
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j + 0.5, i + 0.7, f'({cm[i, j]})',
                       ha='center', va='center', fontsize=9, color='gray')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.2)

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AP = {ap:.3f})')
    ax.fill_between(recall, precision, alpha=0.2)

    # Add baseline (proportion of positive class)
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved PR curve to {save_path}")

    return fig


def compare_models(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compare multiple models and create a results table.

    Args:
        results: Dictionary of model_name -> metrics dict
        save_path: Path to save CSV

    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(results).T
    df = df.round(4)
    df = df.sort_values('f1', ascending=False)

    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    print(df.to_string())
    print("="*70 + "\n")

    if save_path:
        df.to_csv(save_path)
        logger.info(f"Saved results table to {save_path}")

    return df


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create bar chart comparing models across metrics.

    Args:
        results: Dictionary of model_name -> metrics dict
        metrics: Metrics to include in comparison
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    df = pd.DataFrame(results).T
    df = df[metrics]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.15
    multiplier = 0

    for metric in metrics:
        offset = width * multiplier
        bars = ax.bar(x + offset, df[metric], width, label=metric.upper())
        multiplier += 1

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison Across Metrics', fontsize=14)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved model comparison to {save_path}")

    return fig


class Evaluator:
    """
    Comprehensive evaluator for model performance.

    Handles evaluation, visualization, and results storage.
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize evaluator.

        Args:
            output_dir: Directory for saving outputs
        """
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.results_dir = self.output_dir / "results"

        # Create directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.all_results: Dict[str, Dict] = {}

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        save_plots: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a single model.

        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            save_plots: Whether to save visualizations

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name}...")

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_prob)
        self.all_results[model_name] = metrics

        # Print classification report
        print(f"\n{model_name} Classification Report:")
        print(get_classification_report(y_true, y_pred))

        if save_plots:
            # Confusion matrix
            plot_confusion_matrix(
                y_true, y_pred,
                title=f"{model_name} - Confusion Matrix",
                save_path=str(self.plots_dir / f"{model_name}_confusion_matrix.png")
            )
            plt.close()

            # ROC curve (if probabilities available)
            if y_prob is not None:
                plot_roc_curve(
                    y_true, y_prob,
                    title=f"{model_name} - ROC Curve",
                    save_path=str(self.plots_dir / f"{model_name}_roc_curve.png")
                )
                plt.close()

                plot_precision_recall_curve(
                    y_true, y_prob,
                    title=f"{model_name} - Precision-Recall Curve",
                    save_path=str(self.plots_dir / f"{model_name}_pr_curve.png")
                )
                plt.close()

        return metrics

    def save_results(self, filename: str = "results.csv"):
        """
        Save all results to CSV and JSON.

        Args:
            filename: Base filename for results
        """
        if not self.all_results:
            logger.warning("No results to save")
            return

        # Compare models
        df = compare_models(
            self.all_results,
            save_path=str(self.results_dir / filename)
        )

        # Save as JSON too
        json_path = self.results_dir / filename.replace('.csv', '.json')
        with open(json_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)

        # Model comparison plot
        plot_model_comparison(
            self.all_results,
            save_path=str(self.plots_dir / "model_comparison.png")
        )
        plt.close()

        logger.info(f"Results saved to {self.results_dir}")

        return df
