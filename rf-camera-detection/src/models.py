"""
Models Module
=============
Machine learning models for both Track A (WiFi traffic) and Track B (IQ signals).

Track A: Decision Tree, Linear SVM, Logistic Regression, Random Forest, XGBoost, 1D-CNN/LSTM
Track B: CNN+LSTM for IQ signal classification (scaffold)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# =============================================================================
# TRACK A: Traditional ML Models
# =============================================================================

def get_decision_tree(config: Optional[dict] = None, class_weight: Optional[dict] = None):
    """Create a Decision Tree classifier."""
    cfg = config or {}
    cw = class_weight
    if cw is None and cfg.get('class_weight') == 'balanced':
        cw = 'balanced'

    return DecisionTreeClassifier(
        criterion=cfg.get('criterion', 'gini'),
        max_depth=cfg.get('max_depth', 12),
        min_samples_split=cfg.get('min_samples_split', 5),
        min_samples_leaf=cfg.get('min_samples_leaf', 2),
        class_weight=cw,
        random_state=42
    )


def get_linear_svm(config: Optional[dict] = None, class_weight: Optional[dict] = None):
    """Create a scalable linear SVM classifier."""
    cfg = config or {}
    cw = class_weight
    if cw is None and cfg.get('class_weight') == 'balanced':
        cw = 'balanced'

    return LinearSVC(
        C=cfg.get('C', 1.0),
        class_weight=cw,
        max_iter=cfg.get('max_iter', 2000),
        tol=cfg.get('tol', 1e-3),
        dual='auto',
        random_state=42
    )

def get_logistic_regression(config: Optional[dict] = None, class_weight: Optional[dict] = None):
    """
    Create Logistic Regression classifier.

    Args:
        config: Model configuration
        class_weight: Optional class weights for imbalance

    Returns:
        LogisticRegression model
    """
    cfg = config or {}

    # Use provided class_weight or get from config
    cw = class_weight
    if cw is None and cfg.get('class_weight') == 'balanced':
        cw = 'balanced'

    return LogisticRegression(
        C=cfg.get('C', 1.0),
        max_iter=cfg.get('max_iter', 1000),
        class_weight=cw,
        random_state=42,
        n_jobs=-1
    )


def get_random_forest(config: Optional[dict] = None, class_weight: Optional[dict] = None):
    """
    Create Random Forest classifier.

    Args:
        config: Model configuration
        class_weight: Optional class weights for imbalance

    Returns:
        RandomForestClassifier model
    """
    cfg = config or {}

    cw = class_weight
    if cw is None and cfg.get('class_weight') == 'balanced':
        cw = 'balanced'

    return RandomForestClassifier(
        n_estimators=cfg.get('n_estimators', 100),
        max_depth=cfg.get('max_depth', 20),
        min_samples_split=cfg.get('min_samples_split', 5),
        min_samples_leaf=cfg.get('min_samples_leaf', 2),
        class_weight=cw,
        random_state=42,
        n_jobs=cfg.get('n_jobs', -1)
    )


def get_xgboost(config: Optional[dict] = None, scale_pos_weight: Optional[float] = None):
    """
    Create XGBoost classifier.

    Args:
        config: Model configuration
        scale_pos_weight: Weight for positive class (for imbalance)

    Returns:
        XGBClassifier model
    """
    cfg = config or {}

    spw = scale_pos_weight
    if spw is None:
        spw = cfg.get('scale_pos_weight', 1.0)

    return XGBClassifier(
        n_estimators=cfg.get('n_estimators', 100),
        max_depth=cfg.get('max_depth', 6),
        learning_rate=cfg.get('learning_rate', 0.1),
        subsample=cfg.get('subsample', 0.8),
        colsample_bytree=cfg.get('colsample_bytree', 0.8),
        scale_pos_weight=spw if spw != "auto" else 1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )


# =============================================================================
# TRACK A: Neural Network Models for Flow Sequences
# =============================================================================

class FlowCNNLSTM(nn.Module):
    """
    1D CNN + LSTM model for packet sequence classification.

    Input: (batch, channels, sequence_length)
    - channels: 2 (packet sizes, inter-arrival times)

    Architecture:
    1. 1D CNN layers for local pattern extraction
    2. LSTM for temporal dependencies
    3. Fully connected classifier
    """

    def __init__(
        self,
        sequence_length: int = 100,
        input_channels: int = 2,
        cnn_filters: List[int] = [32, 64],
        cnn_kernel_size: int = 3,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        """
        Initialize the CNN+LSTM model.

        Args:
            sequence_length: Length of input sequences
            input_channels: Number of input channels (2 for packet size + IAT)
            cnn_filters: List of filter counts for CNN layers
            cnn_kernel_size: Kernel size for CNN layers
            lstm_hidden_size: LSTM hidden state size
            lstm_num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_classes: Number of output classes
        """
        super(FlowCNNLSTM, self).__init__()

        self.sequence_length = sequence_length
        self.input_channels = input_channels

        # CNN layers
        cnn_layers = []
        in_channels = input_channels
        for filters in cnn_filters:
            cnn_layers.extend([
                nn.Conv1d(in_channels, filters, cnn_kernel_size, padding=cnn_kernel_size // 2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_channels = filters

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate CNN output size
        cnn_out_length = sequence_length
        for _ in cnn_filters:
            cnn_out_length = cnn_out_length // 2

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        # Classifier
        lstm_out_size = lstm_hidden_size * 2  # Bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # CNN: (batch, channels, seq) -> (batch, filters[-1], seq//4)
        x = self.cnn(x)

        # Reshape for LSTM: (batch, seq, features)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Use last output
        x = lstm_out[:, -1, :]

        # Classify
        x = self.classifier(x)

        return x


# =============================================================================
# TRACK B: IQ Signal CNN+LSTM Model
# =============================================================================

class IQSignalCNNLSTM(nn.Module):
    """
    CNN + LSTM model for IQ signal classification (camera detection).

    Input: (batch, 2, sample_length) where 2 = I and Q channels

    This model is designed to:
    1. Extract frequency and temporal patterns from raw IQ signals
    2. Detect characteristics unique to camera WiFi transmissions
    3. Distinguish camera-on vs camera-off states

    Architecture:
    1. 1D CNN layers for extracting signal features
    2. Bidirectional LSTM for temporal patterns
    3. Attention mechanism (optional)
    4. Binary classifier

    TODO: Fine-tune architecture when real SDR data is available
    """

    def __init__(
        self,
        sample_length: int = 1024,
        input_channels: int = 2,  # I and Q
        cnn_filters: List[int] = [64, 128, 256],
        cnn_kernel_size: int = 7,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.4,
        use_attention: bool = True,
        num_classes: int = 2
    ):
        """
        Initialize IQ Signal model.

        Args:
            sample_length: Number of IQ samples per input
            input_channels: Number of input channels (2 for I/Q)
            cnn_filters: Filter counts for CNN layers
            cnn_kernel_size: Kernel size for 1D convolutions
            lstm_hidden_size: LSTM hidden size
            lstm_num_layers: Number of LSTM layers
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
            num_classes: Number of output classes (2 for binary)
        """
        super(IQSignalCNNLSTM, self).__init__()

        self.sample_length = sample_length
        self.use_attention = use_attention

        # CNN Feature Extractor
        cnn_layers = []
        in_channels = input_channels
        for i, filters in enumerate(cnn_filters):
            cnn_layers.extend([
                nn.Conv1d(in_channels, filters, cnn_kernel_size,
                         padding=cnn_kernel_size // 2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout if i > 0 else dropout / 2)
            ])
            in_channels = filters

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate CNN output size
        cnn_out_length = sample_length
        for _ in cnn_filters:
            cnn_out_length = cnn_out_length // 2

        self.cnn_out_channels = cnn_filters[-1]
        self.cnn_out_length = cnn_out_length

        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_out_size = lstm_hidden_size * 2  # Bidirectional

        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_out_size, lstm_hidden_size),
                nn.Tanh(),
                nn.Linear(lstm_hidden_size, 1, bias=False)
            )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(lstm_hidden_size // 2, num_classes)
        )

    def attention_pooling(
        self,
        lstm_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention pooling over LSTM outputs.

        Args:
            lstm_output: (batch, seq_len, hidden*2)

        Returns:
            Weighted sum: (batch, hidden*2)
        """
        attention_weights = self.attention(lstm_output)  # (batch, seq, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted = lstm_output * attention_weights
        return weighted.sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: IQ signal of shape (batch, 2, sample_length)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, filters, reduced_length)

        # Reshape for LSTM: (batch, seq, features)
        lstm_in = cnn_out.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(lstm_in)  # (batch, seq, hidden*2)

        # Pooling
        if self.use_attention:
            features = self.attention_pooling(lstm_out)
        else:
            features = lstm_out[:, -1, :]  # Last timestep

        # Classification
        logits = self.classifier(features)

        return logits


# =============================================================================
# Model Factory
# =============================================================================

def get_model(
    model_name: str,
    config: Optional[dict] = None,
    class_weight: Optional[dict] = None,
    **kwargs
):
    """
    Factory function to get a model by name.

    Args:
        model_name: Name of the model
        config: Model configuration
        class_weight: Class weights for imbalanced data
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Model instance
    """
    model_name = model_name.lower().replace("-", "_").replace(" ", "_")

    # Traditional ML models
    if model_name in ["decision_tree", "decisiontree", "dt"]:
        return get_decision_tree(config, class_weight)

    elif model_name in ["linear_svm", "svm", "linearsvc"]:
        return get_linear_svm(config, class_weight)

    elif model_name in ["logistic_regression", "lr", "logreg"]:
        return get_logistic_regression(config, class_weight)

    elif model_name in ["random_forest", "rf"]:
        return get_random_forest(config, class_weight)

    elif model_name in ["xgboost", "xgb"]:
        spw = None
        if class_weight:
            # Convert class weight dict to scale_pos_weight
            spw = class_weight.get(1, 1.0) / class_weight.get(0, 1.0)
        return get_xgboost(config, spw)

    # Neural network models
    elif model_name in ["flow_cnn_lstm", "cnn_lstm"]:
        cfg = config or {}
        return FlowCNNLSTM(
            sequence_length=cfg.get('sequence_length', 100),
            input_channels=cfg.get('input_channels', 2),
            cnn_filters=cfg.get('cnn_filters', [32, 64]),
            cnn_kernel_size=cfg.get('cnn_kernel_size', 3),
            lstm_hidden_size=cfg.get('lstm_hidden_size', 64),
            lstm_num_layers=cfg.get('lstm_num_layers', 1),
            dropout=cfg.get('dropout', 0.3),
            **kwargs
        )

    elif model_name in ["iq_cnn_lstm", "iq_model"]:
        cfg = config or {}
        return IQSignalCNNLSTM(
            sample_length=cfg.get('sample_length', 1024),
            input_channels=cfg.get('input_channels', 2),
            cnn_filters=cfg.get('cnn_filters', [64, 128, 256]),
            cnn_kernel_size=cfg.get('cnn_kernel_size', 7),
            lstm_hidden_size=cfg.get('lstm_hidden_size', 128),
            lstm_num_layers=cfg.get('lstm_num_layers', 2),
            dropout=cfg.get('dropout', 0.4),
            use_attention=cfg.get('use_attention', True),
            **kwargs
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
