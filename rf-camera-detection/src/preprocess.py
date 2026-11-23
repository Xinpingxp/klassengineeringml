"""
Preprocessing Module
====================
Data cleaning, normalization, and train/test splitting for both tracks.
Implements device/session-based splitting to avoid data leakage.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)


# =============================================================================
# TRACK A: WiFi Traffic Preprocessing
# =============================================================================

class FlowPreprocessor:
    """
    Preprocessor for network flow data.

    Handles:
    - Missing value imputation
    - Feature scaling
    - Train/val/test splitting (by device to avoid leakage)
    - Class imbalance handling
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize preprocessor.

        Args:
            config: Configuration dictionary with preprocessing settings
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []

    def identify_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """
        Identify numeric, categorical, and MAC columns.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (numeric_cols, categorical_cols, mac_cols)
        """
        numeric_cols = []
        categorical_cols = []
        mac_cols = []

        # Columns to exclude from features
        exclude_patterns = ['label', 'is_camera', '_source', 'timestamp', 'time']

        for col in df.columns:
            col_lower = col.lower()

            # Skip excluded columns
            if any(ex in col_lower for ex in exclude_patterns):
                continue

            # Check for MAC columns
            if any(x in col_lower for x in ['mac', 'hw_addr']):
                mac_cols.append(col)
                continue

            # Check data type
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_cols.append(col)
            elif df[col].dtype == 'object':
                # Check if it's categorical (limited unique values)
                n_unique = df[col].nunique()
                if n_unique < 100:  # Arbitrary threshold
                    categorical_cols.append(col)
                # Otherwise skip (likely IPs, etc.)

        self.numeric_columns = numeric_cols
        self.categorical_columns = categorical_cols

        logger.info(f"Identified {len(numeric_cols)} numeric columns")
        logger.info(f"Identified {len(categorical_cols)} categorical columns")
        logger.info(f"Identified {len(mac_cols)} MAC columns")

        return numeric_cols, categorical_cols, mac_cols

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the flow data.

        - Remove duplicates
        - Handle missing values
        - Remove infinite values

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        initial_rows = len(df)

        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        required_numeric_cols = [col for col in self.numeric_columns if col in df.columns]
        df = df.dropna(subset=required_numeric_cols)

        logger.info(f"Final row count after cleaning: {len(df)}")

        return df

    def encode_categoricals(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical columns using LabelEncoder.

        Args:
            df: Input DataFrame
            fit: Whether to fit the encoders (True for training data)

        Returns:
            DataFrame with encoded categoricals
        """
        df = df.copy()

        for col in self.categorical_columns:
            if col not in df.columns:
                continue

            if fit:
                le = LabelEncoder()
                # Handle unseen values by including a placeholder
                unique_vals = df[col].astype(str).unique().tolist()
                le.fit(unique_vals + ['_unknown_'])
                self.label_encoders[col] = le

            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Transform, handling unseen values
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_
                    else le.transform(['_unknown_'])[0]
                )

        return df

    def get_device_groups(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract device grouping for stratified splitting.

        Uses MAC addresses to identify unique devices.

        Args:
            df: Input DataFrame

        Returns:
            Array of group labels (one per row)
        """
        for device_col in ['device_id', '_source_file']:
            if device_col in df.columns:
                groups = pd.factorize(df[device_col].astype(str))[0]
                logger.info(f"Found {len(np.unique(groups))} device groups from {device_col}")
                return groups

        # Find MAC columns
        mac_cols = [c for c in df.columns
                   if any(x in c.lower() for x in ['mac', 'hw_addr'])]

        if not mac_cols:
            logger.warning("No MAC columns found. Using random splitting.")
            return np.arange(len(df))

        # Use source MAC as primary device identifier
        src_mac_col = None
        for col in mac_cols:
            if 'src' in col.lower() or 'source' in col.lower():
                src_mac_col = col
                break
        if not src_mac_col:
            src_mac_col = mac_cols[0]

        # Create group labels
        groups = pd.factorize(df[src_mac_col].astype(str))[0]

        logger.info(f"Found {len(np.unique(groups))} unique devices for grouping")

        return groups

    def split_by_device(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by device to avoid data leakage.

        Ensures all flows from a single device are in the same split.

        Args:
            df: Input DataFrame with 'is_camera' label
            test_size: Fraction for test set
            val_size: Fraction for validation set (from training portion)
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        groups = self.get_device_groups(df)

        # First split: train+val vs test
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss.split(df, groups=groups))

        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]
        train_val_groups = groups[train_val_idx]

        # Second split: train vs val
        if val_size > 0:
            adjusted_val_size = val_size / (1 - test_size)
            gss2 = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=random_state)
            train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_groups))

            train_df = train_val_df.iloc[train_idx]
            val_df = train_val_df.iloc[val_idx]
        else:
            train_df = train_val_df
            val_df = pd.DataFrame()

        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Log class distribution in each split
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            if len(split_df) > 0 and 'is_camera' in split_df.columns:
                camera_pct = 100 * split_df['is_camera'].mean()
                logger.info(f"  {name} camera %: {camera_pct:.2f}%")

        return train_df, val_df, test_df

    def handle_imbalance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "class_weight"
    ) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
        """
        Handle class imbalance.

        Args:
            X: Feature matrix
            y: Labels
            method: "smote", "undersample", "class_weight", or "none"

        Returns:
            Tuple of (X_resampled, y_resampled, class_weights)
            class_weights is None unless method="class_weight"
        """
        class_weights = None

        if method == "smote":
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            logger.info(f"After SMOTE: {len(y)} samples, {y.sum()} positive")

        elif method == "undersample":
            rus = RandomUnderSampler(random_state=42)
            X, y = rus.fit_resample(X, y)
            logger.info(f"After undersampling: {len(y)} samples")

        elif method == "class_weight":
            # Compute balanced class weights
            n_samples = len(y)
            n_positive = y.sum()
            n_negative = n_samples - n_positive

            if n_positive > 0 and n_negative > 0:
                class_weights = {
                    0: n_samples / (2 * n_negative),
                    1: n_samples / (2 * n_positive)
                }
                logger.info(f"Class weights: {class_weights}")

        return X, y, class_weights

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        fit_scaler: bool = True
    ) -> np.ndarray:
        """
        Prepare feature matrix from DataFrame.

        Args:
            df: Input DataFrame
            feature_cols: Columns to use as features. If None, uses all numeric.
            fit_scaler: Whether to fit the scaler (True for training)

        Returns:
            Scaled feature matrix
        """
        if feature_cols is None:
            feature_cols = self.numeric_columns

        if not feature_cols:
            raise ValueError("No feature columns specified or identified")

        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        if len(available_cols) < len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            logger.warning(f"Missing feature columns: {missing}")

        self.feature_columns = available_cols
        X = df[available_cols].values.astype(np.float32)

        # Handle any remaining NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X


# =============================================================================
# TRACK B: IQ Signal Preprocessing
# =============================================================================

class IQPreprocessor:
    """
    Preprocessor for IQ signal data.

    Handles:
    - Normalization (per-sample or global)
    - Segmentation into fixed-length windows
    - Train/test splitting
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize IQ preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sample_length = config.get('sample_length', 1024) if config else 1024
        self.global_mean: Optional[np.ndarray] = None
        self.global_std: Optional[np.ndarray] = None

    def normalize(
        self,
        X: np.ndarray,
        method: str = "per_sample",
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize IQ data.

        Args:
            X: IQ data of shape (N, 2, L) where L is sample length
            method: "per_sample" or "global"
            fit: Whether to fit normalization params (True for training)

        Returns:
            Normalized data
        """
        X = X.copy().astype(np.float32)

        if method == "per_sample":
            # Normalize each sample independently
            for i in range(len(X)):
                mean = X[i].mean()
                std = X[i].std() + 1e-8
                X[i] = (X[i] - mean) / std

        elif method == "global":
            if fit:
                self.global_mean = X.mean(axis=(0, 2), keepdims=True)
                self.global_std = X.std(axis=(0, 2), keepdims=True) + 1e-8

            if self.global_mean is not None:
                X = (X - self.global_mean) / self.global_std

        return X

    def segment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        segment_length: Optional[int] = None,
        overlap: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment IQ data into fixed-length windows.

        Args:
            X: IQ data of shape (N, 2, L)
            y: Labels of shape (N,)
            segment_length: Length of each segment
            overlap: Overlap fraction between segments (0.0 to 0.5)

        Returns:
            Tuple of (segmented_X, segmented_y)
        """
        if segment_length is None:
            segment_length = self.sample_length

        current_length = X.shape[2]
        if current_length == segment_length:
            return X, y

        step = int(segment_length * (1 - overlap))
        X_segments = []
        y_segments = []

        for i in range(len(X)):
            for start in range(0, current_length - segment_length + 1, step):
                segment = X[i, :, start:start + segment_length]
                X_segments.append(segment)
                y_segments.append(y[i])

        return np.array(X_segments), np.array(y_segments)

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split IQ data into train/val/test sets.

        Args:
            X: Feature data
            y: Labels
            test_size: Test set fraction
            val_size: Validation set fraction
            stratify: Whether to stratify by label
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        stratify_y = y if stratify else None

        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=stratify_y, random_state=random_state
        )

        # Second split: train vs val
        if val_size > 0:
            adjusted_val_size = val_size / (1 - test_size)
            stratify_train = y_train_val if stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=adjusted_val_size,
                stratify=stratify_train, random_state=random_state
            )
        else:
            X_train, y_train = X_train_val, y_train_val
            X_val, y_val = np.array([]), np.array([])

        logger.info(f"IQ split sizes - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test
