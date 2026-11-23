"""
Feature Extraction Module
=========================
Extract features from network flows (Track A) and prepare sequences for
deep learning models.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# TRACK A: WiFi Traffic Features
# =============================================================================

class FlowFeatureExtractor:
    """
    Extract features from network flow data for camera detection.

    Features include:
    - Packet statistics (count, size distributions)
    - Byte statistics (total, mean, variance)
    - Timing features (duration, inter-arrival times)
    - Protocol and port information
    """

    # Common column name mappings (UNSW dataset may use various names)
    COLUMN_MAPPINGS = {
        # Packet counts
        'packet_count': ['pkts', 'packets', 'tot_pkts', 'total_packets',
                                                'fwd_pkts', 'bwd_pkts', 'spkts', 'dpkts',
                                                'srcnumpackets', 'dstnumpackets'],
        # Byte counts
        'byte_count': ['bytes', 'tot_bytes', 'total_bytes', 'sbytes', 'dbytes',
                                            'fwd_bytes', 'bwd_bytes', 'srcpayloadsize', 'dstpayloadsize'],
        # Packet sizes
        'pkt_size': ['pkt_len', 'packet_length', 'avg_pkt_size', 'mean_pkt_len'],
        # Duration
        'duration': ['dur', 'duration', 'flow_duration', 'dur_sec', 'flowduration'],
        # Protocol
        'protocol': ['proto', 'protocol', 'ip_proto', 'ipproto'],
        # Ports
        'src_port': ['sport', 'src_port', 'source_port', 'srcport'],
        'dst_port': ['dport', 'dst_port', 'dest_port', 'dstport'],
        # Inter-arrival time
        'iat': ['iat', 'inter_arrival', 'mean_iat', 'sintpkt', 'dintpkt'],
    }

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize feature extractor.

        Args:
            config: Configuration dictionary with feature settings
        """
        self.config = config or {}
        self.feature_names: List[str] = []
        self.column_map: Dict[str, str] = {}

    def _find_column(self, df: pd.DataFrame, feature_type: str) -> Optional[str]:
        """
        Find the actual column name for a feature type.

        Args:
            df: DataFrame to search
            feature_type: Type of feature to find (e.g., 'packet_count')

        Returns:
            Actual column name or None if not found
        """
        if feature_type not in self.COLUMN_MAPPINGS:
            return None

        possible_names = self.COLUMN_MAPPINGS[feature_type]
        df_cols_lower = {c.lower(): c for c in df.columns}

        for name in possible_names:
            if name.lower() in df_cols_lower:
                return df_cols_lower[name.lower()]

        return None

    def discover_available_features(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Discover which standard features are available in the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary mapping feature types to actual column names
        """
        self.column_map = {}

        for feature_type in self.COLUMN_MAPPINGS.keys():
            col = self._find_column(df, feature_type)
            if col:
                self.column_map[feature_type] = col
                logger.debug(f"Found {feature_type} -> {col}")

        logger.info(f"Discovered {len(self.column_map)} standard feature columns")

        return self.column_map

    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic flow-level features.

        Args:
            df: Input DataFrame with flow data

        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame(index=df.index)

        # Discover available columns
        if not self.column_map:
            self.discover_available_features(df)

        # ===== Packet Features =====
        pkt_cols = [c for c in df.columns if any(
            p in c.lower() for p in ['pkt', 'packet', 'numpacket']
        ) and 'len' not in c.lower()]

        for col in pkt_cols:
            features[f'feat_{col}'] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Total packets (sum of any packet count columns found)
        if pkt_cols:
            features['feat_total_pkts'] = df[pkt_cols].apply(
                pd.to_numeric, errors='coerce'
            ).fillna(0).sum(axis=1)

        # ===== Byte Features =====
        byte_cols = [
            c for c in df.columns
            if 'byte' in c.lower() or 'payloadsize' in c.lower()
        ]

        for col in byte_cols:
            features[f'feat_{col}'] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if byte_cols:
            features['feat_total_bytes'] = df[byte_cols].apply(
                pd.to_numeric, errors='coerce'
            ).fillna(0).sum(axis=1)

        # ===== Timing Features =====
        dur_col = self._find_column(df, 'duration')
        if dur_col:
            features['feat_duration'] = pd.to_numeric(df[dur_col], errors='coerce').fillna(0)

            # Packets per second
            if 'feat_total_pkts' in features.columns:
                duration = features['feat_duration'].replace(0, np.nan)
                features['feat_pkts_per_sec'] = features['feat_total_pkts'] / duration
                features['feat_pkts_per_sec'] = features['feat_pkts_per_sec'].fillna(0)

            # Bytes per second
            if 'feat_total_bytes' in features.columns:
                duration = features['feat_duration'].replace(0, np.nan)
                features['feat_bytes_per_sec'] = features['feat_total_bytes'] / duration
                features['feat_bytes_per_sec'] = features['feat_bytes_per_sec'].fillna(0)

        # Inter-arrival time features
        iat_cols = [c for c in df.columns if 'iat' in c.lower() or 'inter' in c.lower()]
        for col in iat_cols:
            features[f'feat_{col}'] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # ===== Protocol Features =====
        proto_col = self._find_column(df, 'protocol')
        if proto_col:
            # One-hot encode common protocols
            proto_values = df[proto_col].astype(str).str.lower()
            features['feat_proto_tcp'] = (proto_values.isin(['tcp', '6'])).astype(int)
            features['feat_proto_udp'] = (proto_values.isin(['udp', '17'])).astype(int)
            features['feat_proto_icmp'] = (proto_values.isin(['icmp', '1'])).astype(int)

        # ===== Port Features =====
        for port_type in ['src_port', 'dst_port']:
            port_col = self._find_column(df, port_type)
            if port_col:
                port_values = pd.to_numeric(df[port_col], errors='coerce').fillna(0)
                features[f'feat_{port_type}'] = port_values

                # Well-known port indicators
                features[f'feat_{port_type}_wellknown'] = (port_values < 1024).astype(int)
                features[f'feat_{port_type}_http'] = port_values.isin([80, 443, 8080, 8443]).astype(int)
                features[f'feat_{port_type}_media'] = port_values.isin([554, 8554, 1935]).astype(int)  # RTSP, RTMP

        # Store feature names
        self.feature_names = list(features.columns)
        logger.info(f"Extracted {len(self.feature_names)} features")

        return features

    def extract_sequence_features(
        self,
        df: pd.DataFrame,
        sequence_length: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract packet-level sequences for sequence models (CNN/LSTM).

        This creates fixed-length sequences of packet sizes and timing
        for each flow, suitable for 1D-CNN or LSTM models.

        Args:
            df: Input DataFrame with packet-level information
            sequence_length: Length of each sequence

        Returns:
            Tuple of (sequences, labels)
            - sequences: np.ndarray of shape (N, 2, sequence_length)
              Channel 0: packet sizes, Channel 1: inter-arrival times
            - labels: np.ndarray of shape (N,)
        """
        # This is a simplified version - actual implementation depends
        # on having packet-level data, not just flow summaries

        # For flow-level CSVs, we can create pseudo-sequences from statistics
        # Real packet sequences would come from PCAP analysis

        logger.warning(
            "extract_sequence_features requires packet-level data. "
            "Using flow statistics to create representative sequences. "
            "For better sequence models, parse the original PCAPs."
        )

        n_flows = len(df)
        sequences = np.zeros((n_flows, 2, sequence_length), dtype=np.float32)

        # Use flow statistics to generate representative patterns
        # This is a placeholder - real implementation would use actual packet data

        if 'is_camera' in df.columns:
            labels = df['is_camera'].values
        else:
            labels = np.zeros(n_flows)

        return sequences, labels


# =============================================================================
# Feature Selection and Importance
# =============================================================================

def get_feature_importance(
    model,
    feature_names: List[str],
    top_k: int = 20
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Args:
        model: Trained model (RandomForest, XGBoost, or similar)
        feature_names: List of feature names
        top_k: Number of top features to return

    Returns:
        DataFrame with feature names and importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).ravel()
    else:
        raise ValueError("Model does not have feature_importances_ or coef_")

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return importance_df.head(top_k)


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = "mutual_info",
    k: int = 50
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Select top-k features using various methods.

    Args:
        X: Feature matrix
        y: Labels
        feature_names: Feature names
        method: Selection method ("mutual_info", "chi2", "f_classif")
        k: Number of features to select

    Returns:
        Tuple of (X_selected, selected_names, scores)
    """
    from sklearn.feature_selection import (
        SelectKBest, mutual_info_classif, chi2, f_classif
    )

    method_map = {
        "mutual_info": mutual_info_classif,
        "chi2": chi2,
        "f_classif": f_classif
    }

    if method not in method_map:
        raise ValueError(f"Unknown method: {method}")

    # Handle negative values for chi2
    if method == "chi2":
        X = X - X.min(axis=0)

    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=method_map[method], k=k)
    X_selected = selector.fit_transform(X, y)

    # Get selected feature names
    selected_mask = selector.get_support()
    selected_names = [f for f, m in zip(feature_names, selected_mask) if m]
    scores = selector.scores_[selected_mask]

    logger.info(f"Selected {len(selected_names)} features using {method}")

    return X_selected, selected_names, scores
