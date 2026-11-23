#!/usr/bin/env python3
"""
Training Script
===============
End-to-end training pipeline for RF camera detection.

Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --track a  # WiFi only
    python src/train.py --config configs/config.yaml --track b  # IQ only (scaffold)
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import (
    load_config, set_random_seed, UNSWIoTLoader, IQSignalLoader,
    check_data_availability
)
from preprocess import FlowPreprocessor, IQPreprocessor
from features import FlowFeatureExtractor, get_feature_importance
from models import get_model, count_parameters
from evaluate import Evaluator, compute_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# TRACK A: WiFi Traffic Training
# =============================================================================

def train_track_a(config: dict, output_dir: Path) -> Dict[str, Dict]:
    """
    Train Track A models (WiFi traffic-based camera detection).

    Args:
        config: Configuration dictionary
        output_dir: Output directory for models and results

    Returns:
        Dictionary of model results
    """
    logger.info("="*60)
    logger.info("TRACK A: WiFi Traffic-Based Camera Detection")
    logger.info("="*60)

    track_cfg = config.get('track_a', {})
    data_dir = Path(config['paths']['data_raw'])

    # Check data availability
    status = check_data_availability()
    if not status['unsw_iot']['available']:
        print("\n" + "!"*60)
        print("UNSW IoT dataset not found!")
        print(status['unsw_iot']['message'])
        print("!"*60 + "\n")
        raise FileNotFoundError("UNSW IoT dataset not available. Please download it first.")

    # ===== Load Data =====
    logger.info("Loading UNSW IoT dataset...")
    loader = UNSWIoTLoader(data_dir, config)

    try:
        df = loader.get_labeled_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # ===== Preprocess =====
    logger.info("Preprocessing data...")
    preprocessor = FlowPreprocessor(track_cfg)

    # Identify and clean
    preprocessor.identify_columns(df)
    df = preprocessor.clean_data(df)
    df = preprocessor.encode_categoricals(df)

    # ===== Extract Features =====
    logger.info("Extracting features...")
    extractor = FlowFeatureExtractor(track_cfg.get('features', {}))
    features_df = extractor.extract_basic_features(df)

    # Combine with labels
    if 'device_id' in df.columns:
        features_df['device_id'] = df['device_id'].values
    features_df['is_camera'] = df['is_camera'].values

    # ===== Split Data =====
    logger.info("Splitting data by device...")
    split_cfg = track_cfg.get('split', {})

    train_df, val_df, test_df = preprocessor.split_by_device(
        features_df,
        test_size=split_cfg.get('test_size', 0.2),
        val_size=split_cfg.get('val_size', 0.1),
        random_state=config.get('random_seed', 42)
    )

    # Prepare feature matrices
    feature_cols = [c for c in train_df.columns if c.startswith('feat_')]

    X_train = preprocessor.prepare_features(train_df, feature_cols, fit_scaler=True)
    y_train = train_df['is_camera'].values

    X_val = preprocessor.prepare_features(val_df, feature_cols, fit_scaler=False) if len(val_df) > 0 else None
    y_val = val_df['is_camera'].values if len(val_df) > 0 else None

    X_test = preprocessor.prepare_features(test_df, feature_cols, fit_scaler=False)
    y_test = test_df['is_camera'].values

    logger.info(f"Feature matrix shape: {X_train.shape}")
    logger.info(f"Training samples: {len(y_train)}, positive: {y_train.sum()}")

    # ===== Handle Class Imbalance =====
    imbalance_cfg = track_cfg.get('imbalance', {})
    imbalance_method = imbalance_cfg.get('method', 'class_weight')

    X_train_resampled, y_train_resampled, class_weights = preprocessor.handle_imbalance(
        X_train, y_train, method=imbalance_method
    )

    # Calculate scale_pos_weight for XGBoost
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    # ===== Train Models =====
    evaluator = Evaluator(str(output_dir))
    models_cfg = track_cfg.get('models', {})
    results = {}

    # Traditional ML Models
    traditional_models = [
        ('DecisionTree', 'decision_tree', models_cfg.get('decision_tree', {})),
        ('LinearSVM', 'linear_svm', models_cfg.get('linear_svm', {})),
        ('LogisticRegression', 'logistic_regression', models_cfg.get('logistic_regression', {})),
        ('RandomForest', 'random_forest', models_cfg.get('random_forest', {})),
        ('XGBoost', 'xgboost', models_cfg.get('xgboost', {})),
    ]

    for model_name, model_type, model_cfg in traditional_models:
        logger.info(f"\nTraining {model_name}...")

        # Get model with appropriate class weights
        if model_type == 'xgboost':
            model = get_model(model_type, model_cfg, scale_pos_weight=scale_pos_weight)
        else:
            model = get_model(model_type, model_cfg, class_weight=class_weights)

        # Train
        if imbalance_method in ['smote', 'undersample']:
            model.fit(X_train_resampled, y_train_resampled)
        else:
            model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = None

        # Evaluate
        metrics = evaluator.evaluate_model(model_name, y_test, y_pred, y_prob)
        results[model_name] = metrics
        evaluator.save_results("track_a_results.csv")

        # Feature importance
        try:
            importance = get_feature_importance(model, feature_cols, top_k=15)
            print(f"\nTop features for {model_name}:")
            print(importance.to_string(index=False))
        except Exception as e:
            logger.debug(f"Could not get feature importance: {e}")

        # Save model
        model_path = output_dir / "models" / f"{model_name.lower()}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {model_path}")

    # ===== Optional: CNN-LSTM Model =====
    cnn_cfg = models_cfg.get('cnn_lstm', {})
    if cnn_cfg.get('enabled', False):
        logger.info("\nTraining CNN-LSTM (sequence model)...")
        # Note: This requires packet-level sequences, not flow summaries
        # For flow CSVs, we'd need to parse PCAPs or use flow stats as pseudo-sequences
        logger.warning("CNN-LSTM requires packet-level sequences. Skipping for flow CSVs.")
        logger.warning("To enable: parse PCAPs or provide packet-level data.")

    # ===== Save Results =====
    evaluator.save_results("track_a_results.csv")

    # Save preprocessing artifacts
    artifacts_path = output_dir / "models" / "preprocessing.pkl"
    with open(artifacts_path, 'wb') as f:
        pickle.dump({
            'scaler': preprocessor.scaler,
            'label_encoders': preprocessor.label_encoders,
            'feature_columns': feature_cols,
            'camera_macs': loader.camera_macs,
            'device_mapping': loader.device_mac_mapping
        }, f)
    logger.info(f"Saved preprocessing artifacts to {artifacts_path}")

    return results


# =============================================================================
# TRACK B: IQ Signal Training (Scaffold)
# =============================================================================

def train_track_b(config: dict, output_dir: Path) -> Dict[str, Dict]:
    """
    Train Track B models (IQ signal-based camera detection).

    NOTE: This is a scaffold. Full training requires SDR capture data.

    Args:
        config: Configuration dictionary
        output_dir: Output directory

    Returns:
        Dictionary of model results
    """
    logger.info("="*60)
    logger.info("TRACK B: IQ Signal-Based Camera Detection (SCAFFOLD)")
    logger.info("="*60)

    track_cfg = config.get('track_b', {})
    iq_cfg = track_cfg.get('iq_model', {})

    # Check data availability
    status = check_data_availability()

    # ===== Try to load RadioML for pipeline testing =====
    if status['radioml']['available']:
        logger.info("Loading RadioML 2016.10a for pipeline testing...")

        loader = IQSignalLoader(track_cfg)
        try:
            X, y, label_names = loader.load_radioml(
                track_cfg.get('radioml', {}).get('dataset_path',
                    'data/raw/RML2016.10a_dict.pkl')
            )

            # For pipeline testing, use artificial binary labels
            X, y_binary = loader.get_data_for_binary_classification('radioml')

            logger.info(f"Loaded {len(X)} samples, shape: {X.shape}")

        except Exception as e:
            logger.warning(f"Could not load RadioML: {e}")
            logger.info("Skipping Track B training.")
            return {}
    else:
        logger.warning("RadioML dataset not available.")
        logger.info(status['radioml']['message'])

        # Check for SDR captures
        if status['sdr_captures']['available']:
            logger.info("Loading SDR captures...")
            # TODO: Implement when SDR data is available
            logger.error("SDR capture loading not yet implemented.")
            return {}
        else:
            logger.warning(status['sdr_captures']['message'])
            logger.info("No IQ data available. Skipping Track B training.")
            return {}

    # ===== Preprocess IQ Data =====
    preprocessor = IQPreprocessor(iq_cfg)

    # Normalize
    X = preprocessor.normalize(X, method='per_sample', fit=True)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X, y_binary,
        test_size=0.2,
        val_size=0.1,
        random_state=config.get('random_seed', 42)
    )

    # ===== Create Model =====
    model = get_model('iq_cnn_lstm', iq_cfg)
    logger.info(f"IQ Model parameters: {count_parameters(model):,}")

    # ===== Training Setup =====
    device = torch.device(config.get('device', 'cpu'))
    model = model.to(device)

    # Data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    batch_size = iq_cfg.get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=iq_cfg.get('learning_rate', 0.0001))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # ===== Training Loop =====
    epochs = iq_cfg.get('epochs', 100)
    patience = iq_cfg.get('early_stopping_patience', 15)
    best_val_loss = float('inf')
    patience_counter = 0

    logger.info("Training IQ Signal model...")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), output_dir / "models" / "iq_model_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # ===== Evaluate =====
    model.load_state_dict(torch.load(output_dir / "models" / "iq_model_best.pt"))
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y_batch.numpy())

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)

    evaluator = Evaluator(str(output_dir))
    metrics = evaluator.evaluate_model("IQ_CNN_LSTM", y_true, y_pred, y_prob)
    evaluator.save_results("track_b_results.csv")

    logger.info("\n" + "="*60)
    logger.info("TRACK B NOTE: Using RadioML with artificial binary labels")
    logger.info("for pipeline testing. Replace with actual SDR camera data")
    logger.info("for real camera detection.")
    logger.info("="*60)

    return {"IQ_CNN_LSTM": metrics}


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RF Camera Detection Training Pipeline"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--track",
        type=str,
        choices=['a', 'b', 'both'],
        default='a',
        help="Which track to train: a (WiFi), b (IQ), or both"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_random_seed(config.get('random_seed', 42))

    # Setup output directories
    output_dir = Path(config['paths']['outputs'])
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    # Check data availability first
    print("\n" + "="*60)
    print("RF Camera Detection Pipeline")
    print("="*60)

    status = check_data_availability(args.config)
    print("\nData Availability:")
    for name, info in status.items():
        symbol = "✓" if info['available'] else "✗"
        print(f"  [{symbol}] {name}")
    print()

    results = {}

    # Train Track A
    if args.track in ['a', 'both']:
        try:
            results['track_a'] = train_track_a(config, output_dir)
        except FileNotFoundError as e:
            logger.error(f"Track A failed: {e}")
            if args.track == 'a':
                sys.exit(1)

    # Train Track B
    if args.track in ['b', 'both']:
        try:
            results['track_b'] = train_track_b(config, output_dir)
        except Exception as e:
            logger.error(f"Track B failed: {e}")
            if args.track == 'b':
                sys.exit(1)

    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"  - Models: {output_dir}/models/")
    print(f"  - Plots: {output_dir}/plots/")
    print(f"  - Results: {output_dir}/results/")

    return results


if __name__ == "__main__":
    main()
