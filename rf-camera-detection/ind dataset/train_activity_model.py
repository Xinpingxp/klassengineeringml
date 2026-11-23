#!/usr/bin/env python3
"""Evaluate camera streaming-versus-idle activity on the staged v3 captures."""

import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CAMERA_MAC = "60:15:6f:2e:63:91"
WINDOW_SECONDS = 5.0
ADDRESS_COLUMNS = ["addr1_ra", "addr2_ta", "addr3", "addr4", "source", "destination"]
CSV_COLUMNS = [
    "esp_timestamp_us", "frame_type", "frame_subtype", "frame_length_bytes",
    "addr1_ra", "addr2_ta", "addr3", "addr4", "source", "destination",
    "phy_mcs",
]
FEATURE_COLUMNS = [
    "blockack_rate",
    "blockack_uplink_share",
    "payload_frame_rate",
    "payload_bytes_per_second",
    "payload_mean_size",
    "payload_std_size",
    "payload_p95_size",
    "payload_max_size",
    "payload_large_frame_share",
    "payload_uplink_share",
    "payload_iat_mean_ms",
    "payload_iat_std_ms",
]


def _normalise_addresses(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[ADDRESS_COLUMNS].astype("string").fillna("").apply(
        lambda column: column.str.strip().str.lower()
    )


def extract_session_windows(session_dir: Path) -> pd.DataFrame:
    metadata = json.loads((session_dir / "session.json").read_text())
    state = metadata.get("camera_state")
    if state not in {"streaming", "idle"}:
        return pd.DataFrame()

    frames = pd.read_csv(session_dir / "all_wifi_packets.csv", usecols=CSV_COLUMNS)
    frames["esp_timestamp_us"] = pd.to_numeric(frames["esp_timestamp_us"], errors="coerce")
    frames = frames.dropna(subset=["esp_timestamp_us"]).copy()
    if frames.empty:
        return pd.DataFrame()

    addresses = _normalise_addresses(frames)
    camera_mask = addresses.eq(CAMERA_MAC).any(axis=1)
    frames = frames.assign(
        _camera=camera_mask.to_numpy(),
        _addr1=addresses["addr1_ra"].to_numpy(),
        _addr2=addresses["addr2_ta"].to_numpy(),
        _source=addresses["source"].to_numpy(),
    )

    # Anchor windows to the full capture, not just camera frames; idle windows
    # often contain very few camera-originated records.
    capture_start = float(frames["esp_timestamp_us"].min())
    capture_end = float(frames["esp_timestamp_us"].max())
    capture_seconds = max((capture_end - capture_start) / 1e6, 0.0)
    n_windows = max(1, math.ceil(capture_seconds / WINDOW_SECONDS))
    frames["window"] = np.floor(
        ((frames["esp_timestamp_us"] - capture_start) / 1e6) / WINDOW_SECONDS
    ).astype(int)
    frames = frames[frames["window"] < n_windows]

    frame_type = pd.to_numeric(frames["frame_type"], errors="coerce")
    frame_subtype = pd.to_numeric(frames["frame_subtype"], errors="coerce")
    frame_length = pd.to_numeric(frames["frame_length_bytes"], errors="coerce").fillna(0)
    phy_mcs = pd.to_numeric(frames["phy_mcs"], errors="coerce")
    timestamps = frames["esp_timestamp_us"].to_numpy(dtype=float) / 1e6

    output = []
    for window in range(n_windows):
        in_window = frames["window"].to_numpy() == window
        camera_rows = frames.loc[in_window & frames["_camera"].to_numpy()].copy()
        camera_types = frame_type.loc[camera_rows.index]
        camera_subtypes = frame_subtype.loc[camera_rows.index]
        camera_lengths = frame_length.loc[camera_rows.index]
        camera_mcs = phy_mcs.loc[camera_rows.index]

        blockack_mask = (
            (camera_types == 1)
            & (camera_subtypes == 9)
            & (
                camera_rows["_addr1"].eq(CAMERA_MAC)
                | camera_rows["_addr2"].eq(CAMERA_MAC)
            )
        )
        blockacks = camera_rows.loc[blockack_mask]
        payload_mask = (camera_types == 2) & camera_mcs.notna()
        payload = camera_rows.loc[payload_mask]
        payload_sizes = camera_lengths.loc[payload.index].to_numpy(dtype=float)
        window_start = capture_start / 1e6 + window * WINDOW_SECONDS
        window_duration = min(WINDOW_SECONDS, max(capture_end / 1e6 - window_start, 1e-6))

        payload_times = np.sort(payload["esp_timestamp_us"].to_numpy(dtype=float) / 1e6)
        payload_iat = np.diff(payload_times) * 1000 if len(payload_times) > 1 else np.array([])
        uplink_count = int(blockacks["_addr1"].eq(CAMERA_MAC).sum())
        payload_uplink_count = int(payload["_source"].eq(CAMERA_MAC).sum())

        output.append({
            "session": session_dir.name,
            "state": state,
            "label": int(state == "streaming"),
            "window": window,
            "capture_window_seconds": window_duration,
            "blockack_rate": len(blockacks) / window_duration,
            "blockack_uplink_share": uplink_count / len(blockacks) if len(blockacks) else 0.0,
            "payload_frame_rate": len(payload) / window_duration,
            "payload_bytes_per_second": payload_sizes.sum() / window_duration,
            "payload_mean_size": payload_sizes.mean() if len(payload_sizes) else 0.0,
            "payload_std_size": payload_sizes.std() if len(payload_sizes) else 0.0,
            "payload_p95_size": np.percentile(payload_sizes, 95) if len(payload_sizes) else 0.0,
            "payload_max_size": payload_sizes.max() if len(payload_sizes) else 0.0,
            "payload_large_frame_share": np.mean(payload_sizes >= 1000) if len(payload_sizes) else 0.0,
            "payload_uplink_share": payload_uplink_count / len(payload) if len(payload) else 0.0,
            "payload_iat_mean_ms": payload_iat.mean() if len(payload_iat) else 0.0,
            "payload_iat_std_ms": payload_iat.std() if len(payload_iat) else 0.0,
            "camera_related_frames": len(camera_rows),
        })

    return pd.DataFrame(output)


def build_model() -> object:
    return make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", C=1.0, max_iter=10000, tol=1e-3, dual="auto", random_state=42),
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    session_frames = []
    for session_dir in sorted(DATA_DIR.iterdir()):
        if session_dir.is_dir() and (session_dir / "all_wifi_packets.csv").exists():
            extracted = extract_session_windows(session_dir)
            if not extracted.empty:
                session_frames.append(extracted)

    if not session_frames:
        raise RuntimeError(f"No usable streaming/idle sessions found under {DATA_DIR}")

    data = pd.concat(session_frames, ignore_index=True)
    if data["label"].nunique() != 2:
        raise RuntimeError("Both streaming and idle sessions are required for training")

    X = data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float64)
    y = data["label"].to_numpy(dtype=int)
    groups = data["session"].to_numpy()
    logo = LeaveOneGroupOut()
    out_of_fold_predictions = np.full(len(data), -1, dtype=int)
    out_of_fold_scores = np.full(len(data), np.nan, dtype=float)

    for train_indices, test_indices in logo.split(X, y, groups):
        if np.unique(y[train_indices]).size != 2:
            raise RuntimeError("A session-held-out fold has only one training class")
        model = build_model()
        model.fit(X[train_indices], y[train_indices])
        out_of_fold_predictions[test_indices] = model.predict(X[test_indices])
        out_of_fold_scores[test_indices] = model.decision_function(X[test_indices])

    if (out_of_fold_predictions < 0).any() or np.isnan(out_of_fold_scores).any():
        raise RuntimeError("Cross-validation did not generate predictions for every window")

    metrics = {
        "task": "camera streaming vs idle; not camera vs non-camera",
        "camera_mac": CAMERA_MAC,
        "window_seconds": WINDOW_SECONDS,
        "split": "leave-one-session-out; all windows from each test session held out",
        "sessions": int(data["session"].nunique()),
        "windows": int(len(data)),
        "streaming_windows": int(y.sum()),
        "idle_windows": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y, out_of_fold_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, out_of_fold_predictions)),
        "precision": float(precision_score(y, out_of_fold_predictions, zero_division=0)),
        "recall": float(recall_score(y, out_of_fold_predictions, zero_division=0)),
        "f1": float(f1_score(y, out_of_fold_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, out_of_fold_scores)),
        "average_precision": float(average_precision_score(y, out_of_fold_scores)),
        "warning": "Single camera, one network/channel; evaluates camera activity across sessions, not general camera identification.",
    }

    predictions = data[["session", "state", "label", "window", "capture_window_seconds"]].copy()
    predictions["prediction"] = out_of_fold_predictions
    predictions["decision_score"] = out_of_fold_scores
    predictions.to_csv(RESULTS_DIR / "window_predictions.csv", index=False)

    session_summary = predictions.groupby(["session", "state"], as_index=False).agg(
        windows=("label", "size"),
        correct=("prediction", lambda values: int((values == predictions.loc[values.index, "label"]).sum())),
        mean_score=("decision_score", "mean"),
    )
    session_summary["accuracy"] = session_summary["correct"] / session_summary["windows"]
    session_summary.to_csv(RESULTS_DIR / "session_results.csv", index=False)

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "metrics.csv", index=False)
    (RESULTS_DIR / "classification_report.txt").write_text(
        classification_report(y, out_of_fold_predictions, target_names=["idle", "streaming"], zero_division=0)
    )
    pd.DataFrame(
        confusion_matrix(y, out_of_fold_predictions, labels=[0, 1]),
        index=["actual_idle", "actual_streaming"],
        columns=["predicted_idle", "predicted_streaming"],
    ).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y, out_of_fold_predictions, labels=[0, 1], display_labels=["Idle", "Streaming"], ax=ax, colorbar=False
    )
    ax.set_title("Leave-one-session-out confusion matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(y, out_of_fold_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Leave-one-session-out ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y, out_of_fold_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {metrics['average_precision']:.3f}")
    ax.axhline(y.mean(), color="k", linestyle="--", label=f"Prevalence = {y.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Leave-one-session-out precision-recall")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    final_model = build_model()
    final_model.fit(X, y)
    joblib.dump({"model": final_model, "feature_columns": FEATURE_COLUMNS, "camera_mac": CAMERA_MAC}, RESULTS_DIR / "linear_svm_activity.joblib")

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved predictions, metrics, plots, and final all-window model under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()