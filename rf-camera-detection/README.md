# RF-Based Camera Presence Detection

Binary classification pipeline for detecting camera presence in the 2.4GHz space.

## Project Overview

This capstone project compares two approaches for detecting cameras via RF analysis:

- **Track A (WiFi Traffic-Based)**: Analyzes network flow statistics to identify camera traffic patterns
- **Track B (Raw IQ Signal-Based)**: Uses SDR captures to detect camera transmissions directly *(scaffold)*

## Repository Structure

```
rf-camera-detection/
├── configs/
│   └── config.yaml          # Hyperparameters, paths, settings
├── data/
│   ├── raw/                  # Downloaded datasets (gitignored)
│   └── processed/            # Processed data (gitignored)
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis
├── outputs/
│   ├── models/               # Trained model files
│   ├── plots/                # Visualizations
│   └── results/              # Metrics, comparisons
├── src/
│   ├── data_loader.py        # Dataset loaders (UNSW, RadioML, SDR)
│   ├── preprocess.py         # Data cleaning, splitting
│   ├── features.py           # Feature extraction
│   ├── models.py             # ML/DL model definitions
│   ├── train.py              # Training pipeline
│   └── evaluate.py           # Metrics and visualization
├── requirements.txt
└── README.md
```

## Setup

### 1. Create Environment

```bash
cd rf-camera-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

#### Track A: UNSW IoT Traffic Dataset (Required)

1. Go to: https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b94b
2. Download the **flow CSV files** (not the 27GB PCAPs)
3. Extract to `data/raw/`

The dataset should include:
- Flow CSV files with network traffic statistics
- A device list file mapping MAC addresses to device names

#### Track B: RadioML 2016.10a (Optional - for pipeline testing)

1. Download from: https://www.deepsig.ai/datasets
   - Or search "RadioML 2016.10a" on Kaggle
2. Save as `data/raw/RML2016.10a_dict.pkl`

#### Track B: Custom SDR Captures (TODO)

When you collect SDR data:
1. Save IQ files to `data/raw/sdr_captures/`
2. Create `data/raw/sdr_captures/labels.csv`:
   ```csv
   filename,label
   capture_001.iq,camera_on
   capture_002.iq,camera_off
   ```

## Usage

### Quick Start

```bash
# Run Track A training (WiFi traffic analysis)
python src/train.py --config configs/config.yaml --track a

# Run with custom config
python src/train.py --config path/to/config.yaml
```

### Check Data Availability

```bash
python src/data_loader.py
```

### Exploratory Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

## Track A: WiFi Traffic Detection

### Dataset
UNSW IoT Traffic dataset with 27 IoT devices. We label flows as:
- **Camera (1)**: Netatmo Welcome, TP-Link Day Night Cloud camera, Samsung SmartCam, Dropcam, Insteon Camera, Withings Smart Baby Monitor, Belkin Netcam
- **Non-Camera (0)**: All other devices

### Features
- Packet statistics (count, size distributions)
- Byte statistics (total, mean, variance)
- Timing features (duration, inter-arrival times)
- Protocol and port information

### Models
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. CNN-LSTM *(optional, requires packet-level data)*

### Data Splitting
Split by **device** (not random rows) to prevent data leakage.

## Track B: IQ Signal Detection (Scaffold)

### Architecture
CNN + LSTM model for raw IQ signal classification:
- Input: (batch, 2, sample_length) - I and Q channels
- 1D CNN layers for feature extraction
- Bidirectional LSTM for temporal patterns
- Attention pooling
- Binary classifier

### Adding Your SDR Data

1. Collect IQ captures with camera on/off
2. Save to `data/raw/sdr_captures/`
3. Create `labels.csv`
4. Update `configs/config.yaml`:
   ```yaml
   track_b:
     sdr_capture:
       data_path: "data/raw/sdr_captures"
       labels_file: "data/raw/sdr_captures/labels.csv"
   ```
5. Complete `load_sdr_captures()` in `src/data_loader.py`

## Configuration

Key settings in `configs/config.yaml`:

```yaml
random_seed: 42  # Reproducibility

track_a:
  split:
    test_size: 0.2
    split_by: "device"  # Prevents leakage
  imbalance:
    method: "class_weight"  # Or "smote", "undersample"
  models:
    random_forest:
      n_estimators: 100
      max_depth: 20

track_b:
  iq_model:
    sample_length: 1024
    cnn_filters: [64, 128, 256]
    lstm_hidden_size: 128
```

## Outputs

After training:
```
outputs/
├── models/
│   ├── logisticregression.pkl
│   ├── randomforest.pkl
│   ├── xgboost.pkl
│   ├── preprocessing.pkl
│   └── iq_model_best.pt (Track B)
├── plots/
│   ├── *_confusion_matrix.png
│   ├── *_roc_curve.png
│   └── model_comparison.png
└── results/
    ├── track_a_results.csv
    └── track_a_results.json
```

## Camera Device MAC Mapping

The pipeline automatically identifies camera devices from the UNSW dataset's device list. Expected camera devices:

| Device | Description |
|--------|-------------|
| Netatmo Welcome | Indoor security camera |
| TP-Link Day Night Cloud camera | IP camera |
| Samsung SmartCam | WiFi camera |
| Dropcam | Cloud-connected camera |
| Insteon Camera | Home automation camera |
| Withings Smart Baby Monitor | Baby monitor with camera |
| Belkin Netcam | WiFi camera |

The actual MAC addresses are extracted from the dataset's device list file.

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Average Precision
- Confusion matrices
- Model comparison tables

## Reproducibility

- Global random seed set in config
- Device-based splitting (not random)
- All preprocessing artifacts saved
- Model checkpoints saved

## License

For academic/research use. See dataset licenses for data usage terms.
