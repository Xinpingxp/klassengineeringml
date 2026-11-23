"""
Data Loader Module
==================
Handles loading data for both Track A (WiFi traffic) and Track B (IQ signals).

Track A: UNSW IoT Traffic Dataset (Flow CSVs)
Track B: RadioML 2016.10a + custom SDR captures (scaffold)
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# =============================================================================
# TRACK A: UNSW IoT Traffic Dataset Loader
# =============================================================================

class UNSWIoTLoader:
    """
    Loader for UNSW IoT Traffic Dataset (2018).

    Dataset: https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b94b

    The dataset contains network traffic from 27 IoT devices captured over
    several weeks. We use the flow-level CSVs (not the raw PCAPs).

    Camera devices to identify:
    - Netatmo Welcome
    - TP-Link Day Night Cloud camera
    - Samsung SmartCam
    - Dropcam
    - Insteon Camera
    - Withings Smart Baby Monitor
    - Belkin Netcam
    """

    # Known camera device name patterns (case-insensitive matching)
    CAMERA_DEVICE_PATTERNS = [
        "netatmo welcome",
        "tp-link day night cloud camera",
        "samsung smartcam",
        "dropcam",
        "insteon camera",
        "withings smart baby monitor",
        "belkin netcam",
        "belkin wemo netcam",  # Alternative name
    ]

    CAMERA_FILE_PATTERNS = (
        "nestdropcam",
        "belkincamera",
        "netatmowelcome",
        "tplinkcamera",
        "canarycamera",
        "withingsbabymonitor",
        "samsungcamera",
        "augustdoorbell",
        "ringdoorbell",
    )

    def __init__(self, data_dir: str, config: Optional[dict] = None):
        """
        Initialize the UNSW IoT loader.

        Args:
            data_dir: Path to the raw data directory containing UNSW files
            config: Optional configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}
        self.device_mac_mapping: Dict[str, str] = {}  # device_name -> MAC
        self.camera_macs: set = set()
        self.flows_df: Optional[pd.DataFrame] = None

    def discover_files(self) -> Dict[str, List[Path]]:
        """
        Discover available data files in the data directory.

        Returns:
            Dictionary with file types and their paths
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Please download the UNSW IoT dataset from:\n"
                f"https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b94b\n"
                f"and extract it to {self.data_dir}"
            )

        files = {
            "flow_csvs": [],
            "device_lists": [],
            "other": []
        }

        for f in self.data_dir.rglob("*"):
            if f.is_file():
                name_lower = f.name.lower()
                if f.suffix.lower() == ".csv":
                    # Check if it's a device list or flow data
                    if any(x in name_lower for x in ["device", "mac", "list"]):
                        files["device_lists"].append(f)
                    else:
                        files["flow_csvs"].append(f)
                elif f.suffix.lower() in [".txt", ".md"]:
                    if any(x in name_lower for x in ["device", "mac", "list"]):
                        files["device_lists"].append(f)
                    else:
                        files["other"].append(f)
                else:
                    files["other"].append(f)

        logger.info(f"Discovered files in {self.data_dir}:")
        logger.info(f"  Flow CSVs: {len(files['flow_csvs'])}")
        logger.info(f"  Device lists: {len(files['device_lists'])}")
        logger.info(f"  Other files: {len(files['other'])}")

        return files

    def load_device_mapping(self, device_file: Optional[Path] = None) -> Dict[str, str]:
        """
        Load device name to MAC address mapping.

        The UNSW dataset typically includes a device list file that maps
        device names to their MAC addresses.

        Args:
            device_file: Path to the device list file. If None, will search
                        for it in the data directory.

        Returns:
            Dictionary mapping device names to MAC addresses
        """
        if device_file is None:
            # Try to find device list file
            files = self.discover_files()
            if not files["device_lists"]:
                logger.warning(
                    "No device list file found. Will attempt to infer device "
                    "mappings from flow data or you may need to provide one."
                )
                return {}
            device_file = files["device_lists"][0]
            logger.info(f"Using device list file: {device_file}")

        device_file = Path(device_file)
        if not device_file.exists():
            raise FileNotFoundError(f"Device list file not found: {device_file}")

        # Parse the device file - format varies, try common formats
        mapping = {}

        if device_file.suffix.lower() == ".csv":
            df = pd.read_csv(device_file)
            # Try to identify MAC and device name columns
            cols_lower = {c.lower(): c for c in df.columns}

            mac_col = None
            name_col = None

            for key in ["mac", "mac_address", "macaddress", "mac address"]:
                if key in cols_lower:
                    mac_col = cols_lower[key]
                    break

            for key in ["device", "device_name", "devicename", "name", "device name"]:
                if key in cols_lower:
                    name_col = cols_lower[key]
                    break

            if mac_col and name_col:
                for _, row in df.iterrows():
                    mac = str(row[mac_col]).strip().lower()
                    name = str(row[name_col]).strip()
                    mapping[name] = mac
            else:
                logger.warning(f"Could not identify MAC/device columns in {device_file}")
                logger.info(f"Available columns: {list(df.columns)}")
        else:
            # Try to parse as text file (various formats)
            with open(device_file, 'r') as f:
                content = f.read()

            # Try comma/tab separated
            for line in content.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Try different separators
                for sep in [',', '\t', '|', ';']:
                    if sep in line:
                        parts = [p.strip() for p in line.split(sep)]
                        if len(parts) >= 2:
                            # Identify which part is MAC (contains colons)
                            mac_part = None
                            name_part = None
                            for p in parts:
                                if ':' in p and len(p) == 17:  # MAC format
                                    mac_part = p.lower()
                                elif not mac_part:
                                    name_part = p
                            if mac_part and name_part:
                                mapping[name_part] = mac_part
                        break

        self.device_mac_mapping = mapping
        logger.info(f"Loaded {len(mapping)} device-MAC mappings")

        return mapping

    def identify_camera_macs(self) -> set:
        """
        Identify MAC addresses belonging to camera devices.

        Returns:
            Set of MAC addresses for camera devices
        """
        if not self.device_mac_mapping:
            logger.warning("No device mapping loaded. Call load_device_mapping() first.")
            return set()

        camera_macs = set()
        camera_devices_found = []

        for device_name, mac in self.device_mac_mapping.items():
            name_lower = device_name.lower()
            for pattern in self.CAMERA_DEVICE_PATTERNS:
                if pattern in name_lower or name_lower in pattern:
                    camera_macs.add(mac.lower())
                    camera_devices_found.append((device_name, mac))
                    break

        self.camera_macs = camera_macs

        # Print the mapping for verification
        print("\n" + "="*60)
        print("CAMERA DEVICE MAPPING (for verification)")
        print("="*60)
        if camera_devices_found:
            for name, mac in sorted(camera_devices_found):
                print(f"  {name}: {mac}")
        else:
            print("  No camera devices found in device mapping!")
            print("  Available devices:")
            for name, mac in sorted(self.device_mac_mapping.items()):
                print(f"    {name}: {mac}")
        print("="*60 + "\n")

        logger.info(f"Identified {len(camera_macs)} camera MACs out of "
                   f"{len(self.device_mac_mapping)} total devices")

        return camera_macs

    def load_flow_csvs(self, max_files: Optional[int] = None) -> pd.DataFrame:
        """
        Load flow CSV files into a single DataFrame.

        Args:
            max_files: Maximum number of CSV files to load (for testing)

        Returns:
            DataFrame with all flow data
        """
        files = self.discover_files()
        flow_files = files["flow_csvs"]

        if not flow_files:
            raise FileNotFoundError(
                f"No flow CSV files found in {self.data_dir}\n"
                "Please ensure the UNSW IoT flow CSVs are extracted there."
            )

        if max_files:
            flow_files = flow_files[:max_files]

        logger.info(f"Loading {len(flow_files)} flow CSV files...")

        dfs = []
        for f in flow_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                df['_source_file'] = f.name
                dfs.append(df)
                logger.debug(f"Loaded {len(df)} rows from {f.name}")
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        if not dfs:
            raise ValueError("No CSV files could be loaded successfully")

        self.flows_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(self.flows_df)} total flow records")
        logger.info(f"Columns: {list(self.flows_df.columns)}")

        return self.flows_df

    def create_labels(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create binary labels: 1 = camera device, 0 = non-camera.

        The labeling is based on source or destination MAC addresses matching
        known camera device MACs.

        Args:
            df: DataFrame with flow data. If None, uses self.flows_df

        Returns:
            DataFrame with 'is_camera' label column added
        """
        if df is None:
            df = self.flows_df
        if df is None:
            raise ValueError("No flow data loaded. Call load_flow_csvs() first.")

        df = df.copy()
        if self.camera_macs:
            mac_cols = [
                col for col in df.columns
                if any(x in col.lower() for x in ['mac', 'hw_addr', 'hardware'])
            ]

            if not mac_cols:
                raise ValueError("Camera MACs were found, but no MAC columns exist in flow data.")

            logger.info(f"Using MAC columns for labeling: {mac_cols}")
            df['is_camera'] = df.apply(
                lambda row: int(any(
                    str(row[col]).lower().strip() in self.camera_macs
                    for col in mac_cols
                )),
                axis=1
            )
        else:
            if '_source_file' not in df.columns:
                raise ValueError(
                    "No camera MAC mapping or per-device source filenames are available for labeling."
                )

            source_names = (
                df['_source_file'].astype(str).str.lower()
                .str.replace(r'[^a-z0-9]', '', regex=True)
            )
            camera_pattern = '|'.join(self.CAMERA_FILE_PATTERNS)
            camera_mask = source_names.str.contains(camera_pattern, regex=True)
            if not camera_mask.any():
                raise ValueError(
                    "No camera MAC mapping or recognized camera device names in flow filenames."
                )
            df['is_camera'] = camera_mask.astype(int)

        if '_source_file' in df.columns:
            df['device_id'] = df['_source_file'].astype(str)

        # Print label distribution
        camera_count = df['is_camera'].sum()
        non_camera_count = len(df) - camera_count

        print("\n" + "="*60)
        print("LABEL DISTRIBUTION")
        print("="*60)
        print(f"  Camera flows (label=1):     {camera_count:,} ({100*camera_count/len(df):.2f}%)")
        print(f"  Non-camera flows (label=0): {non_camera_count:,} ({100*non_camera_count/len(df):.2f}%)")
        print(f"  Total flows:                {len(df):,}")
        print(f"  Class imbalance ratio:      1:{non_camera_count/max(camera_count,1):.1f}")
        print("="*60 + "\n")

        self.flows_df = df
        return df

    def get_labeled_data(self, max_files: Optional[int] = None) -> pd.DataFrame:
        """
        Convenience method to load data and create labels in one call.

        Args:
            max_files: Maximum number of CSV files to load

        Returns:
            DataFrame with flow data and labels
        """
        self.load_device_mapping()
        self.identify_camera_macs()
        self.load_flow_csvs(max_files=max_files)
        return self.create_labels()


# =============================================================================
# TRACK B: IQ Signal Data Loader (Scaffold)
# =============================================================================

class IQSignalLoader:
    """
    Loader for IQ signal data.

    Supports:
    1. RadioML 2016.10a dataset (for pipeline testing)
    2. Custom SDR captures (TODO: implement when data available)

    The loader is designed to be extensible for different IQ data sources
    while providing a consistent interface to the model.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the IQ signal loader.

        Args:
            config: Configuration dictionary with paths and settings
        """
        self.config = config or {}
        self.data: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.label_names: Optional[List[str]] = None

    def load_radioml(self, pkl_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load RadioML 2016.10a dataset from pickle file.

        The RadioML dataset contains modulated RF signals with different
        modulation types and SNR levels. We use it to test the IQ pipeline.

        Dataset format:
        - Dictionary with keys (modulation_type, snr)
        - Values are numpy arrays of shape (num_samples, 2, 128)
        - 2 channels = I and Q

        Args:
            pkl_path: Path to RML2016.10a_dict.pkl

        Returns:
            Tuple of (data, labels, label_names)
            - data: np.ndarray of shape (N, 2, 128)
            - labels: np.ndarray of shape (N,) with integer labels
            - label_names: List of modulation type names
        """
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"RadioML dataset not found: {pkl_path}\n"
                "Please download RadioML 2016.10a from:\n"
                "  https://www.deepsig.ai/datasets\n"
                "  or search 'RadioML 2016.10a' on Kaggle"
            )

        logger.info(f"Loading RadioML dataset from {pkl_path}")

        with open(pkl_path, 'rb') as f:
            # RadioML uses Python 2 pickle format
            try:
                data_dict = pickle.load(f, encoding='latin1')
            except TypeError:
                data_dict = pickle.load(f)

        # Extract unique modulation types (ignoring SNR for now)
        mod_types = sorted(set(key[0] for key in data_dict.keys()))
        logger.info(f"Found {len(mod_types)} modulation types: {mod_types}")

        # Build data arrays
        X_list = []
        y_list = []

        for (mod_type, snr), samples in data_dict.items():
            mod_idx = mod_types.index(mod_type)
            X_list.append(samples)
            y_list.extend([mod_idx] * len(samples))

        self.data = np.vstack(X_list)
        self.labels = np.array(y_list)
        self.label_names = mod_types

        logger.info(f"Loaded {len(self.data)} samples with shape {self.data.shape}")

        return self.data, self.labels, self.label_names

    def load_sdr_captures(self, data_path: str, labels_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load custom SDR captures for camera detection.

        TODO: Implement this when SDR data is available.

        Expected format:
        - IQ samples in binary files (.iq, .raw, .bin) or numpy (.npy)
        - Labels file (CSV) mapping filenames to labels:
            filename,label
            capture_001.iq,camera_on
            capture_002.iq,camera_off

        Args:
            data_path: Directory containing IQ capture files
            labels_file: CSV file with filename,label columns

        Returns:
            Tuple of (data, labels)
        """
        # TODO: Implement SDR data loading
        # This is where your self-collected SDR data will plug in

        data_path = Path(data_path)
        labels_file = Path(labels_file)

        if not data_path.exists():
            raise FileNotFoundError(
                f"SDR data directory not found: {data_path}\n"
                "TODO: Add your SDR captures here with a labels.csv file.\n"
                "Expected format:\n"
                "  - IQ files: .iq, .raw, .bin, or .npy format\n"
                "  - Labels: CSV with columns 'filename' and 'label'\n"
                "  - Labels should be 'camera_on' or 'camera_off'"
            )

        if not labels_file.exists():
            raise FileNotFoundError(
                f"Labels file not found: {labels_file}\n"
                "Please create a labels.csv with columns: filename,label"
            )

        # Load labels
        labels_df = pd.read_csv(labels_file)
        if not all(col in labels_df.columns for col in ['filename', 'label']):
            raise ValueError("Labels file must have 'filename' and 'label' columns")

        # TODO: Load IQ files based on extension
        # Example implementation:
        """
        X_list = []
        y_list = []

        for _, row in labels_df.iterrows():
            filepath = data_path / row['filename']

            # Load based on file extension
            if filepath.suffix == '.npy':
                iq_data = np.load(filepath)
            elif filepath.suffix in ['.iq', '.raw', '.bin']:
                # Assume interleaved float32 I/Q samples
                raw = np.fromfile(filepath, dtype=np.float32)
                iq_data = raw.reshape(-1, 2).T  # Shape: (2, N)

            # Segment into fixed-length chunks
            sample_length = self.config.get('sample_length', 1024)
            for i in range(0, len(iq_data[0]) - sample_length, sample_length):
                segment = iq_data[:, i:i+sample_length]
                X_list.append(segment)
                y_list.append(1 if row['label'] == 'camera_on' else 0)

        self.data = np.array(X_list)
        self.labels = np.array(y_list)
        """

        raise NotImplementedError(
            "SDR data loading not yet implemented. "
            "TODO: Complete this method when SDR captures are available."
        )

    def get_data_for_binary_classification(
        self,
        source: str = "radioml"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get IQ data prepared for binary camera classification.

        For RadioML (testing): converts multi-class to binary (for pipeline testing)
        For SDR captures: returns actual camera_on/camera_off labels

        Args:
            source: "radioml" or "sdr"

        Returns:
            Tuple of (X, y) where X has shape (N, 2, sample_length)
            and y is binary labels
        """
        if source == "radioml":
            if self.data is None:
                raise ValueError("No data loaded. Call load_radioml() first.")
            # For pipeline testing, create artificial binary labels
            # (This is just for testing the pipeline structure)
            y_binary = (self.labels % 2).astype(int)  # Arbitrary binary split
            logger.warning(
                "Using RadioML with artificial binary labels for pipeline testing. "
                "Replace with actual SDR camera data for real classification."
            )
            return self.data, y_binary

        elif source == "sdr":
            if self.data is None:
                raise ValueError("No SDR data loaded. Call load_sdr_captures() first.")
            return self.data, self.labels

        else:
            raise ValueError(f"Unknown source: {source}. Use 'radioml' or 'sdr'.")


# =============================================================================
# Utility Functions
# =============================================================================

def check_data_availability(config_path: str = "configs/config.yaml") -> dict:
    """
    Check which datasets are available and provide download instructions.

    Returns:
        Dictionary with dataset availability status
    """
    config = load_config(config_path)
    raw_dir = Path(config['paths']['data_raw'])

    status = {
        "unsw_iot": {"available": False, "path": None, "message": ""},
        "radioml": {"available": False, "path": None, "message": ""},
        "sdr_captures": {"available": False, "path": None, "message": ""}
    }

    # Check UNSW IoT
    unsw_files = list(raw_dir.glob("**/*.csv"))
    if unsw_files:
        status["unsw_iot"]["available"] = True
        status["unsw_iot"]["path"] = str(raw_dir)
        status["unsw_iot"]["message"] = f"Found {len(unsw_files)} CSV files"
    else:
        status["unsw_iot"]["message"] = (
            "Not found. Download from:\n"
            "  https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b94b\n"
            f"  Extract flow CSVs to: {raw_dir}"
        )

    # Check RadioML
    radioml_path = raw_dir / "RML2016.10a_dict.pkl"
    if radioml_path.exists():
        status["radioml"]["available"] = True
        status["radioml"]["path"] = str(radioml_path)
        status["radioml"]["message"] = "Found"
    else:
        status["radioml"]["message"] = (
            "Not found. Download from:\n"
            "  https://www.deepsig.ai/datasets\n"
            "  or search 'RadioML 2016.10a' on Kaggle\n"
            f"  Save as: {radioml_path}"
        )

    # Check SDR captures
    sdr_path = raw_dir / "sdr_captures"
    if sdr_path.exists() and (sdr_path / "labels.csv").exists():
        status["sdr_captures"]["available"] = True
        status["sdr_captures"]["path"] = str(sdr_path)
        status["sdr_captures"]["message"] = "Found"
    else:
        status["sdr_captures"]["message"] = (
            f"Not found. TODO: Add your SDR captures to:\n"
            f"  {sdr_path}/\n"
            f"  with a labels.csv file"
        )

    return status


if __name__ == "__main__":
    # Test data loading
    print("\n" + "="*60)
    print("RF Camera Detection - Data Loader Test")
    print("="*60)

    # Check data availability
    print("\nChecking data availability...")
    status = check_data_availability()

    for dataset, info in status.items():
        print(f"\n{dataset}:")
        print(f"  Available: {info['available']}")
        print(f"  {info['message']}")

    # Try to load UNSW data if available
    if status["unsw_iot"]["available"]:
        print("\n" + "-"*60)
        print("Loading UNSW IoT dataset...")
        print("-"*60)

        config = load_config()
        set_random_seed(config['random_seed'])

        loader = UNSWIoTLoader(config['paths']['data_raw'], config)

        try:
            df = loader.get_labeled_data(max_files=5)  # Load subset for testing
            print(f"\nSample of loaded data:")
            print(df.head())
        except Exception as e:
            print(f"Error loading data: {e}")
