# UCI_HAR_Model_With_label.py
"""
Generate Scalogram, Spectrogram, Kurtogram, and XYZ_Combined images
for the UCI Human Activity Recognition (HAR) dataset.

Mirrors all functionality of SisFall_Model_With_lable.py exactly.
Adapted for UCI HAR dataset folder structure and label system.

Dataset folder structure expected:
  <BASE_DIR>/
    UCI HAR Dataset/
      activity_labels.txt
      train/
        y_train.txt
        subject_train.txt
        Inertial Signals/
          total_acc_x_train.txt   (and _y_, _z_)
          body_acc_x_train.txt    (and _y_, _z_)
          body_gyro_x_train.txt   (and _y_, _z_)
      test/
        y_test.txt
        subject_test.txt
        Inertial Signals/
          total_acc_x_test.txt    (and _y_, _z_)
          body_acc_x_test.txt     (and _y_, _z_)
          body_gyro_x_test.txt    (and _y_, _z_)

Output folder structure (same pattern as SisFall):
  <BASE_DIR>/
    Generated Images/
      TotalAcc/
        WALKING/
          Scalogram/subject_01/row_0000_X.png
          Spectrogram/subject_01/row_0000_X.png
          Kurtogram/subject_01/row_0000_X.png
          XYZ_Combined/subject_01/row_0000_XYZ.png
        SITTING/ ...
        STANDING/ ...
        WALKING_UPSTAIRS/ ...
        WALKING_DOWNSTAIRS/ ...
        LAYING/ ...
      BodyAcc/ ...
      BodyGyro/ ...

Usage:
  python UCI_HAR_Model_With_label.py
  python UCI_HAR_Model_With_label.py "D:/path/to/project"
"""

import sys
from pathlib import Path
import traceback
import warnings
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from scipy import signal, stats
from PIL import Image

# ============ CONFIG ============
DEFAULT_BASE_DIR = Path(r"D:\4.1\CSE 400-A\UciHar-Model").resolve()
RAW_DATA_DIR     = 'UCI HAR Dataset'
OUTPUT_DIR       = 'Generated Images'

# Signal processing
# UCI HAR uses 50 Hz (SisFall used 200 Hz)
FS = 50.0

# UCI HAR is already pre-windowed into 128-sample (2.56 s) windows with 50% overlap.
# Each row in the Inertial Signals files IS one complete window segment.
# No further segmentation is needed — we process each row as one segment.
WINDOW_SAMPLES = 128   # samples per pre-windowed row (2.56 s at 50 Hz)

# CWT / Spectrogram / Kurtogram params  — identical to SisFall model
CWT_WAVELET      = 'morl'
CWT_SCALES       = np.arange(1, 128)
STFT_NFFT        = 256
STFT_NPERSEG     = 64    # Reduced to fit 128-sample windows (must be <= signal length)
STFT_NOOVERLAP   = 32    # 50% overlap of STFT segments
KURTOGRAM_WINDOW = 32    # Reduced to fit 128-sample windows
KURTOGRAM_STEP   = 16

# Output  — identical to SisFall model
DPI       = 300
RESIZE_TO = None   # e.g., (224, 224) for ML models

# Device definitions: device_name -> (x_filekey, y_filekey, z_filekey)
# These keys are used to build the Inertial Signals filenames:
#   e.g. "total_acc_x" -> "total_acc_x_train.txt"
DEVICES: Dict[str, Tuple[str, str, str]] = {
    'TotalAcc': ('total_acc_x', 'total_acc_y', 'total_acc_z'),
    'BodyAcc':  ('body_acc_x',  'body_acc_y',  'body_acc_z'),
    'BodyGyro': ('body_gyro_x', 'body_gyro_y', 'body_gyro_z'),
}

# Activity label map (loaded from activity_labels.txt at runtime)
# Fallback hardcoded in case file is missing
FALLBACK_ACTIVITY_MAP = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING',
}
# ==================================


# ─────────────────────────────────────────────────────────────
#  SignalProcessor  (identical to SisFall model)
# ─────────────────────────────────────────────────────────────
class SignalProcessor:
    """Centralized signal processing utilities. Identical to SisFall model."""

    @staticmethod
    def read_space_separated(fpath: Path) -> np.ndarray:
        """
        Read a UCI HAR Inertial Signals file.
        Each row has WINDOW_SAMPLES float values separated by spaces.
        Returns a 2D numpy array: shape (n_rows, WINDOW_SAMPLES).
        """
        try:
            df = pd.read_csv(fpath, header=None, sep=r'\s+', engine='python')
            return df.values.astype(float)
        except Exception as e:
            raise IOError(f"Cannot read file {fpath}: {e}")

    @staticmethod
    def normalize(arr: np.ndarray) -> np.ndarray:
        """Zero-mean normalization with max scaling. Identical to SisFall model."""
        if arr is None or arr.size == 0:
            return arr
        arr = arr.astype(float)
        arr = arr - np.nanmean(arr)
        max_abs = np.nanmax(np.abs(arr))
        return arr / max_abs if np.isfinite(max_abs) and max_abs > 0 else arr

    @staticmethod
    def load_activity_map(dataset_dir: Path) -> Dict[int, str]:
        """
        Load activity_labels.txt -> {1: 'WALKING', 2: 'WALKING_UPSTAIRS', ...}
        Falls back to hardcoded map if file is missing.
        """
        label_file = dataset_dir / 'activity_labels.txt'
        if not label_file.exists():
            warnings.warn(f"activity_labels.txt not found, using fallback map.")
            return FALLBACK_ACTIVITY_MAP
        try:
            df = pd.read_csv(label_file, header=None, sep=r'\s+', engine='python')
            return {int(row[0]): str(row[1]).strip() for _, row in df.iterrows()}
        except Exception as e:
            warnings.warn(f"Failed to parse activity_labels.txt: {e}. Using fallback.")
            return FALLBACK_ACTIVITY_MAP

    @staticmethod
    def load_label_ids(fpath: Path) -> np.ndarray:
        """Load y_train.txt or y_test.txt -> 1D integer array of activity IDs."""
        df = pd.read_csv(fpath, header=None, sep=r'\s+', engine='python')
        return df.values.flatten().astype(int)

    @staticmethod
    def load_subject_ids(fpath: Path) -> np.ndarray:
        """Load subject_train.txt or subject_test.txt -> 1D integer array of subject IDs."""
        df = pd.read_csv(fpath, header=None, sep=r'\s+', engine='python')
        return df.values.flatten().astype(int)


# ─────────────────────────────────────────────────────────────
#  ImageGenerator  (100% identical to SisFall model)
# ─────────────────────────────────────────────────────────────
class ImageGenerator:
    """Generate time-frequency representation images. Identical to SisFall model."""

    @staticmethod
    def save_fig(fig, out_path: Path, dpi=DPI, resize_to=RESIZE_TO):
        """Save figure to PNG with tight cropping, no margins, high DPI."""
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        if resize_to:
            try:
                img = Image.open(out_path).resize(resize_to, Image.BILINEAR)
                img.save(out_path)
            except Exception as e:
                warnings.warn(f"Resize failed {out_path}: {e}")

    @staticmethod
    def scalogram(sig, fs, scales, wavelet, out_path: Path,
                  title='', t_offset=0.0, vmin=None, vmax=None):
        """CWT-based scalogram — clean image with no labels or colorbar."""
        coeffs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1.0 / fs)
        mag = np.abs(coeffs)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        duration = (len(sig) / fs) if fs > 0 else len(sig)
        extent = [t_offset, t_offset + duration, max(scales), min(scales)]
        ax.imshow(mag, aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.axis('off')
        ImageGenerator.save_fig(fig, out_path)

    @staticmethod
    def spectrogram(sig, fs, nperseg, noverlap, nfft, out_path: Path,
                    title='', t_offset=0.0, vmin=None, vmax=None):
        """STFT-based spectrogram — clean image with no labels or colorbar."""
        f, t_seg, Sxx = signal.spectrogram(
            sig, fs=fs, window='hann',
            nperseg=nperseg, noverlap=noverlap,
            nfft=nfft, scaling='spectrum'
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        t_axis = t_seg + t_offset
        ax.pcolormesh(t_axis, f, Sxx_db, shading='gouraud', vmin=vmin, vmax=vmax)
        ax.axis('off')
        ImageGenerator.save_fig(fig, out_path)

    @staticmethod
    def kurtogram(sig, fs, scales, window_samples, step, out_path: Path,
                  title='', t_offset=0.0, vmin=None, vmax=None):
        """Kurtosis-based time-frequency map from CWT — clean image with no labels or colorbar."""
        coeffs, _ = pywt.cwt(sig, scales, CWT_WAVELET, sampling_period=1.0 / fs)
        mag = np.abs(coeffs)
        n_scales, n_times = mag.shape

        if n_times < 1:
            raise ValueError('Empty CWT result')

        ws = max(3, int(window_samples))
        st = max(1, int(step))
        positions = list(range(0, max(1, n_times - ws + 1), st))

        if not positions:
            positions = [0]

        K = np.zeros((n_scales, len(positions)))
        for si in range(n_scales):
            for pi, p in enumerate(positions):
                window = mag[si, p: p + ws]
                if window.size >= 3:
                    K[si, pi] = stats.kurtosis(window, fisher=False, bias=False) - 3.0
                else:
                    K[si, pi] = 0.0

        time_pos = (np.array(positions) / fs) + t_offset if fs > 0 else np.array(positions, dtype=float)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        t_start = time_pos[0]
        t_end   = time_pos[-1] if len(time_pos) > 1 else t_start + n_times / fs
        extent = [t_start, t_end, max(scales), min(scales)]
        ax.imshow(K, aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.axis('off')
        ImageGenerator.save_fig(fig, out_path)

    @staticmethod
    def xyz_with_original(x_norm, y_norm, z_norm, original_signal, fs,
                          out_path: Path, title='XYZ + Original', t_offset_s=0.0):
        """
        Four subplots stacked vertically:
          1) X normalised signal   (blue)
          2) Y normalised signal   (orange)
          3) Z normalised signal   (green)
          4) Resultant magnitude   (black)
        Clean — no labels, titles, ticks, or margins. Identical to SisFall model.
        """
        n = min(len(x_norm), len(y_norm), len(z_norm), len(original_signal))
        if n < 8:
            raise ValueError("Insufficient data to plot XYZ combined image")

        x    = x_norm[:n]
        y    = y_norm[:n]
        z    = z_norm[:n]
        orig = original_signal[:n]
        t    = (np.arange(n) / fs + t_offset_s) if (fs and fs > 0) else np.arange(n)

        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

        axes[0].plot(t, x,    color='tab:blue',   linewidth=1.5)
        axes[0].axis('off')

        axes[1].plot(t, y,    color='tab:orange',  linewidth=1.5)
        axes[1].axis('off')

        axes[2].plot(t, z,    color='tab:green',   linewidth=1.5)
        axes[2].axis('off')

        axes[3].plot(t, orig, color='black',       linewidth=1.0)
        axes[3].axis('off')

        ImageGenerator.save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────
#  DataProcessor  (adapted for UCI HAR; same pattern as SisFall)
# ─────────────────────────────────────────────────────────────
class DataProcessor:
    """Main processing pipeline — UCI HAR adaptation of SisFall DataProcessor."""

    # Hardcoded global scaling bounds (same values as SisFall model)
    GLOBAL_BOUNDS: Dict[str, Tuple] = {
        'scalogram':   (0.01, 5.0),
        'spectrogram': (-120, -5),
        'kurtogram':   (-2, 120),
    }

    @staticmethod
    def compute_global_bounds():
        """
        Use hardcoded bounds — identical approach to SisFall model.
        Skips expensive computation across all files.
        """
        print("[INFO] Using predefined global scaling bounds...")
        DataProcessor.GLOBAL_BOUNDS = {
            'scalogram':   (0.01, 5.0),
            'spectrogram': (-120, -5),
            'kurtogram':   (-2, 120),
        }
        print(f"[INFO] Scalogram:   {DataProcessor.GLOBAL_BOUNDS['scalogram']}")
        print(f"[INFO] Spectrogram: {DataProcessor.GLOBAL_BOUNDS['spectrogram']}")
        print(f"[INFO] Kurtogram:   {DataProcessor.GLOBAL_BOUNDS['kurtogram']}")
        print("[INFO] Global bounds loaded!")

    @staticmethod
    def load_split(dataset_dir: Path, split: str,
                   activity_map: Dict[int, str]):
        """
        Load one split ('train' or 'test').

        Returns
        -------
        label_names  : List[str]  activity name per row  e.g. 'WALKING'
        subject_tags : List[str]  subject folder name    e.g. 'subject_01'
        signal_data  : Dict[str, np.ndarray]
            Keys: 'total_acc_x', 'total_acc_y', ... 'body_gyro_z'
            Values: 2D array of shape (n_rows, WINDOW_SAMPLES)
        """
        split_dir    = dataset_dir / split
        inertial_dir = split_dir / 'Inertial Signals'

        # ── Labels ──────────────────────────────────────────────────────────
        y_file = split_dir / f'y_{split}.txt'
        if not y_file.exists():
            raise FileNotFoundError(f"Label file not found: {y_file}")
        label_ids   = SignalProcessor.load_label_ids(y_file)
        label_names = [activity_map.get(lid, f'UNKNOWN_{lid}') for lid in label_ids]

        # ── Subjects ─────────────────────────────────────────────────────────
        subj_file = split_dir / f'subject_{split}.txt'
        if not subj_file.exists():
            raise FileNotFoundError(f"Subject file not found: {subj_file}")
        subject_ids  = SignalProcessor.load_subject_ids(subj_file)
        subject_tags = [f'subject_{sid:02d}' for sid in subject_ids]

        # ── Inertial signal files ─────────────────────────────────────────────
        signal_data: Dict[str, np.ndarray] = {}
        for device_name, (fk_x, fk_y, fk_z) in DEVICES.items():
            for fkey in (fk_x, fk_y, fk_z):
                fname = f'{fkey}_{split}.txt'
                fpath = inertial_dir / fname
                if not fpath.exists():
                    raise FileNotFoundError(f"Signal file not found: {fpath}")
                signal_data[fkey] = SignalProcessor.read_space_separated(fpath)

        # Validate row counts match
        n_rows = len(label_names)
        for key, arr in signal_data.items():
            if arr.shape[0] != n_rows:
                raise ValueError(
                    f"Row count mismatch: labels={n_rows}, {key}={arr.shape[0]}"
                )

        print(f"[INFO] Split '{split}': {n_rows} windows loaded, "
              f"{len(set(subject_tags))} subjects, "
              f"{len(set(label_names))} activities")
        return label_names, subject_tags, signal_data

    @staticmethod
    def process_row(row_idx: int,
                    label_name: str,
                    subject_tag: str,
                    signal_data: Dict[str, np.ndarray],
                    output_root: Path):
        """
        Process a single pre-windowed row (window) and generate all images.
        Mirrors DataProcessor.process_file() in the SisFall model exactly.

        Parameters
        ----------
        row_idx      : integer row index (used for output filename)
        label_name   : activity class string, e.g. 'WALKING'
        subject_tag  : subject folder name,   e.g. 'subject_01'
        signal_data  : dict of all loaded 2D arrays  {filekey: (n_rows, 128)}
        output_root  : root output directory
        """
        cls = label_name       # equivalent to 'Fall' / 'Daily Living' in SisFall
        base = f'row_{row_idx:05d}'   # base filename

        # t_offset for this row: row starts at time 0 (each row is independent window)
        t_offset = 0.0

        for device_name, (fk_x, fk_y, fk_z) in DEVICES.items():
            axis_filekeys = {'X': fk_x, 'Y': fk_y, 'Z': fk_z}

            norm_sigs: Dict[str, np.ndarray] = {}
            raw_sigs:  Dict[str, np.ndarray] = {}

            for axis, fkey in axis_filekeys.items():
                raw_window = signal_data[fkey][row_idx, :].copy()  # 128 samples
                raw_window = raw_window[np.isfinite(raw_window)]

                if raw_window.size < 8:
                    print(f"[SKIP] {base} {device_name} {axis}: too few samples")
                    continue

                sig = SignalProcessor.normalize(raw_window)

                norm_sigs[axis] = sig
                raw_sigs[axis]  = raw_window

                fname_out = f'{base}_{axis}.png'

                # Capture current sig / device / axis for lambdas (avoid closure bug)
                _sig     = sig
                _device  = device_name
                _axis    = axis

                for img_type, gen_func in [
                    ('Scalogram',
                     lambda p, s=_sig, d=_device, a=_axis: ImageGenerator.scalogram(
                         s, FS, CWT_SCALES, CWT_WAVELET, p,
                         f'{base}_{d}_{a} Scalogram', t_offset,
                         *DataProcessor.GLOBAL_BOUNDS['scalogram']
                     )),
                    ('Spectrogram',
                     lambda p, s=_sig, d=_device, a=_axis: ImageGenerator.spectrogram(
                         s, FS, STFT_NPERSEG, STFT_NOOVERLAP, STFT_NFFT, p,
                         f'{base}_{d}_{a} Spectrogram', t_offset,
                         *DataProcessor.GLOBAL_BOUNDS['spectrogram']
                     )),
                    ('Kurtogram',
                     lambda p, s=_sig, d=_device, a=_axis: ImageGenerator.kurtogram(
                         s, FS, CWT_SCALES, KURTOGRAM_WINDOW, KURTOGRAM_STEP, p,
                         f'{base}_{d}_{a} Kurtogram', t_offset,
                         *DataProcessor.GLOBAL_BOUNDS['kurtogram']
                     )),
                ]:
                    try:
                        out_folder = output_root / device_name / cls / img_type / subject_tag
                        out_folder.mkdir(parents=True, exist_ok=True)
                        gen_func(out_folder / fname_out)
                    except Exception as e:
                        print(f"[ERROR] {img_type} {base}_{device_name}_{axis}: {e}")

            # ── XYZ_Combined (identical logic to SisFall model) ──────────────
            if len(norm_sigs) == 3 and len(raw_sigs) == 3:
                try:
                    x_raw = raw_sigs.get('X', np.array([]))
                    y_raw = raw_sigs.get('Y', np.array([]))
                    z_raw = raw_sigs.get('Z', np.array([]))

                    if x_raw.size > 0 and y_raw.size > 0 and z_raw.size > 0:
                        min_len = min(len(x_raw), len(y_raw), len(z_raw))
                        x_raw = x_raw[:min_len]
                        y_raw = y_raw[:min_len]
                        z_raw = z_raw[:min_len]

                        resultant      = np.sqrt(x_raw**2 + y_raw**2 + z_raw**2)
                        resultant_norm = SignalProcessor.normalize(resultant)

                        fname_xyz  = f'{base}_XYZ.png'
                        out_folder = output_root / device_name / cls / 'XYZ_Combined' / subject_tag
                        out_folder.mkdir(parents=True, exist_ok=True)

                        ImageGenerator.xyz_with_original(
                            norm_sigs['X'], norm_sigs['Y'], norm_sigs['Z'],
                            resultant_norm, FS,
                            out_folder / fname_xyz,
                            title=f'{base}_{device_name} XYZ + Resultant',
                            t_offset_s=t_offset
                        )
                except Exception as e:
                    print(f"[ERROR] XYZ_Combined {base}_{device_name}: {e}")

        print(f"[OK] {base} | {cls} | {subject_tag}")


# ─────────────────────────────────────────────────────────────
#  run_pipeline  (same structure as SisFall run_pipeline)
# ─────────────────────────────────────────────────────────────
def run_pipeline(base_dir: Path):
    """
    Main pipeline — mirrors run_pipeline() in SisFall model.
    Processes both train and test splits of the UCI HAR dataset.
    """
    base_dir     = base_dir.resolve()
    dataset_dir  = base_dir / RAW_DATA_DIR
    output_root  = base_dir / OUTPUT_DIR

    if not dataset_dir.exists():
        print(f"[FATAL] Dataset not found: {dataset_dir}")
        print(f"        Please place the UCI HAR Dataset folder at: {dataset_dir}")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    # Load activity labels
    activity_map = SignalProcessor.load_activity_map(dataset_dir)
    print(f"[INFO] Activity classes: {list(activity_map.values())}")

    # Load global bounds (hardcoded — same as SisFall)
    DataProcessor.compute_global_bounds()

    # Process both splits
    for split in ('train', 'test'):
        print(f"\n{'='*55}")
        print(f"  Processing split: {split.upper()}")
        print(f"{'='*55}")

        try:
            label_names, subject_tags, signal_data = DataProcessor.load_split(
                dataset_dir, split, activity_map
            )
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            print(f"[SKIP] Skipping split '{split}'.")
            continue
        except Exception as e:
            print(f"[ERROR] Failed to load split '{split}': {e}")
            traceback.print_exc()
            continue

        n_rows = len(label_names)
        print(f"[INFO] Processing {n_rows} windows for split '{split}' ...")

        for row_idx in range(n_rows):
            try:
                DataProcessor.process_row(
                    row_idx=row_idx,
                    label_name=label_names[row_idx],
                    subject_tag=subject_tags[row_idx],
                    signal_data=signal_data,
                    output_root=output_root
                )
            except Exception as e:
                print(f"[ERROR] Row {row_idx:05d}: {e}")
                traceback.print_exc()

    print(f"\n{'='*55}")
    print(f"  Complete. Output: {output_root}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    base = DEFAULT_BASE_DIR
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1])
        base = arg if arg.is_dir() else base

    print(f"Base directory : {base}")
    print(f"Dataset folder : {base / RAW_DATA_DIR}")
    print(f"Output folder  : {base / OUTPUT_DIR}")
    print()

    try:
        run_pipeline(base)
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
