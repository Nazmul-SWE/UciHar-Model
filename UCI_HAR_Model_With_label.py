# UCI_HAR_Model_With_label.py
"""
Generate Scalogram, Spectrogram, Kurtogram, and XYZ_Combined images
for the UCI Human Activity Recognition (HAR) dataset.

Mirrors all functionality of SisFall_Model_With_lable.py exactly.
Labels (activity class + subject + axis) are printed as visible text
on every generated image — same as SisFall model.

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

Output folder structure (identical pattern to SisFall):
  <BASE_DIR>/
    Generated Images/
      TotalAcc/
        WALKING/
          Scalogram/  subject_01/  row_00000_X.png
          Spectrogram/subject_01/  row_00000_X.png
          Kurtogram/  subject_01/  row_00000_X.png
          XYZ_Combined/subject_01/ row_00000_XYZ.png
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
import matplotlib.patches as mpatches
import pywt
from scipy import signal, stats
from PIL import Image

# ============ CONFIG ============
DEFAULT_BASE_DIR = Path(r"D:\4.1\CSE 400-A\UciHar-Model").resolve()
RAW_DATA_DIR     = 'UCI HAR Dataset'
OUTPUT_DIR       = 'Generated Images'

# Signal processing
# UCI HAR uses 50 Hz  (SisFall used 200 Hz)
FS = 50.0

# UCI HAR raw Inertial Signals are pre-windowed: each row = 128 samples (2.56 s, 50% overlap)
WINDOW_SAMPLES = 128

# CWT / Spectrogram / Kurtogram params
# Wavelet & scale range identical to SisFall model
CWT_WAVELET      = 'morl'
CWT_SCALES       = np.arange(1, 128)
STFT_NFFT        = 256
STFT_NPERSEG     = 64    # must be <= WINDOW_SAMPLES (128)
STFT_NOOVERLAP   = 32    # 50 % overlap of STFT frames
KURTOGRAM_WINDOW = 32    # must be <= WINDOW_SAMPLES
KURTOGRAM_STEP   = 16

# Output
DPI       = 300
RESIZE_TO = None   # set e.g. (224, 224) for ML models

# ── Label overlay style (text burned onto every image) ──────────
LABEL_FONTSIZE   = 13
LABEL_COLOR      = 'white'
LABEL_BG_COLOR   = 'black'     # background box behind label text
LABEL_BG_ALPHA   = 0.55        # transparency of background box
LABEL_POSITION   = 'top-left'  # 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'

# Device definitions  →  device_name : (x_file_key, y_file_key, z_file_key)
# Keys map to Inertial Signals filenames:
#   e.g.  'total_acc_x'  →  'total_acc_x_train.txt'
DEVICES: Dict[str, Tuple[str, str, str]] = {
    'TotalAcc': ('total_acc_x', 'total_acc_y', 'total_acc_z'),
    'BodyAcc':  ('body_acc_x',  'body_acc_y',  'body_acc_z'),
    'BodyGyro': ('body_gyro_x', 'body_gyro_y', 'body_gyro_z'),
}

# Hardcoded fallback activity map (used if activity_labels.txt is missing)
FALLBACK_ACTIVITY_MAP: Dict[int, str] = {
    1: 'WALKING',
    2: 'WALKING_UPSTAIRS',
    3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING',
    5: 'STANDING',
    6: 'LAYING',
}
# ==================================


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: draw the label overlay on an axes object
# ─────────────────────────────────────────────────────────────────────────────
def _draw_label_overlay(ax, label_text: str,
                        position: str = LABEL_POSITION,
                        fontsize: int = LABEL_FONTSIZE,
                        color: str = LABEL_COLOR,
                        bg_color: str = LABEL_BG_COLOR,
                        bg_alpha: float = LABEL_BG_ALPHA):
    """
    Burn a text label onto the axes.

    The label is drawn in axes-fraction coordinates so it always appears
    at a fixed corner regardless of the data range.  A semi-transparent
    background box makes it readable over any image content.

    Parameters
    ----------
    ax          : matplotlib Axes
    label_text  : string to display  (e.g. "WALKING | TotalAcc | X-axis")
    position    : 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'
    fontsize    : font size in points
    color       : text colour
    bg_color    : background box colour
    bg_alpha    : background box alpha (0 = transparent, 1 = opaque)
    """
    pos_map = {
        'top-left':     (0.01, 0.97, 'left',  'top'),
        'top-right':    (0.99, 0.97, 'right', 'top'),
        'bottom-left':  (0.01, 0.03, 'left',  'bottom'),
        'bottom-right': (0.99, 0.03, 'right', 'bottom'),
    }
    x, y, ha, va = pos_map.get(position, pos_map['top-left'])

    ax.text(
        x, y, label_text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=color,
        ha=ha, va=va,
        fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor=bg_color,
            alpha=bg_alpha,
            edgecolor='none'
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SignalProcessor
# ─────────────────────────────────────────────────────────────────────────────
class SignalProcessor:
    """Centralized signal processing utilities. Identical to SisFall model."""

    @staticmethod
    def read_space_separated(fpath: Path) -> np.ndarray:
        """
        Read one UCI HAR Inertial Signals file.
        Each row contains WINDOW_SAMPLES whitespace-separated float values.
        Returns ndarray of shape  (n_rows, WINDOW_SAMPLES).
        """
        try:
            df = pd.read_csv(fpath, header=None, sep=r'\s+', engine='python')
            return df.values.astype(float)
        except Exception as exc:
            raise IOError(f"Cannot read file {fpath}: {exc}")

    @staticmethod
    def normalize(arr: np.ndarray) -> np.ndarray:
        """Zero-mean normalization with max-abs scaling. Identical to SisFall model."""
        if arr is None or arr.size == 0:
            return arr
        arr = arr.astype(float)
        arr = arr - np.nanmean(arr)
        max_abs = np.nanmax(np.abs(arr))
        return arr / max_abs if (np.isfinite(max_abs) and max_abs > 0) else arr

    @staticmethod
    def load_activity_map(dataset_dir: Path) -> Dict[int, str]:
        """
        Parse  activity_labels.txt  →  {1: 'WALKING', 2: 'WALKING_UPSTAIRS', ...}
        Falls back to hardcoded map when the file is absent.
        """
        label_file = dataset_dir / 'activity_labels.txt'
        if not label_file.exists():
            warnings.warn("activity_labels.txt not found — using hardcoded fallback map.")
            return FALLBACK_ACTIVITY_MAP
        try:
            df = pd.read_csv(label_file, header=None, sep=r'\s+', engine='python')
            return {int(row[0]): str(row[1]).strip() for _, row in df.iterrows()}
        except Exception as exc:
            warnings.warn(f"Failed to parse activity_labels.txt ({exc}) — using fallback.")
            return FALLBACK_ACTIVITY_MAP

    @staticmethod
    def load_label_ids(fpath: Path) -> np.ndarray:
        """Read  y_train.txt / y_test.txt  →  1-D int array of activity IDs."""
        df = pd.read_csv(fpath, header=None, sep=r'\s+', engine='python')
        return df.values.flatten().astype(int)

    @staticmethod
    def load_subject_ids(fpath: Path) -> np.ndarray:
        """Read  subject_train.txt / subject_test.txt  →  1-D int array of subject IDs."""
        df = pd.read_csv(fpath, header=None, sep=r'\s+', engine='python')
        return df.values.flatten().astype(int)


# ─────────────────────────────────────────────────────────────────────────────
#  ImageGenerator  — every method burns a visible label onto the image
# ─────────────────────────────────────────────────────────────────────────────
class ImageGenerator:
    """
    Generate time-frequency representation images.
    Every image has the activity label + device + axis printed on it,
    mirroring the label system of SisFall_Model_With_lable.py.
    """

    @staticmethod
    def save_fig(fig, out_path: Path, dpi: int = DPI, resize_to=RESIZE_TO):
        """
        Save figure to PNG — publication-ready layout.
        Uses tight_layout() so axis labels, titles, and colorbars
        are never clipped.  Saves at high DPI (300).
        """
        plt.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        if resize_to:
            try:
                img = Image.open(out_path).resize(resize_to, Image.BILINEAR)
                img.save(out_path)
            except Exception as exc:
                warnings.warn(f"Resize failed for {out_path}: {exc}")

    # ── Scalogram ─────────────────────────────────────────────────────────────
    @staticmethod
    def scalogram(sig: np.ndarray, fs: float, scales, wavelet: str,
                  out_path: Path, label_text: str = '',
                  t_offset: float = 0.0, vmin=None, vmax=None):
        """
        CWT-based scalogram — publication-ready.
        Includes:
          - Descriptive title (sample ID + activity label)
          - X-axis label: Time (s)
          - Y-axis label: Scale
          - Axis ticks enabled
          - Colorbar with label: Magnitude
          - tight_layout()
          - DPI = 300
          - Readable font sizes
          - Label overlay (top-left corner)
        """
        coeffs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1.0 / fs)
        mag = np.abs(coeffs)

        fig, ax = plt.subplots(figsize=(9, 5))
        duration = (len(sig) / fs) if fs > 0 else len(sig)
        extent   = [t_offset, t_offset + duration, float(max(scales)), float(min(scales))]

        im = ax.imshow(
            mag, aspect='auto', extent=extent,
            vmin=vmin, vmax=vmax, origin='upper'
        )
        ax.invert_yaxis()

        # ── Axes labels & ticks ──
        ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold', labelpad=6)
        ax.set_ylabel('Scale',    fontsize=13, fontweight='bold', labelpad=6)
        ax.tick_params(axis='both', which='major', labelsize=11, length=4, width=1)

        # ── Title: sample ID + activity from label_text ──
        title_str = f'Scalogram  —  {label_text}' if label_text else 'Scalogram'
        ax.set_title(title_str, fontsize=13, fontweight='bold', pad=8)

        # ── Colorbar ──
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Magnitude', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        # ── Label overlay (top-left corner) ──
        if label_text:
            _draw_label_overlay(ax, label_text)

        ImageGenerator.save_fig(fig, out_path)

    # ── Spectrogram ───────────────────────────────────────────────────────────
    @staticmethod
    def spectrogram(sig: np.ndarray, fs: float,
                    nperseg: int, noverlap: int, nfft: int,
                    out_path: Path, label_text: str = '',
                    t_offset: float = 0.0, vmin=None, vmax=None):
        """
        STFT-based power spectrogram (dB scale) — publication-ready.
        Includes:
          - Descriptive title (sample ID + activity label)
          - X-axis label: Time (s)
          - Y-axis label: Frequency (Hz)
          - Axis ticks enabled
          - Colorbar with label: Power (dB)
          - tight_layout()
          - DPI = 300
          - Readable font sizes
          - Label overlay (top-left corner)
        """
        f, t_seg, Sxx = signal.spectrogram(
            sig, fs=fs, window='hann',
            nperseg=nperseg, noverlap=noverlap,
            nfft=nfft, scaling='spectrum'
        )
        Sxx_db = 10.0 * np.log10(Sxx + 1e-12)

        fig, ax = plt.subplots(figsize=(9, 5))
        t_axis = t_seg + t_offset
        pcm = ax.pcolormesh(
            t_axis, f, Sxx_db,
            shading='gouraud', vmin=vmin, vmax=vmax
        )

        # ── Axes labels & ticks ──
        ax.set_xlabel('Time (s)',        fontsize=13, fontweight='bold', labelpad=6)
        ax.set_ylabel('Frequency (Hz)',  fontsize=13, fontweight='bold', labelpad=6)
        ax.tick_params(axis='both', which='major', labelsize=11, length=4, width=1)

        # ── Title ──
        title_str = f'Spectrogram  —  {label_text}' if label_text else 'Spectrogram'
        ax.set_title(title_str, fontsize=13, fontweight='bold', pad=8)

        # ── Colorbar ──
        cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Power (dB)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        # ── Label overlay (top-left corner) ──
        if label_text:
            _draw_label_overlay(ax, label_text)

        ImageGenerator.save_fig(fig, out_path)

    # ── Kurtogram ─────────────────────────────────────────────────────────────
    @staticmethod
    def kurtogram(sig: np.ndarray, fs: float, scales,
                  window_samples: int, step: int,
                  out_path: Path, label_text: str = '',
                  t_offset: float = 0.0, vmin=None, vmax=None):
        """
        Kurtosis-based time-frequency map from CWT — publication-ready.
        Includes:
          - Descriptive title (sample ID + activity label)
          - X-axis label: Time (s)
          - Y-axis label: Scale
          - Axis ticks enabled
          - Colorbar with label: Excess Kurtosis
          - tight_layout()
          - DPI = 300
          - Readable font sizes
          - Label overlay (top-left corner)
        """
        coeffs, _ = pywt.cwt(sig, scales, CWT_WAVELET, sampling_period=1.0 / fs)
        mag = np.abs(coeffs)
        n_scales, n_times = mag.shape

        if n_times < 1:
            raise ValueError('Empty CWT result — signal too short.')

        ws        = max(3, int(window_samples))
        st        = max(1, int(step))
        positions = list(range(0, max(1, n_times - ws + 1), st))
        if not positions:
            positions = [0]

        K = np.zeros((n_scales, len(positions)))
        for si in range(n_scales):
            for pi, p in enumerate(positions):
                window = mag[si, p: p + ws]
                K[si, pi] = (
                    stats.kurtosis(window, fisher=False, bias=False) - 3.0
                    if window.size >= 3 else 0.0
                )

        time_pos = (
            (np.array(positions) / fs) + t_offset
            if fs > 0 else np.array(positions, dtype=float)
        )
        t_start = float(time_pos[0])
        t_end   = float(time_pos[-1]) if len(time_pos) > 1 else t_start + n_times / fs
        extent  = [t_start, t_end, float(max(scales)), float(min(scales))]

        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(
            K, aspect='auto', extent=extent,
            vmin=vmin, vmax=vmax, origin='upper'
        )
        ax.invert_yaxis()

        # ── Axes labels & ticks ──
        ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold', labelpad=6)
        ax.set_ylabel('Scale',    fontsize=13, fontweight='bold', labelpad=6)
        ax.tick_params(axis='both', which='major', labelsize=11, length=4, width=1)

        # ── Title ──
        title_str = f'Kurtogram  —  {label_text}' if label_text else 'Kurtogram'
        ax.set_title(title_str, fontsize=13, fontweight='bold', pad=8)

        # ── Colorbar ──
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Excess Kurtosis', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        # ── Label overlay (top-left corner) ──
        if label_text:
            _draw_label_overlay(ax, label_text)

        ImageGenerator.save_fig(fig, out_path)

    # ── XYZ_Combined ──────────────────────────────────────────────────────────
    @staticmethod
    def xyz_with_original(x_norm: np.ndarray, y_norm: np.ndarray,
                          z_norm: np.ndarray, original_signal: np.ndarray,
                          fs: float, out_path: Path,
                          label_text: str = 'XYZ + Original',
                          t_offset_s: float = 0.0):
        """
        Four subplots stacked vertically — publication-ready.
          1) X normalised signal   (blue)
          2) Y normalised signal   (orange)
          3) Z normalised signal   (green)
          4) Resultant magnitude   (black)

        Includes:
          - Descriptive overall title (sample ID + activity label)
          - Individual subplot titles: X-axis, Y-axis, Z-axis, Resultant
          - Y-axis label on each subplot: Amplitude (Normalised)
          - X-axis label on bottom subplot only: Time (s)
          - Axis ticks enabled on all subplots
          - tight_layout()
          - DPI = 300
          - Readable font sizes
          - Label overlay (top-left of first subplot)
        """
        n = min(len(x_norm), len(y_norm), len(z_norm), len(original_signal))
        if n < 8:
            raise ValueError("Insufficient data to plot XYZ combined image (need >= 8 samples).")

        x    = x_norm[:n]
        y    = y_norm[:n]
        z    = z_norm[:n]
        orig = original_signal[:n]
        t    = (
            np.arange(n) / fs + t_offset_s
            if (fs and fs > 0) else np.arange(n, dtype=float)
        )

        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

        # ── Overall figure title ──
        fig_title = f'XYZ + Resultant  —  {label_text}' if label_text else 'XYZ + Resultant'
        fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=1.01)

        subplot_info = [
            (axes[0], t, x,    'tab:blue',   'X-axis Signal'),
            (axes[1], t, y,    'tab:orange',  'Y-axis Signal'),
            (axes[2], t, z,    'tab:green',   'Z-axis Signal'),
            (axes[3], t, orig, 'black',       'Resultant Magnitude'),
        ]

        for i, (ax, t_data, sig_data, color, sub_title) in enumerate(subplot_info):
            ax.plot(t_data, sig_data, color=color, linewidth=1.5)
            ax.set_title(sub_title, fontsize=11, fontweight='bold', pad=4)
            ax.set_ylabel('Amplitude\n(Normalised)', fontsize=10,
                          fontweight='bold', labelpad=4)
            ax.tick_params(axis='both', which='major', labelsize=9,
                           length=3, width=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # X-axis label only on the bottom subplot
            if i == 3:
                ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold', labelpad=6)

        # ── Label overlay on top subplot ──
        if label_text:
            _draw_label_overlay(axes[0], label_text)

        ImageGenerator.save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  DataProcessor
# ─────────────────────────────────────────────────────────────────────────────
class DataProcessor:
    """Main processing pipeline — UCI HAR adaptation of SisFall DataProcessor."""

    # Hardcoded global scaling bounds — same values as SisFall model
    GLOBAL_BOUNDS: Dict[str, Tuple] = {
        'scalogram':   (0.01, 5.0),
        'spectrogram': (-120, -5),
        'kurtogram':   (-2, 120),
    }

    @staticmethod
    def compute_global_bounds():
        """
        Use hardcoded bounds to skip expensive per-file computation.
        Identical approach to SisFall model.
        """
        print("[INFO] Using predefined global scaling bounds...")
        DataProcessor.GLOBAL_BOUNDS = {
            'scalogram':   (0.01, 5.0),
            'spectrogram': (-120, -5),
            'kurtogram':   (-2, 120),
        }
        print(f"[INFO] Scalogram   : {DataProcessor.GLOBAL_BOUNDS['scalogram']}")
        print(f"[INFO] Spectrogram : {DataProcessor.GLOBAL_BOUNDS['spectrogram']}")
        print(f"[INFO] Kurtogram   : {DataProcessor.GLOBAL_BOUNDS['kurtogram']}")
        print("[INFO] Global bounds loaded!")

    @staticmethod
    def load_split(dataset_dir: Path, split: str,
                   activity_map: Dict[int, str]):
        """
        Load one split ('train' or 'test') completely into memory.

        Returns
        -------
        label_names  : List[str]   activity name per row  e.g. 'WALKING'
        subject_tags : List[str]   subject folder name    e.g. 'subject_01'
        signal_data  : Dict[str, np.ndarray]
            keys   = file-key strings  e.g. 'total_acc_x', 'body_gyro_z'
            values = 2-D arrays of shape (n_rows, WINDOW_SAMPLES)
        """
        split_dir    = dataset_dir / split
        inertial_dir = split_dir / 'Inertial Signals'

        # ── Activity labels ──────────────────────────────────────────────────
        y_file = split_dir / f'y_{split}.txt'
        if not y_file.exists():
            raise FileNotFoundError(f"Label file not found: {y_file}")
        label_ids   = SignalProcessor.load_label_ids(y_file)
        label_names = [activity_map.get(int(lid), f'UNKNOWN_{lid}') for lid in label_ids]

        # ── Subject IDs ──────────────────────────────────────────────────────
        subj_file = split_dir / f'subject_{split}.txt'
        if not subj_file.exists():
            raise FileNotFoundError(f"Subject file not found: {subj_file}")
        subject_ids  = SignalProcessor.load_subject_ids(subj_file)
        subject_tags = [f'subject_{int(sid):02d}' for sid in subject_ids]

        # ── Inertial signal files ────────────────────────────────────────────
        signal_data: Dict[str, np.ndarray] = {}
        for _device_name, (fk_x, fk_y, fk_z) in DEVICES.items():
            for fkey in (fk_x, fk_y, fk_z):
                fname = f'{fkey}_{split}.txt'
                fpath = inertial_dir / fname
                if not fpath.exists():
                    raise FileNotFoundError(f"Signal file not found: {fpath}")
                signal_data[fkey] = SignalProcessor.read_space_separated(fpath)

        # ── Validate row counts ──────────────────────────────────────────────
        n_rows = len(label_names)
        for key, arr in signal_data.items():
            if arr.shape[0] != n_rows:
                raise ValueError(
                    f"Row-count mismatch: labels={n_rows}, '{key}'={arr.shape[0]}"
                )

        print(
            f"[INFO] Split '{split}': {n_rows} windows, "
            f"{len(set(subject_tags))} subjects, "
            f"{len(set(label_names))} activity classes"
        )
        return label_names, subject_tags, signal_data

    @staticmethod
    def process_row(row_idx: int,
                    label_name: str,
                    subject_tag: str,
                    signal_data: Dict[str, np.ndarray],
                    output_root: Path):
        """
        Process one pre-windowed row and generate all four image types.

        Output path pattern (identical to SisFall model):
          output_root / device / cls / img_type / subject / filename.png

        Label text is printed visibly on every generated image.

        Parameters
        ----------
        row_idx      : row index within the split (used in output filename)
        label_name   : activity class string  e.g. 'WALKING'
        subject_tag  : subject folder name    e.g. 'subject_01'
        signal_data  : {file_key: 2-D array (n_rows, 128)}
        output_root  : root of 'Generated Images' folder
        """
        cls    = label_name                  # folder label — same role as 'Fall'/'Daily Living'
        base   = f'row_{row_idx:05d}'        # base filename
        t_offset = 0.0                       # each row is an independent window starting at t=0

        for device_name, (fk_x, fk_y, fk_z) in DEVICES.items():
            axis_filekeys: Dict[str, str] = {'X': fk_x, 'Y': fk_y, 'Z': fk_z}

            norm_sigs: Dict[str, np.ndarray] = {}
            raw_sigs:  Dict[str, np.ndarray] = {}

            for axis, fkey in axis_filekeys.items():

                # Extract the 128-sample window for this row & axis
                raw_window = signal_data[fkey][row_idx, :].copy()
                raw_window = raw_window[np.isfinite(raw_window)]

                if raw_window.size < 8:
                    print(f"[SKIP] {base} {device_name} {axis}: too few valid samples")
                    continue

                sig = SignalProcessor.normalize(raw_window)

                norm_sigs[axis] = sig
                raw_sigs[axis]  = raw_window

                fname_out = f'{base}_{axis}.png'

                # Label text burned onto image:
                #   "WALKING | TotalAcc | X-axis | subject_01"
                img_label = f'{cls}  |  {device_name}  |  {axis}-axis  |  {subject_tag}'

                # Capture loop variables for lambda closures (avoids Python closure bug)
                _sig    = sig
                _device = device_name
                _axis   = axis
                _label  = img_label

                for img_type, gen_func in [
                    ('Scalogram',
                     lambda p, s=_sig, lbl=_label: ImageGenerator.scalogram(
                         s, FS, CWT_SCALES, CWT_WAVELET, p,
                         label_text=lbl,
                         t_offset=t_offset,
                         vmin=DataProcessor.GLOBAL_BOUNDS['scalogram'][0],
                         vmax=DataProcessor.GLOBAL_BOUNDS['scalogram'][1],
                     )),
                    ('Spectrogram',
                     lambda p, s=_sig, lbl=_label: ImageGenerator.spectrogram(
                         s, FS, STFT_NPERSEG, STFT_NOOVERLAP, STFT_NFFT, p,
                         label_text=lbl,
                         t_offset=t_offset,
                         vmin=DataProcessor.GLOBAL_BOUNDS['spectrogram'][0],
                         vmax=DataProcessor.GLOBAL_BOUNDS['spectrogram'][1],
                     )),
                    ('Kurtogram',
                     lambda p, s=_sig, lbl=_label: ImageGenerator.kurtogram(
                         s, FS, CWT_SCALES, KURTOGRAM_WINDOW, KURTOGRAM_STEP, p,
                         label_text=lbl,
                         t_offset=t_offset,
                         vmin=DataProcessor.GLOBAL_BOUNDS['kurtogram'][0],
                         vmax=DataProcessor.GLOBAL_BOUNDS['kurtogram'][1],
                     )),
                ]:
                    try:
                        # Output path: output_root/device/cls/img_type/subject/file.png
                        out_folder = output_root / device_name / cls / img_type / subject_tag
                        out_folder.mkdir(parents=True, exist_ok=True)
                        gen_func(out_folder / fname_out)
                    except Exception as exc:
                        print(f"[ERROR] {img_type} {base}_{device_name}_{axis}: {exc}")

            # ── XYZ_Combined image ───────────────────────────────────────────
            # Generated only when all three axes were processed successfully
            if len(norm_sigs) == 3 and len(raw_sigs) == 3:
                try:
                    x_raw = raw_sigs.get('X', np.array([]))
                    y_raw = raw_sigs.get('Y', np.array([]))
                    z_raw = raw_sigs.get('Z', np.array([]))

                    if x_raw.size > 0 and y_raw.size > 0 and z_raw.size > 0:
                        min_len = min(len(x_raw), len(y_raw), len(z_raw))
                        x_raw   = x_raw[:min_len]
                        y_raw   = y_raw[:min_len]
                        z_raw   = z_raw[:min_len]

                        resultant      = np.sqrt(x_raw**2 + y_raw**2 + z_raw**2)
                        resultant_norm = SignalProcessor.normalize(resultant)

                        fname_xyz = f'{base}_XYZ.png'

                        # Label for XYZ image
                        xyz_label = f'{cls}  |  {device_name}  |  XYZ + Resultant  |  {subject_tag}'

                        out_folder = output_root / device_name / cls / 'XYZ_Combined' / subject_tag
                        out_folder.mkdir(parents=True, exist_ok=True)

                        ImageGenerator.xyz_with_original(
                            norm_sigs['X'], norm_sigs['Y'], norm_sigs['Z'],
                            resultant_norm, FS,
                            out_folder / fname_xyz,
                            label_text=xyz_label,
                            t_offset_s=t_offset,
                        )
                except Exception as exc:
                    print(f"[ERROR] XYZ_Combined {base}_{device_name}: {exc}")

        print(f"[OK] {base}  |  {cls}  |  {subject_tag}")


# ─────────────────────────────────────────────────────────────────────────────
#  run_pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(base_dir: Path):
    """
    Main pipeline — mirrors run_pipeline() in SisFall_Model_With_lable.py.
    Processes both 'train' and 'test' splits of the UCI HAR dataset.
    """
    base_dir    = base_dir.resolve()
    dataset_dir = base_dir / RAW_DATA_DIR
    output_root = base_dir / OUTPUT_DIR

    if not dataset_dir.exists():
        print(f"[FATAL] Dataset folder not found: {dataset_dir}")
        print(f"        Expected path: {dataset_dir}")
        print(f"        Please download the UCI HAR dataset and place it there.")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    # Load activity label map from dataset
    activity_map = SignalProcessor.load_activity_map(dataset_dir)
    print(f"[INFO] Activity classes: {list(activity_map.values())}")

    # Initialise global image-scaling bounds (hardcoded, same as SisFall)
    DataProcessor.compute_global_bounds()

    # Process both splits
    for split in ('train', 'test'):
        print(f"\n{'=' * 60}")
        print(f"  Split: {split.upper()}")
        print(f"{'=' * 60}")

        try:
            label_names, subject_tags, signal_data = DataProcessor.load_split(
                dataset_dir, split, activity_map
            )
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            print(f"[SKIP]  Skipping split '{split}'.")
            continue
        except Exception as exc:
            print(f"[ERROR] Failed to load split '{split}': {exc}")
            traceback.print_exc()
            continue

        n_rows = len(label_names)
        print(f"[INFO] Generating images for {n_rows} windows ...")

        for row_idx in range(n_rows):
            try:
                DataProcessor.process_row(
                    row_idx      = row_idx,
                    label_name   = label_names[row_idx],
                    subject_tag  = subject_tags[row_idx],
                    signal_data  = signal_data,
                    output_root  = output_root,
                )
            except Exception as exc:
                print(f"[ERROR] row {row_idx:05d}: {exc}")
                traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  Complete.  Output saved to: {output_root}")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
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
    except Exception as exc:
        print(f"[FATAL] {exc}")
        traceback.print_exc()
