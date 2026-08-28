"""
Multi-Channel Audio Feature Extractor — v3
Choking vs. Laughing CNN Classifier
══════════════════════════════════════════

Addresses the train≈0.85 / val≈0.78 / test≈0.57 gap and the strong
laughter-bias observed in v2.

CHANGES FROM v2
───────────────
1. SPLIT BEFORE AUGMENTATION
   v2: augmentation wrote `<stem>_<tag>.npy` next to `<stem>.npy` in a
       single class folder; downstream splits at the .npy level leaked
       augmented siblings of the same source across train/val/test.
   v3: stratified file-level split happens FIRST and is persisted as a
       manifest. Each split lives in its own folder. Only TRAIN gets
       augmented; VAL/TEST contain originals only.

2. AUGMENTATION RUNS ON THE RAW SIGNAL
   v2: load → pre-emphasis → normalise → extract → augment(already-filtered)
       → pitch_shift artifacts on a pre-emphasised signal; augmented
       variants saw pre-emphasis applied to a non-canonical input.
   v3: load → augment(raw) → pre-emphasis → extract.

3. LESS AGGRESSIVE AUGMENTATION
   pitch_shift ±1 semitone (was ±2 — too large for laughter contour);
   time_stretch ±10% (was ±15%); SNR-controlled noise at 25 dB instead
   of fixed amplitude; added random gain ±3 dB in raw-audio domain.

4. AMPLITUDE / LOUDNESS PRESERVED
   v2: power_to_db(ref=np.max) + per-channel min-max per clip erased
       any cross-clip loudness difference inside every feature.
   v3: power_to_db(ref=1.0) — absolute dB. No per-channel min-max.
       Z-score normalisation uses per-channel mean/std computed once
       from the TRAIN split and applied at load time (Dataset class
       below). Norm stats are written to norm_stats.json.

5. LESS AGGRESSIVE SILENCE TRIMMING
   top_db raised from 20 → 35. Choking has a lot of low-amplitude
   discriminative content (gasping, gurgling, strained breathing) that
   top_db=20 was cutting away.

6. NaN HANDLING
   Explicit np.nan_to_num after every pyin call.

7. SPECAUGMENT AS TRAINING-TIME UTILITY
   `SpecAugment` class — time + frequency masks applied in __getitem__.
   Not baked into saved tensors, so every epoch sees different masks.

8. AudioFeatureDataset
   Drop-in PyTorch Dataset that loads .npy, applies z-score, and
   optionally applies SpecAugment for training.

OUTPUT LAYOUT
─────────────
data/processed/
├── manifest.json           train/val/test file lists (source paths)
├── norm_stats.json         per-channel mean/std from train
├── meta.json               config snapshot + active channel order
├── train/<class>/*.npy     originals + augmentations
├── val/<class>/*.npy       originals only
└── test/<class>/*.npy      originals only
"""

import os
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import librosa.display
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    # ── Audio ────────────────────────────────────────────────
    "sample_rate":   22050,
    "duration":      3.0,
    "mono":          True,
    "trim_silence":  True,
    "top_db":        35,             # ↑ from 20 — preserve quiet choking content
    "peak_normalize": True,          # remove mic-gain confounds; relative dynamics preserved

    # ── Pre-emphasis (applied AFTER augmentation, BEFORE features) ──
    "preemphasis":   True,
    "preemph_coef":  0.97,

    # ── Spectrogram ──────────────────────────────────────────
    "n_mels":        128,
    "n_fft":         2048,
    "n_fft_wide":    512,
    "hop_length":    512,
    "fmin":          50,
    "fmax":          8000,

    # ── MFCCs / contrast ─────────────────────────────────────
    "n_mfcc":        20,
    "n_bands":       6,

    # ── Output tensor shape per channel ──────────────────────
    "img_height":    128,
    "img_width":     128,

    # ── Channels ─────────────────────────────────────────────
    "channels": {
        "log_mel":            True,
        "harmonic_mel":       True,
        "percussive_mel":     True,
        "delta":              True,
        "delta_delta":        True,
        "zcr":                True,
        "rms":                True,
        "spectral_centroid":  True,
        "spectral_rolloff":   True,
        "spectral_bandwidth": True,
        "spectral_contrast":  True,
        "chroma":             True,
        "mfcc":               True,
        "tonnetz":            True,
        "spectral_flatness":  True,
        "f0":                 True,
        "voiced_prob":        True,
        "spectral_flux":      True,
        "wide_band_mel":      True,
        "rms_periodicity":    True,
    },

    # ── Augmentation (TRAIN and Val only, on raw signal) ─────────────
    "augment": {
    "time_stretch_rates": [0.95, 1.05],
    "pitch_shift_steps":  [-0.5, 0.5],
    "noise_snr_db":       [30],
    "gain_db":            [-2, 2],

    "max_aug_per_source": {
        "train": 3,
        "val": 1,
        "test": 0
    }
},

    # ── Splits ───────────────────────────────────────────────
    "test_size":        0.25,
    "val_size":         0.15,                  # of full set
    "split_seed":       42,
    "stratify":         True,

    # ── Paths ────────────────────────────────────────────────
    "raw_dir":       "data/raw_prepared",
    "processed_dir": "data/processed",
    "spec_dir":      "data/spectrograms",
    "classes":       ["choking", "laughter"],
}

SUPPORTED_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════
def resize_2d(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bi-linear resize a 2-D feature map to (h, w)."""
    from skimage.transform import resize as sk_resize
    return sk_resize(arr, (h, w), anti_aliasing=True, preserve_range=True)


def broadcast_1d(row: np.ndarray, h: int, w: int) -> np.ndarray:
    """Expand (T,) or (1,T) to (h, w) by resizing to (1, w) and tiling rows."""
    row = row.flatten()
    row_resized = resize_2d(row[np.newaxis, :], 1, w).flatten()
    return np.tile(row_resized, (h, 1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Audio loading (NO pre-emphasis, NO per-clip min-max)
# ═══════════════════════════════════════════════════════════════
def load_audio(path: str, cfg: dict) -> np.ndarray:
    """Load → trim silence → fix length → peak-normalise. Returns raw signal."""
    y, _ = librosa.load(
        path,
        sr=cfg["sample_rate"],
        mono=cfg["mono"],
        duration=cfg["duration"] + 0.5,
    )

    if cfg["trim_silence"]:
        y, _ = librosa.effects.trim(y, top_db=cfg["top_db"])

    target = int(cfg["sample_rate"] * cfg["duration"])
    if len(y) > target:
        # centred crop — keeps the loudest middle portion
        start = (len(y) - target) // 2
        y = y[start:start + target]
    elif len(y) < target:
        # centred pad
        pad_total = target - len(y)
        pad_left = pad_total // 2
        y = np.pad(y, (pad_left, pad_total - pad_left))

    if cfg["peak_normalize"]:
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

    return y


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Augmentation (RAW signal, before pre-emphasis)
# ═══════════════════════════════════════════════════════════════
def _fix_length(sig: np.ndarray, target: int) -> np.ndarray:
    if len(sig) > target:
        start = (len(sig) - target) // 2
        sig = sig[start:start + target]
    elif len(sig) < target:
        pad = target - len(sig)
        sig = np.pad(sig, (pad // 2, pad - pad // 2))
    peak = np.max(np.abs(sig))
    return sig / peak if peak > 0 else sig


def augment_raw(y: np.ndarray, sr: int, target_len: int, aug_cfg: dict) -> list:
    """
    Returns a list of (tag, waveform) tuples — augmented variants of `y`.
    All variants are exactly `target_len` samples long and peak-normalised.
    Augmentation operates on the RAW signal; pre-emphasis is applied later.
    """
    variants = []

    # Time stretching — preserves pitch, changes duration
    for rate in aug_cfg["time_stretch_rates"]:
        try:
            y_ts = librosa.effects.time_stretch(y, rate=rate)
            variants.append((f"ts{rate:.2f}", _fix_length(y_ts, target_len)))
        except Exception:
            pass

    # Pitch shifting — preserves duration, changes pitch
    for st in aug_cfg["pitch_shift_steps"]:
        try:
            y_ps = librosa.effects.pitch_shift(y, sr=sr, n_steps=st)
            variants.append((f"ps{st:+d}", _fix_length(y_ps, target_len)))
        except Exception:
            pass

    # SNR-controlled additive Gaussian noise — physically meaningful level
    sig_power = np.mean(y ** 2) + 1e-10
    for snr_db in aug_cfg["noise_snr_db"]:
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        variants.append((f"n{snr_db}", _fix_length(y + noise, target_len)))

    # Random gain — preserves spectral shape but shifts loudness
    # (peak-normalisation will undo this, so we apply gain in dB AFTER peak-norm
    #  by scaling and then re-clipping; mainly affects relative dynamics of pad regions)
    for gain_db in aug_cfg["gain_db"]:
        g = 10 ** (gain_db / 20)
        y_g = np.clip(y * g, -1.0, 1.0)
        variants.append((f"g{gain_db:+d}", _fix_length(y_g, target_len)))

    return variants


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Feature extraction (NO per-channel min-max)
# ═══════════════════════════════════════════════════════════════
def _preemphasise(y: np.ndarray, cfg: dict) -> np.ndarray:
    if cfg["preemphasis"]:
        return librosa.effects.preemphasis(y, coef=cfg["preemph_coef"])
    return y


def extract_channels(y_raw: np.ndarray, cfg: dict) -> dict:
    """
    Returns {channel_name: (H, W) float32 array} WITHOUT per-channel
    normalisation. Values are raw dB / raw feature units. Z-score
    normalisation is applied later using train-set statistics.

    `y_raw` is the post-augmentation, pre-emphasis signal.
    """
    y = _preemphasise(y_raw, cfg)
    sr = cfg["sample_rate"]
    H, W = cfg["img_height"], cfg["img_width"]
    sel = cfg["channels"]
    out = {}

    # ── Harmonic / percussive separation ───────────────────────────
    y_harm, y_perc = librosa.effects.hpss(y)

    def mel_db(signal, n_fft):
        mel = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_mels=cfg["n_mels"],
            n_fft=n_fft, hop_length=cfg["hop_length"],
            fmin=cfg["fmin"], fmax=cfg["fmax"],
        )
        # ref=1.0 → absolute dB. Preserves cross-clip loudness differences.
        return librosa.power_to_db(mel, ref=1.0)

    log_mel = mel_db(y, cfg["n_fft"])

    if sel["log_mel"]:
        out["log_mel"] = resize_2d(log_mel, H, W)

    if sel["harmonic_mel"]:
        out["harmonic_mel"] = resize_2d(mel_db(y_harm, cfg["n_fft"]), H, W)

    if sel["percussive_mel"]:
        out["percussive_mel"] = resize_2d(mel_db(y_perc, cfg["n_fft"]), H, W)

    if sel["delta"]:
        out["delta"] = resize_2d(librosa.feature.delta(log_mel), H, W)

    if sel["delta_delta"]:
        out["delta_delta"] = resize_2d(librosa.feature.delta(log_mel, order=2), H, W)

    if sel["zcr"]:
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=cfg["n_fft"], hop_length=cfg["hop_length"]
        )
        out["zcr"] = broadcast_1d(zcr, H, W)

    if sel["rms"]:
        rms = librosa.feature.rms(
            y=y, frame_length=cfg["n_fft"], hop_length=cfg["hop_length"]
        )
        # log RMS preserves dynamic range better than linear
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)
        out["rms"] = broadcast_1d(rms_db, H, W)

    if sel["spectral_centroid"]:
        sc = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"]
        )
        out["spectral_centroid"] = broadcast_1d(sc, H, W)

    if sel["spectral_rolloff"]:
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"], roll_percent=0.85,
        )
        out["spectral_rolloff"] = broadcast_1d(rolloff, H, W)

    if sel["spectral_bandwidth"]:
        bw = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"]
        )
        out["spectral_bandwidth"] = broadcast_1d(bw, H, W)

    if sel["spectral_contrast"]:
        sc_feat = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
            n_bands=cfg["n_bands"], fmin=cfg["fmin"],
        )
        out["spectral_contrast"] = resize_2d(sc_feat, H, W)

    if sel["chroma"]:
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
        )
        out["chroma"] = resize_2d(chroma, H, W)

    if sel["mfcc"]:
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=cfg["n_mfcc"],
            n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
            n_mels=cfg["n_mels"],
        )
        out["mfcc"] = resize_2d(mfcc, H, W)

    if sel["tonnetz"]:
        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            tonnetz = np.nan_to_num(tonnetz, nan=0.0, posinf=0.0, neginf=0.0)
            out["tonnetz"] = resize_2d(tonnetz, H, W)
        except Exception:
            out["tonnetz"] = np.zeros((H, W), dtype=np.float32)

    if sel["spectral_flatness"]:
        flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"]
        )
        out["spectral_flatness"] = broadcast_1d(flatness, H, W)

    if sel["f0"] or sel["voiced_prob"]:
        try:
            f0, _, voiced_prob = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C6"),
                sr=sr, hop_length=cfg["hop_length"],
            )
            f0 = np.nan_to_num(f0, nan=0.0)
            voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)
        except Exception:
            n_frames = int(np.ceil(len(y) / cfg["hop_length"]))
            f0 = np.zeros(n_frames)
            voiced_prob = np.zeros(n_frames)

        if sel["f0"]:
            out["f0"] = broadcast_1d(f0, H, W)
        if sel["voiced_prob"]:
            out["voiced_prob"] = broadcast_1d(voiced_prob, H, W)

    if sel["spectral_flux"]:
        stft_mag = np.abs(librosa.stft(
            y, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"]
        ))
        col_sum = stft_mag.sum(axis=0, keepdims=True) + 1e-8
        stft_norm = stft_mag / col_sum
        flux = np.sum(np.abs(np.diff(stft_norm, axis=1, prepend=stft_norm[:, :1])), axis=0)
        out["spectral_flux"] = broadcast_1d(flux, H, W)

    if sel["wide_band_mel"]:
        out["wide_band_mel"] = resize_2d(mel_db(y, cfg["n_fft_wide"]), H, W)

    if sel["rms_periodicity"]:
        rms_frames = librosa.feature.rms(
            y=y, frame_length=cfg["n_fft"], hop_length=cfg["hop_length"]
        )[0]
        frames_mat = librosa.util.frame(
            y, frame_length=cfg["n_fft"], hop_length=cfg["hop_length"]
        )
        T = frames_mat.shape[1]
        periodicity = np.zeros(T, dtype=np.float32)
        for i in range(T):
            ac = librosa.autocorrelate(frames_mat[:, i])
            denom = ac[0] + 1e-6
            periodicity[i] = float(np.max(ac[20:]) / denom) if len(ac) > 20 else 0.0
        min_T = min(len(rms_frames), T)
        combined = rms_frames[:min_T] * periodicity[:min_T]
        out["rms_periodicity"] = broadcast_1d(combined, H, W)

    # Final NaN/Inf sweep — defensive
    for k in out:
        out[k] = np.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return out


def build_tensor(channel_dict: dict) -> np.ndarray:
    """Stack into (C, H, W)."""
    return np.stack([v for v in channel_dict.values()], axis=0).astype(np.float32)


def channel_names(cfg: dict) -> list:
    return [k for k, v in cfg["channels"].items() if v]


def process_clip(y_raw: np.ndarray, cfg: dict) -> np.ndarray:
    """Full pipeline for one waveform: features → tensor."""
    chs = extract_channels(y_raw, cfg)
    return build_tensor(chs)


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Split management
# ═══════════════════════════════════════════════════════════════
def get_source_stem(path):
    """
    Returns the original source recording stem.

    Examples:
    original.wav              -> original
    original__w0.wav          -> original
    original__w0__ts0.95.npy  -> original
    """
    stem = Path(path).stem
    return stem.split("__w")[0]


def build_split_manifest(cfg: dict) -> dict:
    """
    Stratified SOURCE-level split.

    Important:
    If raw files were created by prepare_raw.py, several files may come from
    the same original recording:

        source__w0.wav
        source__w1.wav
        source__w2.wav

    These must all stay in the same split to avoid leakage.
    """
    raw_dir = Path(cfg["raw_dir"])

    groups = defaultdict(list)

    for cls in cfg["classes"]:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            print(f"[WARN] {cls_dir} missing")
            continue

        for f in cls_dir.iterdir():
            if f.suffix.lower() in SUPPORTED_EXT:
                source_stem = get_source_stem(f)
                groups[(cls, source_stem)].append(str(f))

    if not groups:
        raise RuntimeError("No raw audio files found.")

    group_keys = list(groups.keys())
    group_labels = [cls for cls, _ in group_keys]

    stratify = group_labels if cfg["stratify"] else None

    # First split source-groups into trainval and test
    g_trainval, g_test = train_test_split(
        group_keys,
        test_size=cfg["test_size"],
        random_state=cfg["split_seed"],
        stratify=stratify,
    )

    # Then split validation out of trainval
    val_frac_of_remaining = cfg["val_size"] / (1.0 - cfg["test_size"])
    labels_trainval = [cls for cls, _ in g_trainval]
    stratify_tv = labels_trainval if cfg["stratify"] else None

    g_train, g_val = train_test_split(
        g_trainval,
        test_size=val_frac_of_remaining,
        random_state=cfg["split_seed"],
        stratify=stratify_tv,
    )

    def expand_group_keys(keys):
        items = []
        for key in keys:
            cls, _ = key
            for path in groups[key]:
                items.append((path, cls))
        return items

    manifest = {
        "split_seed": cfg["split_seed"],
        "test_size": cfg["test_size"],
        "val_size": cfg["val_size"],

        "train": expand_group_keys(g_train),
        "val": expand_group_keys(g_val),
        "test": expand_group_keys(g_test),
    }

    counts = {
        split: {
            cls: sum(1 for _, label in manifest[split] if label == cls)
            for cls in cfg["classes"]
        }
        for split in ("train", "val", "test")
    }

    source_counts = {
        "train": {
            cls: sum(1 for key in g_train if key[0] == cls)
            for cls in cfg["classes"]
        },
        "val": {
            cls: sum(1 for key in g_val if key[0] == cls)
            for cls in cfg["classes"]
        },
        "test": {
            cls: sum(1 for key in g_test if key[0] == cls)
            for cls in cfg["classes"]
        },
    }

    manifest["counts"] = counts
    manifest["source_counts"] = source_counts

    return manifest


def save_manifest(manifest: dict, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Per-channel normalisation statistics (from TRAIN only)
# ═══════════════════════════════════════════════════════════════
def compute_train_stats(train_dir: Path, n_channels: int) -> dict:
    """
    Per-channel mean and std computed across ALL train tensors.
    Used at training time (and val/test) for z-score normalisation.
    """
    files = sorted(train_dir.rglob("*.npy"))
    if not files:
        raise RuntimeError(f"No train tensors in {train_dir}")

    # Streaming sum / sum-of-squares to avoid loading everything at once
    n = np.zeros(n_channels, dtype=np.float64)
    s = np.zeros(n_channels, dtype=np.float64)
    sq = np.zeros(n_channels, dtype=np.float64)

    for fp in tqdm(files, desc="  computing train stats"):
        t = np.load(str(fp))   # (C, H, W)
        # per-channel sums
        n  += t.shape[1] * t.shape[2]
        s  += t.sum(axis=(1, 2))
        sq += (t.astype(np.float64) ** 2).sum(axis=(1, 2))

    mean = s / n
    var  = sq / n - mean ** 2
    std  = np.sqrt(np.maximum(var, 1e-8))

    return {"mean": mean.tolist(), "std": std.tolist(), "n_samples": len(files)}


def save_norm_stats(stats: dict, channels: list, processed_dir: Path) -> None:
    payload = {
        "channels": channels,
        "mean": stats["mean"],
        "std":  stats["std"],
        "n_train_files": stats["n_samples"],
    }
    with open(processed_dir / "norm_stats.json", "w") as f:
        json.dump(payload, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# STEP 6 — SpecAugment (TRAINING-TIME utility)
# ═══════════════════════════════════════════════════════════════
class SpecAugment:
    """
    Frequency + time masking, applied to a (C, H, W) tensor.
    Use in your Dataset.__getitem__ for train split only.

    Standard SpecAugment (Park et al. 2019). Each epoch sees different
    masks — this is the entire reason to do it on-the-fly, not in
    preprocessing.

    Args:
        freq_mask_param: max width of each frequency mask (in H units).
        time_mask_param: max width of each time mask (in W units).
        n_freq_masks:    number of frequency masks per sample.
        n_time_masks:    number of time masks per sample.
        replace_value:   value to fill masks with. 0 after z-score is fine.
    """
    def __init__(self,
                 freq_mask_param: int = 16,
                 time_mask_param: int = 16,
                 n_freq_masks: int = 2,
                 n_time_masks: int = 2,
                 replace_value: float = 0.0,
                 prob: float = 0.5):
        self.F = freq_mask_param
        self.T = time_mask_param
        self.nF = n_freq_masks
        self.nT = n_time_masks
        self.v = replace_value
        self.prob = prob

    def __call__(self, tensor: np.ndarray) -> np.ndarray:
        if random.random() > self.prob:
            return tensor
        out = tensor.copy()
        _, H, W = out.shape

        for _ in range(self.nF):
            f = random.randint(0, self.F)
            if f == 0 or f >= H:
                continue
            f0 = random.randint(0, H - f)
            out[:, f0:f0 + f, :] = self.v

        for _ in range(self.nT):
            t = random.randint(0, self.T)
            if t == 0 or t >= W:
                continue
            t0 = random.randint(0, W - t)
            out[:, :, t0:t0 + t] = self.v

        return out


# ═══════════════════════════════════════════════════════════════
# STEP 7 — PyTorch Dataset (loads + normalises + optional SpecAug)
# ═══════════════════════════════════════════════════════════════
class AudioFeatureDataset:
    """
    Loads .npy tensors and applies z-score using train stats.
    Pass `spec_augment` (SpecAugment instance) only for the TRAIN split.

    Works as a plain Python iterable; if torch is available it can be
    used as a torch.utils.data.Dataset (compatible __len__ / __getitem__).

    Args:
        split_dir:    e.g. data/processed/train
        classes:      list of class names (folder names under split_dir)
        norm_stats:   dict with "mean" and "std" (lists of length C)
        spec_augment: optional SpecAugment instance (training only)
    """
    def __init__(self,
                 split_dir,
                 classes: list,
                 norm_stats: dict,
                 spec_augment: Optional[SpecAugment] = None):
        self.split_dir = Path(split_dir)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.mean = np.array(norm_stats["mean"], dtype=np.float32)[:, None, None]
        self.std  = np.array(norm_stats["std"],  dtype=np.float32)[:, None, None]
        self.spec_augment = spec_augment

        self.items = []   # (path, label_idx)
        for cls in classes:
            cls_dir = self.split_dir / cls
            if not cls_dir.exists():
                continue
            for fp in sorted(cls_dir.glob("*.npy")):
                self.items.append((str(fp), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        x = np.load(path).astype(np.float32)        # (C, H, W)
        x = (x - self.mean) / self.std              # per-channel z-score
        if self.spec_augment is not None:
            x = self.spec_augment(x)
        return x, label


# ═══════════════════════════════════════════════════════════════
# STEP 8 — Preview PNGs (optional)
# ═══════════════════════════════════════════════════════════════
CHANNEL_META = {
    "log_mel":            ("magma",    "Log-Mel"),
    "harmonic_mel":       ("magma",    "Harmonic Mel"),
    "percussive_mel":     ("magma",    "Percussive Mel"),
    "delta":              ("coolwarm", "Delta"),
    "delta_delta":        ("coolwarm", "Delta-Delta"),
    "zcr":                ("plasma",   "ZCR"),
    "rms":                ("plasma",   "RMS (dB)"),
    "spectral_centroid":  ("viridis",  "Spec Centroid"),
    "spectral_rolloff":   ("viridis",  "Spec Rolloff"),
    "spectral_bandwidth": ("viridis",  "Spec Bandwidth"),
    "spectral_contrast":  ("inferno",  "Spec Contrast"),
    "chroma":             ("PiYG",     "Chroma"),
    "mfcc":               ("RdBu_r",   "MFCCs"),
    "tonnetz":            ("seismic",  "Tonnetz"),
    "spectral_flatness":  ("YlOrRd",   "Spec Flatness"),
    "f0":                 ("cool",     "F0"),
    "voiced_prob":        ("cool",     "Voiced Prob"),
    "spectral_flux":      ("hot",      "Spec Flux"),
    "wide_band_mel":      ("magma",    "Wide-Band Mel"),
    "rms_periodicity":    ("plasma",   "RMS × Periodicity"),
}


def save_preview(channel_dict: dict, save_dir: str, stem: str):
    n = len(channel_dict)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    axes = axes.flatten()
    last_i = 0
    for i, (name, arr) in enumerate(channel_dict.items()):
        cmap, title = CHANNEL_META.get(name, ("gray", name))
        axes[i].imshow(arr, aspect="auto", origin="lower", cmap=cmap)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")
        last_i = i
    for j in range(last_i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle(stem, fontsize=10, y=1.01)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, stem + ".png"), bbox_inches="tight", dpi=90)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
def _process_one_file(src_path: str, cls: str, out_dir: Path, cfg: dict,
                      augment: bool, save_previews: bool, spec_dir: Path) -> int:
    """Process one source file. Returns number of .npy tensors written."""
    target_len = int(cfg["sample_rate"] * cfg["duration"])
    stem = Path(src_path).stem
    written = 0

    y = load_audio(src_path, cfg)

    tensor = process_clip(y, cfg)
    np.save(str(out_dir / cls / f"{stem}.npy"), tensor)
    written += 1

    if save_previews:
        chs = extract_channels(y, cfg)
        save_preview(chs, str(spec_dir / cls), stem)

    if augment:
        variants = augment_raw(
            y,
            cfg["sample_rate"],
            target_len,
            cfg["augment"]
        )

        max_aug = cfg["augment"]["max_aug_per_source"].get(
            out_dir.name,
            len(variants)
        )

        if max_aug < len(variants):
            variants = random.sample(variants, max_aug)

        for tag, y_aug in variants:
            t_aug = process_clip(y_aug, cfg)
            np.save(str(out_dir / cls / f"{stem}__{tag}.npy"), t_aug)
            written += 1

    return written


def run_pipeline(cfg: dict, save_previews: bool = True) -> None:
    set_seed(cfg["split_seed"])
    processed_dir = Path(cfg["processed_dir"])
    spec_dir = Path(cfg["spec_dir"])
    active_chs = channel_names(cfg)

    print(f"\n  Active channels ({len(active_chs)}): {', '.join(active_chs)}")

    # ── 1. Build / save manifest ──────────────────────────────
    print(f"\n{'═'*60}\n  BUILDING SPLIT MANIFEST\n{'═'*60}")
    manifest = build_split_manifest(cfg)
    save_manifest(manifest, processed_dir)
    for split, counts in manifest["counts"].items():
        total = sum(counts.values())
        per_class = ", ".join(f"{c}={n}" for c, n in counts.items())
        print(f"  {split:6s}  {total:4d} files   ({per_class})")

    # ── 2. Process each split ─────────────────────────────────
    for split in ("train", "val", "test"):
        split_dir = processed_dir / split
        split_spec_dir = spec_dir / split
        for cls in cfg["classes"]:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)

        max_aug = cfg["augment"]["max_aug_per_source"].get(split, 0) 
        do_aug = max_aug > 0
        items = manifest[split]
        print(f"\n{'─'*60}")
        print(f"  PROCESSING {split.upper():5s}  "
              f"({len(items)} source files, augment={do_aug})")
        print(f"{'─'*60}")

        total_written, errors = 0, 0
        for src_path, cls in tqdm(items, desc=f"  {split}"):
            try:
                total_written += _process_one_file(
                    src_path, cls, split_dir, cfg,
                    augment=do_aug,
                    save_previews=save_previews,
                    spec_dir=split_spec_dir,
                )
            except Exception as e:
                print(f"\n  [ERR] {Path(src_path).name}: {e}")
                errors += 1
        print(f"  {split}: {total_written} tensors written, {errors} errors")

    # ── 3. Compute train-set normalisation stats ──────────────
    print(f"\n{'═'*60}\n  COMPUTING NORM STATS (from TRAIN only)\n{'═'*60}")
    stats = compute_train_stats(processed_dir / "train", len(active_chs))
    save_norm_stats(stats, active_chs, processed_dir)
    print(f"  per-channel mean range: [{min(stats['mean']):.3f}, "
          f"{max(stats['mean']):.3f}]")
    print(f"  per-channel std  range: [{min(stats['std']):.3f}, "
          f"{max(stats['std']):.3f}]")

    # ── 4. Save meta ──────────────────────────────────────────
    meta = {
        "config": cfg,
        "channels": active_chs,
        "n_channels": len(active_chs),
        "tensor_shape": [len(active_chs), cfg["img_height"], cfg["img_width"]],
        "note": "tensors are UNnormalised; apply norm_stats.json at load time",
    }
    with open(processed_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'═'*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'═'*60}")
    print(f"  CNN input shape: ({len(active_chs)}, "
          f"{cfg['img_height']}, {cfg['img_width']})")
    print(f"  Train tensors include originals + augmentations.")
    print(f"  Val/Test tensors are originals only — no leakage.")
    print(f"  Use AudioFeatureDataset + SpecAugment for training-time aug.")


# ═══════════════════════════════════════════════════════════════
# VERIFY
# ═══════════════════════════════════════════════════════════════
def verify(cfg: dict) -> None:
    print("\n── Verification ─────────────────────────────────────────")
    processed = Path(cfg["processed_dir"])
    for split in ("train", "val", "test"):
        for cls in cfg["classes"]:
            files = list((processed / split / cls).glob("*.npy"))
            if not files:
                print(f"  {split:5s} / {cls:10s}: no files")
                continue
            s = np.load(str(files[0]))
            print(f"  {split:5s} / {cls:10s}  {len(files):4d} samples | "
                  f"shape {tuple(s.shape)} | "
                  f"range [{s.min():.2f}, {s.max():.2f}]")

    # Sanity: no source-stem appears in more than one split
    print("\n  Leakage check (source stems across splits):")
    stems_by_split = {}
    for split in ("train", "val", "test"):
        stems = set()
        for cls in cfg["classes"]:
            for fp in (processed / split / cls).glob("*.npy"):
                # strip augmentation suffix to get source stem
                src_stem = fp.stem.split("__")[0]
                stems.add(src_stem)
        stems_by_split[split] = stems
        print(f"    {split}: {len(stems)} unique source stems")

    train_val = stems_by_split["train"] & stems_by_split["val"]
    train_test = stems_by_split["train"] & stems_by_split["test"]
    val_test = stems_by_split["val"] & stems_by_split["test"]

    if train_val or train_test or val_test:
        print("  ❌ LEAKAGE DETECTED:")
        if train_val:  print(f"    train ∩ val  = {len(train_val)} stems")
        if train_test: print(f"    train ∩ test = {len(train_test)} stems")
        if val_test:   print(f"    val ∩ test   = {len(val_test)} stems")
    else:
        print("  ✓ No source-stem leakage across splits.")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-previews", action="store_true",
                    help="skip PNG previews (faster)")
    ap.add_argument("--verify-only", action="store_true",
                    help="just verify an existing data/processed tree")
    args = ap.parse_args()

    if args.verify_only:
        verify(CONFIG)
    else:
        run_pipeline(CONFIG, save_previews=not args.no_previews)
        verify(CONFIG)
