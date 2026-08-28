"""
Raw Audio Preparation — Stage 0 (runs BEFORE feature extraction)
════════════════════════════════════════════════════════════════
Choking vs. Laughing — noise-floor matching + windowing

WHY THIS EXISTS
───────────────
Analysis revealed a dataset confound: choking clips were all self-recorded
on one mic (noise floor ≈ -54 dB) while laughter clips came from mixed,
cleaner sources (noise floor ≈ -72 to -83 dB). A model can exploit this —
learning "high noise floor = choking" instead of the actual acoustics.
This collapses on any new recording.

This stage equalises the two groups' noise floors and, while we're touching
the raw audio, slices long recordings into multiple windows to grow the set.

WHAT IT DOES (in order, per file)
─────────────────────────────────
1. Load + resample to a single unified rate (16 kHz — the common rate of
   both groups; no point upsampling).
2. DC-offset removal.
3. High-pass filter @ 40 Hz (remove sub-sonic rumble — no useful content).
4. TARGET-AWARE denoise (choking only by default): spectral gating tuned so
   the resulting noise floor lands near the laughter group's level
   (TARGET_NOISE_FLOOR_DB) — NOT cleaner, to avoid an inverted confound.
   Files already at/below target are left alone.
5. Slice into fixed-length windows (WINDOW_SEC) with small overlap
   (OVERLAP — 25%) to use the material well without near-duplicates.
6. Drop silent / contentless windows (windows whose active-energy fraction
   is too low never enter the dataset).
7. Peak-normalise each window, write to the prepared raw tree.

OUTPUT
──────
data/raw_prepared/<class>/<sourcestem>__w0.wav, __w1.wav, ...
  The "<sourcestem>__wN" naming lets the split stage keep all windows from
  one source recording in the SAME split (no leakage).
prep_report.json   — per-file before/after noise floor + window counts.

CHAINING WITH audio_preprocessing_v3.py
───────────────────────────────────────
After this runs, point v3 at data/raw_prepared and use the SLIM config:
windows are already fixed-length, denoised, and silence-screened, so v3
should SKIP trim_silence and length-fixing (set them off in CONFIG) to avoid
double-processing. v3 then only does: load → augment(train) → pre-emphasis
→ extract channels.
"""

import os
import json
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import scipy.signal as sps
import noisereduce as nr
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
PREP = {
    # ── I/O ──────────────────────────────────────────────────
    "raw_dir":        "data/raw",            # input: data/raw/<class>/*.wav
    "out_dir":        "data/raw_prepared",   # output windows
    "classes":        ["choking", "laughter"],

    # ── Unified format ───────────────────────────────────────
    "sample_rate":    16000,

    # ── Cleanup ──────────────────────────────────────────────
    "remove_dc":      True,
    "highpass_hz":    40,

    # ── Target-aware denoise ─────────────────────────────────
    # Only classes in `denoise_classes` get denoised. We push their noise
    # floor toward TARGET (the cleaner group's level) but not past it.
    "denoise_classes":        ["choking"],   # laughter is already clean
    "target_noise_floor_db":  -72.0,         # laughter group sits ~ -72..-83
    "denoise_tolerance_db":   3.0,           # don't bother if already within this
    "denoise_max_prop":       0.9,           # ceiling on aggressiveness
    "denoise_min_prop":       0.3,           # floor (if any denoise is applied)

    # ── Windowing ────────────────────────────────────────────
    "window_sec":     3.0,
    "overlap":        0.25,                  # 25% overlap → hop = 2.25 s
    "pad_last":       True,                  # pad final short window if it has content

    # ── (א) Energy-aware windowing ───────────────────────────
    # Instead of slicing blindly at fixed hops, center windows on energy
    # peaks (where the actual events are). Analysis showed active fraction
    # varies a lot across recordings (42%–99%); blind slicing wastes windows
    # on sparse regions. When True, we detect onset/energy peaks and place
    # windows around them, falling back to fixed slicing if too few peaks.
    "energy_aware":           True,
    "peak_min_separation_sec": 1.5,          # min gap between detected event centers
    "min_peaks_for_aware":     2,            # below this, fall back to fixed slicing

    # ── (ב) Silence screening (stricter) ─────────────────────
    "active_top_db":      30,                # frames within 30 dB of window peak = "active"
    "min_active_frac":    0.22,              # ↑ from 0.10 — windows must carry real content
    "min_window_peak_db": -45.0,            # also drop windows whose peak is below this

    # ── (ג) Normalisation method ─────────────────────────────
    # "peak"  = scale so max |sample| = 1 (sensitive to single spikes)
    # "rms"   = scale to a target RMS (more stable across windows)
    "norm_method":    "rms",
    "target_rms":     0.1,                   # target RMS for "rms" mode
    "rms_peak_guard": 0.99,                  # after RMS norm, clip if peak would exceed this

    # ── (ה) Per-window metadata ──────────────────────────────
    "save_metadata":  True,                  # write windows_metadata.json
}


# ═══════════════════════════════════════════════════════════════
# Noise-floor measurement
# ═══════════════════════════════════════════════════════════════
def measure_noise_floor_db(y: np.ndarray, sr: int) -> float:
    """RMS of the quietest 10% of 100 ms frames, in dBFS."""
    fl = int(0.1 * sr)
    if fl < 1 or len(y) < fl:
        return -np.inf
    r = librosa.feature.rms(y=y, frame_length=fl, hop_length=fl)[0]
    if len(r) == 0:
        return -np.inf
    quiet = np.sort(r)[:max(1, len(r) // 10)]
    return float(20 * np.log10(np.sqrt(np.mean(quiet ** 2)) + 1e-12))


def measure_signal_db(y: np.ndarray, sr: int) -> float:
    """RMS of the loudest 10% of frames, in dBFS."""
    fl = int(0.1 * sr)
    if fl < 1 or len(y) < fl:
        return -np.inf
    r = librosa.feature.rms(y=y, frame_length=fl, hop_length=fl)[0]
    loud = np.sort(r)[-max(1, len(r) // 10):]
    return float(20 * np.log10(np.sqrt(np.mean(loud ** 2)) + 1e-12))


# ═══════════════════════════════════════════════════════════════
# Cleanup primitives
# ═══════════════════════════════════════════════════════════════
def remove_dc(y: np.ndarray) -> np.ndarray:
    return y - np.mean(y)


def highpass(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    sos = sps.butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sps.sosfilt(sos, y).astype(np.float32)


def learn_noise_clip(y: np.ndarray, sr: int) -> np.ndarray:
    """Concatenate the quietest frames to use as the noise profile."""
    fl = int(0.1 * sr)
    r = librosa.feature.rms(y=y, frame_length=fl, hop_length=fl)[0]
    quiet_frames = np.argsort(r)[:15]
    return np.concatenate([y[qf * fl: qf * fl + fl] for qf in quiet_frames])


def target_aware_denoise(y: np.ndarray, sr: int, cfg: dict) -> tuple:
    """
    Denoise toward cfg['target_noise_floor_db'] without overshooting.
    Returns (cleaned_signal, applied_prop_decrease, nf_before, nf_after).

    Strategy: measure current noise floor; if already within tolerance of
    target (or quieter), skip. Otherwise pick a prop_decrease proportional to
    how far above target we are, clamped to [min_prop, max_prop], denoise,
    then verify.
    """
    nf_before = measure_noise_floor_db(y, sr)
    target = cfg["target_noise_floor_db"]
    tol = cfg["denoise_tolerance_db"]

    # Already clean enough → leave it
    if nf_before <= target + tol:
        return y, 0.0, nf_before, nf_before

    # Map distance-above-target → aggressiveness.
    # ~15 dB over target → max aggressiveness; scale linearly down to min.
    gap = nf_before - target                      # positive
    frac = np.clip(gap / 15.0, 0.0, 1.0)
    prop = cfg["denoise_min_prop"] + frac * (cfg["denoise_max_prop"] - cfg["denoise_min_prop"])

    noise_clip = learn_noise_clip(y, sr)
    cleaned = nr.reduce_noise(
        y=y, sr=sr, y_noise=noise_clip,
        prop_decrease=float(prop), stationary=True,
    ).astype(np.float32)

    nf_after = measure_noise_floor_db(cleaned, sr)
    return cleaned, float(prop), nf_before, nf_after


# ═══════════════════════════════════════════════════════════════
# Windowing + silence screening
# ═══════════════════════════════════════════════════════════════
def window_is_active(win: np.ndarray, sr: int, cfg: dict) -> bool:
    """True if the window carries enough real content to keep."""
    peak = np.max(np.abs(win))
    if peak <= 0:
        return False
    peak_db = 20 * np.log10(peak + 1e-12)
    if peak_db < cfg["min_window_peak_db"]:
        return False

    # Fraction of frames within active_top_db of the window's own peak
    fl = int(0.05 * sr)
    if len(win) < fl:
        return False
    r = librosa.feature.rms(y=win, frame_length=fl, hop_length=fl)[0]
    r_db = 20 * np.log10(r + 1e-12)
    active = r_db > (r_db.max() - cfg["active_top_db"])
    return active.mean() >= cfg["min_active_frac"]


def _active_fraction(win: np.ndarray, sr: int, cfg: dict) -> float:
    """Fraction of frames within active_top_db of the window's own peak."""
    fl = int(0.05 * sr)
    if len(win) < fl:
        return 0.0
    r = librosa.feature.rms(y=win, frame_length=fl, hop_length=fl)[0]
    r_db = 20 * np.log10(r + 1e-12)
    return float((r_db > (r_db.max() - cfg["active_top_db"])).mean())


def _detect_event_centers(y: np.ndarray, sr: int, cfg: dict) -> list:
    """
    (א) Find energy-peak sample positions to center windows on.
    Uses an onset-strength envelope; picks peaks separated by at least
    peak_min_separation_sec. Returns a list of center sample indices.
    """
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if env.size == 0:
        return []
    min_sep_frames = max(1, int(cfg["peak_min_separation_sec"] * sr / hop))
    # Height threshold: above the median so we sit on real events, not noise
    thresh = np.median(env) + 0.5 * (env.max() - np.median(env))
    peaks = librosa.util.peak_pick(
        env, pre_max=min_sep_frames, post_max=min_sep_frames,
        pre_avg=min_sep_frames, post_avg=min_sep_frames,
        delta=0.0, wait=min_sep_frames,
    )
    peaks = [p for p in peaks if env[p] >= thresh]
    return [int(p * hop) for p in peaks]


def slice_windows(y: np.ndarray, sr: int, cfg: dict) -> list:
    """
    Slice into fixed-length windows. Returns a list of dicts:
        {"audio": np.ndarray (win_len,), "offset_sec": float,
         "active_frac": float, "mode": "fixed"|"aware"}
    Only windows passing the activity screen are returned.

    If cfg['energy_aware'] and enough event peaks are detected, windows are
    centered on those peaks; otherwise falls back to fixed-hop slicing.
    """
    win_len = int(cfg["window_sec"] * sr)
    n = len(y)
    results = []

    def _try_add(start, mode):
        start = max(0, min(start, max(0, n - win_len)))
        seg = y[start:start + win_len]
        if len(seg) < win_len:
            seg = np.pad(seg, (0, win_len - len(seg)))
        if not window_is_active(seg, sr, cfg):
            return
        results.append({
            "audio": _normalise(seg, cfg),
            "offset_sec": round(start / sr, 3),
            "active_frac": round(_active_fraction(seg, sr, cfg), 3),
            "mode": mode,
        })

    # Very short recording → single padded window
    if n < win_len:
        if cfg["pad_last"]:
            pad = win_len - n
            seg = np.pad(y, (pad // 2, pad - pad // 2))
            if window_is_active(seg, sr, cfg):
                results.append({
                    "audio": _normalise(seg, cfg),
                    "offset_sec": 0.0,
                    "active_frac": round(_active_fraction(seg, sr, cfg), 3),
                    "mode": "pad",
                })
        return results

    # (א) Energy-aware path
    if cfg.get("energy_aware"):
        centers = _detect_event_centers(y, sr, cfg)
        if len(centers) >= cfg["min_peaks_for_aware"]:
            used = []
            for c in centers:
                start = c - win_len // 2
                # avoid near-duplicate windows (centers closer than half a window)
                if used and abs(start - used[-1]) < win_len * (1.0 - cfg["overlap"]):
                    continue
                _try_add(start, "aware")
                used.append(start)
            # If aware slicing yielded too little, also fill gaps with fixed slicing
            if len(results) >= cfg["min_peaks_for_aware"]:
                return results

    # Fixed-hop fallback
    hop = int(win_len * (1.0 - cfg["overlap"])) or win_len
    start = 0
    while start + win_len <= n:
        _try_add(start, "fixed")
        start += hop
    if cfg["pad_last"] and start < n:
        remainder_len = n - start
        if remainder_len > win_len * 0.5:
            _try_add(start, "fixed")

    return results


# ── (ג) Flexible normalisation ───────────────────────────────
def _normalise(w: np.ndarray, cfg: dict) -> np.ndarray:
    method = cfg.get("norm_method", "peak")
    if method == "rms":
        rms = np.sqrt(np.mean(w ** 2)) + 1e-9
        out = w * (cfg["target_rms"] / rms)
        # guard against clipping after RMS scaling
        peak = np.max(np.abs(out))
        guard = cfg.get("rms_peak_guard", 0.99)
        if peak > guard:
            out = out * (guard / peak)
        return out.astype(np.float32)
    # default: peak
    p = np.max(np.abs(w))
    return (w / p).astype(np.float32) if p > 0 else w.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
SUPPORTED = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


def prepare(cfg: dict) -> None:
    raw_dir = Path(cfg["raw_dir"])
    out_dir = Path(cfg["out_dir"])
    sr = cfg["sample_rate"]

    report = {"config": cfg, "files": [], "summary": {}}
    nf_by_class = {c: {"before": [], "after": []} for c in cfg["classes"]}
    window_counts = {c: 0 for c in cfg["classes"]}
    source_counts = {c: 0 for c in cfg["classes"]}
    all_window_meta = []   # (ה) per-window metadata

    for cls in cfg["classes"]:
        cls_in = raw_dir / cls
        cls_out = out_dir / cls
        cls_out.mkdir(parents=True, exist_ok=True)

        if not cls_in.exists():
            print(f"[WARN] {cls_in} missing — skipping")
            continue

        files = [f for f in sorted(cls_in.iterdir()) if f.suffix.lower() in SUPPORTED]
        do_denoise = cls in cfg["denoise_classes"]

        print(f"\n{'─'*64}")
        print(f"  {cls.upper()}  ({len(files)} source files, denoise={do_denoise})")
        print(f"{'─'*64}")

        for fp in tqdm(files, desc=f"  {cls}"):
            try:
                y, _ = librosa.load(str(fp), sr=sr, mono=True)
            except Exception as e:
                print(f"\n  [ERR] load {fp.name}: {e}")
                continue

            if cfg["remove_dc"]:
                y = remove_dc(y)
            if cfg["highpass_hz"]:
                y = highpass(y, sr, cfg["highpass_hz"])

            applied_prop, nf_b, nf_a = 0.0, measure_noise_floor_db(y, sr), None
            if do_denoise:
                y, applied_prop, nf_b, nf_a = target_aware_denoise(y, sr, cfg)
            else:
                nf_a = nf_b

            nf_by_class[cls]["before"].append(nf_b)
            nf_by_class[cls]["after"].append(nf_a)

            # Slice into windows (each is a dict with audio + metadata)
            windows = slice_windows(y, sr, cfg)
            stem = fp.stem
            for i, w in enumerate(windows):
                fname = f"{stem}__w{i}.wav"
                out_path = cls_out / fname
                sf.write(str(out_path), w["audio"], sr)
                all_window_meta.append({
                    "window_file": str(out_path),
                    "filename": fname,
                    "source_stem": stem,           # ← key for source-level splitting
                    "source": str(fp),
                    "class": cls,
                    "window_index": i,
                    "offset_sec": w["offset_sec"],
                    "active_frac": w["active_frac"],
                    "slice_mode": w["mode"],
                })

            window_counts[cls] += len(windows)
            source_counts[cls] += 1

            report["files"].append({
                "source": str(fp),
                "class": cls,
                "noise_floor_before": round(nf_b, 2),
                "noise_floor_after": round(nf_a, 2),
                "denoise_prop": round(applied_prop, 3),
                "n_windows": len(windows),
            })

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'═'*64}\n  SUMMARY\n{'═'*64}")
    for cls in cfg["classes"]:
        nb = nf_by_class[cls]["before"]
        na = nf_by_class[cls]["after"]
        if not nb:
            continue
        print(f"\n  {cls}:")
        print(f"    source files:        {source_counts[cls]}")
        print(f"    windows produced:    {window_counts[cls]}  "
              f"(×{window_counts[cls]/max(1,source_counts[cls]):.1f} per source)")
        print(f"    noise floor before:  mean {np.mean(nb):6.1f} dB   "
              f"[{np.min(nb):.1f}, {np.max(nb):.1f}]")
        print(f"    noise floor after:   mean {np.mean(na):6.1f} dB   "
              f"[{np.min(na):.1f}, {np.max(na):.1f}]")

    # Did the groups converge?
    after_means = {c: float(np.mean(nf_by_class[c]["after"]))
                   for c in cfg["classes"] if nf_by_class[c]["after"]}
    if len(after_means) == 2:
        vals = list(after_means.values())
        gap = abs(vals[0] - vals[1])
        print(f"\n  ▶ Noise-floor gap between classes AFTER prep: {gap:.1f} dB")
        if gap < 6:
            print(f"    ✓ Groups converged — confound largely removed.")
        else:
            print(f"    ⚠ Still {gap:.1f} dB apart — consider adjusting target.")

    report["summary"] = {
        "window_counts": window_counts,
        "source_counts": source_counts,
        "noise_floor_after_mean": after_means,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "prep_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # (ה) Per-window metadata
    if cfg.get("save_metadata"):
        with open(out_dir / "windows_metadata.json", "w", encoding="utf-8") as f:
            json.dump(all_window_meta, f, indent=2, ensure_ascii=False)

    # Slice-mode breakdown (how many windows came from energy-aware vs fixed)
    modes = {}
    for m in all_window_meta:
        modes[m["slice_mode"]] = modes.get(m["slice_mode"], 0) + 1
    if modes:
        mode_str = ", ".join(f"{k}={v}" for k, v in sorted(modes.items()))
        print(f"\n  Window slicing modes: {mode_str}")
        if all_window_meta:
            af = [m["active_frac"] for m in all_window_meta]
            print(f"  Active-fraction of kept windows: "
                  f"mean {np.mean(af):.2f}, min {np.min(af):.2f}")

    print(f"\n  Report → {out_dir / 'prep_report.json'}")
    if cfg.get("save_metadata"):
        print(f"  Metadata → {out_dir / 'windows_metadata.json'}")
    print(f"  Prepared windows → {out_dir}/<class>/<stem>__wN.wav")


# ═══════════════════════════════════════════════════════════════
# Single-file preview (for spot-checking before a full run)
# ═══════════════════════════════════════════════════════════════
def preview_file(path: str, cls: str, cfg: dict, save_png: str = None):
    """Process one file and (optionally) render a before/after figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    sr = cfg["sample_rate"]
    y0, _ = librosa.load(path, sr=sr, mono=True)
    y = remove_dc(y0) if cfg["remove_dc"] else y0.copy()
    if cfg["highpass_hz"]:
        y = highpass(y, sr, cfg["highpass_hz"])

    if cls in cfg["denoise_classes"]:
        y, prop, nf_b, nf_a = target_aware_denoise(y, sr, cfg)
    else:
        prop, nf_b, nf_a = 0.0, measure_noise_floor_db(y, sr), measure_noise_floor_db(y, sr)

    windows = slice_windows(y, sr, cfg)
    modes = {}
    for w in windows:
        modes[w["mode"]] = modes.get(w["mode"], 0) + 1
    mode_str = ", ".join(f"{k}={v}" for k, v in sorted(modes.items())) or "none"
    print(f"  {Path(path).name}: nf {nf_b:.1f} → {nf_a:.1f} dB "
          f"(prop={prop:.2f}), {len(windows)} windows kept [{mode_str}]")

    if save_png:
        def _disp_norm(x):
            p = np.max(np.abs(x)); return x/p if p > 0 else x
        yo = _disp_norm(y0); yc = _disp_norm(y)
        t = np.arange(len(yo)) / sr
        fig, ax = plt.subplots(2, 2, figsize=(15, 7))
        ax[0,0].plot(t, yo, lw=0.3, color='#c0392b'); ax[0,0].set_title('BEFORE waveform')
        ax[0,1].plot(np.arange(len(yc))/sr, yc, lw=0.3, color='#27ae60'); ax[0,1].set_title('AFTER waveform')
        for a in (ax[0,0], ax[0,1]): a.set_ylim(-1.05,1.05); a.grid(alpha=0.2)
        So = librosa.amplitude_to_db(np.abs(librosa.stft(yo, n_fft=1024, hop_length=256)), ref=np.max)
        Sc = librosa.amplitude_to_db(np.abs(librosa.stft(yc, n_fft=1024, hop_length=256)), ref=np.max)
        librosa.display.specshow(So, sr=sr, hop_length=256, x_axis='time', y_axis='hz', ax=ax[1,0], cmap='magma', vmin=-80, vmax=0)
        librosa.display.specshow(Sc, sr=sr, hop_length=256, x_axis='time', y_axis='hz', ax=ax[1,1], cmap='magma', vmin=-80, vmax=0)
        ax[1,0].set_title(f'BEFORE spectrogram (nf {nf_b:.1f} dB)')
        ax[1,1].set_title(f'AFTER spectrogram (nf {nf_a:.1f} dB)')
        plt.tight_layout(); plt.savefig(save_png, dpi=110, bbox_inches='tight'); plt.close()
        print(f"  figure → {save_png}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="path to a single file to preview")
    ap.add_argument("--class-name", default="choking", help="class of the preview file")
    ap.add_argument("--png", help="where to save the preview figure")
    args = ap.parse_args()

    if args.preview:
        preview_file(args.preview, args.class_name, PREP, save_png=args.png)
    else:
        prepare(PREP)
