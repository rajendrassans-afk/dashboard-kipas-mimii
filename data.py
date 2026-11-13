"""Pemindaian dataset MIMII, validasi, dan pemuatan audio."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

EXPECTED_SR = 16000          # MIMII: 16 kHz, 8 kanal -> kita pakai kanal 0 (sama seperti baseline)
EXPECTED_DUR = 10.0          # detik


def load_wav(src, channel: int = 0):
    """Baca WAV (path / file-like) -> (y float32 mono, sr). Kanal dipilih seperti demux_wav baseline."""
    y, sr = sf.read(src, dtype="float32", always_2d=True)
    return y[:, min(channel, y.shape[1] - 1)], sr


def validate_audio(y: np.ndarray, sr: int) -> list[tuple[str, str]]:
    """Daftar (level, pesan). level: ok | warn | err."""
    out = []
    if len(y) == 0:
        return [("err", "File kosong.")]
    if not np.isfinite(y).all():
        out.append(("err", "Terdapat nilai NaN/Inf."))
    if sr != EXPECTED_SR:
        out.append(("warn", f"Sampling rate {sr} Hz != {EXPECTED_SR} Hz (MIMII). Model dilatih pada {EXPECTED_SR} Hz; hasil dapat bias. Audio akan di-resample."))
    dur = len(y) / sr
    if dur < 1.0:
        out.append(("err", f"Durasi {dur:.2f}s terlalu pendek (minimal 1 s)."))
    elif abs(dur - EXPECTED_DUR) > 1.0:
        out.append(("warn", f"Durasi {dur:.1f}s berbeda dari {EXPECTED_DUR:.0f}s (MIMII)."))
    if np.max(np.abs(y)) < 1e-4:
        out.append(("warn", "Amplitudo hampir nol (sinyal sunyi?)."))
    if np.mean(np.abs(y) > 0.999) > 0.001:
        out.append(("warn", "Terdeteksi clipping (>0.1% sampel di batas)."))
    if not out:
        out.append(("ok", f"Format valid: {sr} Hz, {dur:.1f} s."))
    return out


def resample_to(y: np.ndarray, sr: int, target: int = EXPECTED_SR) -> np.ndarray:
    if sr == target:
        return y
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, target)
    return resample_poly(y, target // g, sr // g).astype(np.float32)


_ID = re.compile(r"id_(\d+)")
_SNR = re.compile(r"(min)?(\d+)_?dB", re.I)


def scan_dataset(root: str, machine: str = "fan") -> pd.DataFrame:
    """Cari semua .../normal/*.wav dan .../abnormal/*.wav di bawah root.

    Mendukung struktur resmi MIMII: <snr>_dB_fan/fan/id_00/{normal,abnormal}/*.wav
    Kolom: path, label (0 normal / 1 abnormal), kondisi, id, snr
    """
    rows = []
    rootp = Path(root).expanduser()
    for lab, name in ((0, "normal"), (1, "abnormal")):
        for p in rootp.rglob(f"{name}/*.wav"):
            s = p.as_posix()
            if machine and machine.lower() not in s.lower():
                continue
            mid = _ID.search(s)
            snr = None
            for part in p.parts:
                m = _SNR.search(part)
                if m:
                    snr = -int(m.group(2)) if m.group(1) else int(m.group(2)); break
            rows.append(dict(path=s, label=lab, kondisi=name, id=f"id_{mid.group(1)}" if mid else "id_?", snr=snr))
    return pd.DataFrame(rows, columns=["path", "label", "kondisi", "id", "snr"]).sort_values("path").reset_index(drop=True)
