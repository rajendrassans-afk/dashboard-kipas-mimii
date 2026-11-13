"""Ekstraksi fitur tingkat-klip: STFT -> (log-mel | MFCC + delta | fitur spektral) -> statistik."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from . import mathcore as M
from .data import load_wav, resample_to, EXPECTED_SR

FEATURE_SETS = {
    "MFCC + delta": "mfcc",
    "MFCC + delta + fitur spektral": "mfcc_spec",
    "Log-mel (fitur baseline MIMII)": "logmel",
}


@dataclass(frozen=True)
class FeatureConfig:
    n_fft: int = 1024        # baseline MIMII
    hop: int = 512           # baseline MIMII
    n_mels: int = 64         # baseline MIMII
    n_mfcc: int = 13
    feature_set: str = "mfcc_spec"

    def key(self) -> str:
        return hashlib.md5(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


def _stats(M2: np.ndarray, name: str):
    """mean & std tiap baris sepanjang waktu."""
    names = [f"{name}{i}_mean" for i in range(M2.shape[0])] + [f"{name}{i}_std" for i in range(M2.shape[0])]
    return np.concatenate([M2.mean(1), M2.std(1)]), names


def extract(y: np.ndarray, sr: int, cfg: FeatureConfig):
    """-> (vektor fitur, nama fitur)"""
    y = resample_to(np.asarray(y, dtype=np.float32), sr, EXPECTED_SR)
    parts, names = [], []
    if cfg.feature_set == "logmel":
        L = M.log_mel(y, EXPECTED_SR, cfg.n_fft, cfg.hop, cfg.n_mels)
        v, n = _stats(L, "mel"); parts.append(v); names += n
    else:
        C = M.mfcc(y, EXPECTED_SR, cfg.n_fft, cfg.hop, cfg.n_mels, cfg.n_mfcc)
        for arr, nm in ((C, "mfcc"), (M.delta(C), "d_mfcc")):
            v, n = _stats(arr, nm); parts.append(v); names += n
        if cfg.feature_set == "mfcc_spec":
            sp = M.spectral_features(y, EXPECTED_SR, cfg.n_fft, cfg.hop)
            for k, arr in sp.items():
                v, n = _stats(arr[None, :], k); parts.append(v); names += n
    return np.concatenate(parts), names


def extract_file(path: str, cfg: FeatureConfig):
    y, sr = load_wav(path)
    return extract(y, sr, cfg)


def build_matrix(paths, cfg: FeatureConfig, cache_dir="cache", progress=None):
    """Matriks fitur (n_klip, n_fitur) dengan cache .npy per file-konfigurasi."""
    cdir = Path(cache_dir); cdir.mkdir(exist_ok=True)
    X, names = [], None
    key = cfg.key()
    for i, p in enumerate(paths):
        cf = cdir / f"{hashlib.md5(p.encode()).hexdigest()[:12]}_{key}.npy"
        if cf.exists():
            v = np.load(cf)
        else:
            v, nm = extract_file(p, cfg); np.save(cf, v); names = names or nm
        X.append(v)
        if progress and (i % 20 == 0 or i == len(paths) - 1):
            progress((i + 1) / len(paths))
    if names is None:
        names = extract(np.zeros(EXPECTED_SR * 2, np.float32) + 1e-3, EXPECTED_SR, cfg)[1]
    return np.vstack(X), names
