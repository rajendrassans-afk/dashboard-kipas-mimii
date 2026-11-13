"""Sinyal kipas sintetis (untuk demo konsep & pengujian saja, bukan data MIMII)."""
import numpy as np

def fan(rng, kind="normal", sr=16000, dur=10.0, f0=None):
    t = np.arange(int(sr * dur)) / sr
    f0 = f0 or rng.uniform(45, 55)
    y = sum((0.6 / h) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28)) for h in range(1, 8))
    noise = np.convolve(rng.standard_normal(len(t)), np.ones(8) / 8, "same")
    y = y + 0.5 * noise
    if kind == "imbalance":
        y += 1.2 * np.sin(2 * np.pi * f0 * t) * (1 + 0.4 * np.sin(2 * np.pi * 3 * t))
    elif kind == "rattle":
        for c in rng.uniform(0, dur, rng.integers(10, 30)):
            i = int(c * sr); n = min(400, len(y) - i)
            y[i:i + n] += 1.5 * rng.standard_normal(n) * np.exp(-np.arange(n) / 60)
    elif kind == "friction":
        hp = rng.standard_normal(len(t)); hp = hp - np.convolve(hp, np.ones(6) / 6, "same")
        y += 0.8 * hp
    return (y / (np.max(np.abs(y)) * 1.1)).astype(np.float32)
