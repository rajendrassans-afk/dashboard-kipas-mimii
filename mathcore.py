"""Inti matematika (NumPy murni) untuk pipeline STFT -> Mel -> log -> DCT (MFCC).

Setiap fungsi ditulis dari definisi rumusnya, lalu diverifikasi terhadap
librosa/scipy di tests/verify_math.py. Konvensi mengikuti baseline MIMII
(n_fft=1024, hop=512, n_mels=64).
"""
from __future__ import annotations
import numpy as np

EPS = np.finfo(float).eps


# ---------------------------------------------------------------- DFT / FFT
def dft_naive(x: np.ndarray) -> np.ndarray:
    """DFT langsung, O(N^2):  X[k] = sum_n x[n] e^{-j 2 pi k n / N}."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    W = np.exp(-2j * np.pi * np.outer(n, n) / N)  # matriks Fourier N x N
    return W @ x


def fft_radix2(x: np.ndarray) -> np.ndarray:
    """FFT Cooley-Tukey radix-2 (decimation in time), O(N log N). N harus 2^m.

    X[k]       = E[k] + W_N^k O[k]
    X[k + N/2] = E[k] - W_N^k O[k]
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N & (N - 1):
        raise ValueError("N harus pangkat dua")
    if N == 1:
        return x
    E = fft_radix2(x[0::2])
    O = fft_radix2(x[1::2])
    W = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate([E + W * O, E - W * O])


# ----------------------------------------------------------------- Window
def hann(N: int, periodic: bool = True) -> np.ndarray:
    """w[n] = 0.5 (1 - cos(2 pi n / M)), M = N (periodik) atau N-1 (simetris)."""
    M = N if periodic else N - 1
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N) / M))


# ------------------------------------------------------------------- STFT
def frame_signal(y: np.ndarray, n_fft: int, hop: int, center: bool = True) -> np.ndarray:
    """Potong sinyal jadi frame (n_frames, n_fft). center=True -> padding nol (sama dgn librosa>=0.10)."""
    y = np.asarray(y, dtype=float)
    if center:
        y = np.pad(y, n_fft // 2, mode="constant")
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n_frames = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return y[idx]


def stft(y, n_fft=1024, hop=512, window="hann", center=True) -> np.ndarray:
    """STFT:  X[m,k] = sum_n x[n + mH] w[n] e^{-j 2 pi k n / N}.

    Return (n_fft/2+1, n_frames) kompleks. FFT dihitung dengan np.fft.rfft
    (algoritma sama dengan fft_radix2, hanya lebih cepat & divektorkan).
    """
    w = hann(n_fft) if window == "hann" else np.ones(n_fft)
    frames = frame_signal(y, n_fft, hop, center) * w
    return np.fft.rfft(frames, axis=1).T


def power_spec(S: np.ndarray) -> np.ndarray:
    """|X|^2"""
    return (S.real ** 2 + S.imag ** 2)


# -------------------------------------------------------------- Mel scale
def hz_to_mel(f):
    """Skala mel (HTK):  m = 2595 log10(1 + f/700)."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels=64, fmin=0.0, fmax=None) -> np.ndarray:
    """Filterbank segitiga (n_mels, n_fft/2+1), puncak = 1 (skala HTK, tanpa norm)."""
    fmax = fmax or sr / 2
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / (ce - lo)
        dn = (hi - freqs) / (hi - ce)
        fb[i] = np.maximum(0.0, np.minimum(up, dn))
    return fb


def log_mel(y, sr, n_fft=1024, hop=512, n_mels=64, power=2.0):
    """log-mel dB gaya baseline MIMII:  20/p * log10(mel + eps)."""
    P = power_spec(stft(y, n_fft, hop)) if power == 2.0 else np.abs(stft(y, n_fft, hop))
    mel = mel_filterbank(sr, n_fft, n_mels) @ P
    return 20.0 / power * np.log10(mel + EPS)


# --------------------------------------------------------------- DCT / MFCC
def dct_matrix(n_in: int, n_out: int) -> np.ndarray:
    """DCT-II ortonormal:
    c[k] = a_k sum_n x[n] cos(pi k (2n+1) / (2N)),  a_0 = sqrt(1/N), a_k = sqrt(2/N)."""
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    D = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    D *= np.sqrt(2.0 / n_in)
    D[0] *= 1.0 / np.sqrt(2.0)
    return D


def mfcc(y, sr, n_fft=1024, hop=512, n_mels=64, n_mfcc=13):
    """MFCC = DCT-II( log-mel (dB) ) -> (n_mfcc, n_frames)."""
    L = log_mel(y, sr, n_fft, hop, n_mels)
    return dct_matrix(n_mels, n_mfcc) @ L


def delta(M: np.ndarray, width: int = 2) -> np.ndarray:
    """Delta regresi:  d[t] = sum_{i=1..W} i (c[t+i]-c[t-i]) / (2 sum i^2)."""
    pad = np.pad(M, ((0, 0), (width, width)), mode="edge")
    T = M.shape[1]
    num = sum(i * (pad[:, width + i: width + i + T] - pad[:, width - i: width - i + T])
              for i in range(1, width + 1))
    return num / (2 * sum(i * i for i in range(1, width + 1)))


# ------------------------------------------------------- Fitur spektral
def spectral_features(y, sr, n_fft=1024, hop=512):
    """RMS, centroid, bandwidth, rolloff-85%, flatness (rata-rata & std)."""
    P = power_spec(stft(y, n_fft, hop))
    f = np.linspace(0, sr / 2, P.shape[0])[:, None]
    mag = np.sqrt(P)
    s = mag.sum(0) + EPS
    centroid = (f * mag).sum(0) / s                          # sum f|X| / sum |X|
    bw = np.sqrt((((f - centroid) ** 2) * mag).sum(0) / s)   # sebaran di sekitar centroid
    cum = np.cumsum(P, 0) / (P.sum(0) + EPS)
    rolloff = f[:, 0][np.argmax(cum >= 0.85, axis=0)]
    flat = np.exp(np.mean(np.log(P + EPS), 0)) / (P.mean(0) + EPS)  # rata geometrik / aritmetik
    rms = np.sqrt(np.mean(frame_signal(y, n_fft, hop) ** 2, axis=1))
    return {"rms": rms, "centroid": centroid, "bandwidth": bw, "rolloff": rolloff, "flatness": flat}
