from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis


def rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=1) + 1e-8)


def peak(x: np.ndarray) -> np.ndarray:
    return np.max(np.abs(x), axis=1)


def kurtosis_np(x: np.ndarray) -> np.ndarray:
    return kurtosis(x, axis=1, fisher=False, bias=False, nan_policy="omit")


def spectrum(x: np.ndarray, fs: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    spec = np.abs(np.fft.rfft(x, axis=1))
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / fs)
    return freqs, spec


def cosine_similarity_mean(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    a = a[:n].reshape(n, -1)
    b = b[:n].reshape(n, -1)
    return float(np.mean(np.sum(a * b, axis=1) / ((np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-8)))


def spectrum_corr_mean(real: np.ndarray, fake: np.ndarray) -> float:
    n = min(len(real), len(fake))
    if n == 0:
        return float("nan")
    _, sr = spectrum(real[:n])
    _, sf = spectrum(fake[:n])
    vals = []
    for i in range(n):
        vals.append(np.corrcoef(sr[i], sf[i])[0, 1])
    return float(np.nanmean(vals))


def band_energy_distribution(x: np.ndarray, fs: int = 2048) -> np.ndarray:
    freqs, spec = spectrum(x, fs)
    power = spec**2
    bands = [(0, 50), (50, 100), (100, 200), (200, 500), (500, fs / 2)]
    energies = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        energies.append(power[:, mask].sum(axis=1))
    out = np.stack(energies, axis=1)
    return out / (out.sum(axis=1, keepdims=True) + 1e-8)


def beds_similarity(real: np.ndarray, fake: np.ndarray) -> float:
    if len(real) == 0 or len(fake) == 0:
        return float("nan")
    r = band_energy_distribution(real).mean(axis=0)
    f = band_energy_distribution(fake).mean(axis=0)
    return float(1.0 - np.mean(np.abs(r - f)))


def order_energy_np(x: np.ndarray, fs: int = 2048, rpm: float = 740.0, band_width: float = 2.0) -> np.ndarray:
    freqs, spec = spectrum(x, fs)
    power = spec**2
    fr = rpm / 60.0
    energies = []
    for mul in (0.5, 1.0, 1.5, 2.0, 3.0):
        center = mul * fr
        mask = (freqs >= center - band_width) & (freqs <= center + band_width)
        energies.append(power[:, mask].sum(axis=1))
    mask = (freqs >= 80) & (freqs <= 500)
    energies.append(power[:, mask].sum(axis=1))
    out = np.stack(energies, axis=1)
    return out / (out.sum(axis=1, keepdims=True) + 1e-8)
