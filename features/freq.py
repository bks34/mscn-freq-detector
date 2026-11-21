"""频域特征（对应论文 3.1.3 节）：DWT 子带统计 + 径向功率谱，共 40 维。"""

from __future__ import division, absolute_import, print_function

import numpy as np
import pywt

from .mscn import to_gray, aggd_fit


# ------------------------------------------------------------------ DWT
def dwt_features(img, wavelet="db1", level=3):
    """每个细节子带的 log 能量、标准差、峰度、GGD 形状参数 -> 36 维。"""
    gray = to_gray(img).astype(np.float64)
    coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=level)

    feats = []
    for detail in coeffs[1:]:
        for band in detail:
            band = np.asarray(band).ravel()
            energy = np.log(np.mean(band ** 2) + 1e-12)
            std = float(np.std(band))
            kurt = (float(np.mean((band - band.mean()) ** 4) / (std ** 4 + 1e-12))
                    if std > 0 else 0.0)
            alpha, _, _, _ = aggd_fit(band)
            feats.extend([energy, std, kurt, alpha])
    return np.asarray(feats, dtype=np.float32)


# ------------------------------------------------------------------ 径向谱
def radial_power_spectrum(gray, n_bins=32):
    """按半径做方位平均后的功率谱，返回 (r, P)。"""
    gray = np.asarray(gray, dtype=np.float64)
    power = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    bins = np.linspace(0, min(cy, cx), n_bins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    p_flat = power.ravel()

    r_out, p_out = [], []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        r_out.append(0.5 * (bins[b] + bins[b + 1]))
        p_out.append(p_flat[mask].mean())
    return np.asarray(r_out), np.asarray(p_out)


def spectrum_features(img, n_bins=32):
    """斜率 beta、拟合残差、高频能量占比、谱平坦度 -> 4 维。"""
    r, p = radial_power_spectrum(to_gray(img), n_bins=n_bins)
    good = (r > 0) & (p > 0)
    if good.sum() < 4:
        return np.zeros(4, dtype=np.float32)

    log_r, log_p = np.log(r[good]), np.log(p[good])
    beta, intercept = np.polyfit(log_r, log_p, 1)
    resid = float(np.std(log_p - (beta * log_r + intercept)))

    hf_ratio = float(p[r > 0.5 * r.max()].sum() / (p.sum() + 1e-12))
    p_norm = p / (p.sum() + 1e-12)
    flatness = float(np.exp(np.mean(np.log(p_norm + 1e-12))) / (np.mean(p_norm) + 1e-12))

    return np.asarray([beta, resid, hf_ratio, flatness], dtype=np.float32)


# ------------------------------------------------------------------ 对外接口
def freq_features(img, wavelet="db1", level=3, n_bins=32):
    return np.concatenate([
        dwt_features(img, wavelet=wavelet, level=level),
        spectrum_features(img, n_bins=n_bins),
    ]).astype(np.float32)
