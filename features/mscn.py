"""MSCN / NSS 特征提取（对应论文 3.1.1 节）。

输入：float32 图像，形状 (H, W) / (H, W, 3) / (3, H, W)，取值 [0, 1]。
输出：brisque18(img) -> 18 维；mscn_features(img, scales, chroma) -> 多尺度/色度组合。
"""

from __future__ import division, absolute_import, print_function

import numpy as np
from scipy import special
from scipy.optimize import brentq
from scipy.signal import convolve2d


# ------------------------------------------------------------------ 基础
def _to_float01(img):
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[0] < img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def to_gray(img):
    img = _to_float01(img)
    if img.ndim == 3:
        if img.shape[2] == 1:
            return img[..., 0]
        return (0.299 * img[..., 0] + 0.587 * img[..., 1]
                + 0.114 * img[..., 2]).astype(np.float32)
    return img


def rgb_to_ycbcr(img):
    """返回 (Y, Cb, Cr)，均为 [0,1]。"""
    img = _to_float01(img)
    if img.ndim != 3 or img.shape[2] != 3:
        gray = to_gray(img)
        return gray, gray, gray
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return (y.astype(np.float32), np.clip(cb, 0, 1).astype(np.float32),
            np.clip(cr, 0, 1).astype(np.float32))


def resize_block(img, scale):
    """块平均下采样；不能整除时裁掉边缘。"""
    if scale <= 1:
        return img
    h, w = img.shape[:2]
    h2, w2 = h // scale, w // scale
    img = img[:h2 * scale, :w2 * scale]
    return img.reshape(h2, scale, w2, scale, *img.shape[2:]).mean(axis=(1, 3))


def gaussian_kernel2d(n=7, sigma=7 / 6):
    y, x = np.indices((n, n)) - int(n / 2)
    k = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return k / k.sum()


# ------------------------------------------------------------------ MSCN
def mscn_coefficients(img, kernel_size=7, sigma=7 / 6, c=1 / 255):
    """Î = (I − μ) / (σ + c)，μ、σ 为 7×7 高斯窗下的局部均值与标准差。"""
    gray = to_gray(img)
    kernel = gaussian_kernel2d(kernel_size, sigma)
    mu = convolve2d(gray, kernel, mode="same", boundary="symm")
    e_sq = convolve2d(gray ** 2, kernel, mode="same", boundary="symm")
    var = np.maximum(e_sq - mu ** 2, 0.0)
    return ((gray - mu) / (np.sqrt(var) + c)).astype(np.float32)


def pair_products(mscn):
    return {
        "mscn": mscn,
        "horizontal": mscn[:, :-1] * mscn[:, 1:],
        "vertical": mscn[:-1, :] * mscn[1:, :],
        "main_diagonal": mscn[:-1, :-1] * mscn[1:, 1:],
        "secondary_diagonal": mscn[1:, :-1] * mscn[:-1, 1:],
    }


# ------------------------------------------------------------------ AGGD
def _phi(alpha):
    return special.gamma(2.0 / alpha) ** 2 / (special.gamma(1.0 / alpha)
                                              * special.gamma(3.0 / alpha))


def aggd_fit(x):
    """矩匹配估计 AGGD 参数，返回 (alpha, mean, sigma_l, sigma_r)。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 8 or np.allclose(x, 0):
        return 2.0, 0.0, 0.0, 0.0

    left, right = x[x < 0], x[x >= 0]
    if left.size == 0 or right.size == 0:
        s = float(np.sqrt(np.mean(x ** 2)))
        return 2.0, 0.0, s, s

    r_hat = (np.mean(np.abs(x)) ** 2) / (np.mean(x ** 2) + 1e-12)
    gamma = np.sqrt(np.mean(left ** 2) / (np.mean(right ** 2) + 1e-12))
    r_hat = max(min(r_hat, 1.0), 1e-6)
    R_hat = r_hat * (gamma ** 3 + 1) * (gamma + 1) / ((gamma ** 2 + 1) ** 2)
    R_hat = min(max(R_hat, 1e-6), 1 - 1e-6)

    try:
        alpha = brentq(lambda a: _phi(a) - R_hat, 0.05, 60.0, maxiter=200)
    except ValueError:
        alpha = 2.0
    alpha = float(np.clip(alpha, 0.05, 60.0))

    sigma_l = float(np.sqrt(np.mean(left ** 2)))
    sigma_r = float(np.sqrt(np.mean(right ** 2)))
    mean = ((sigma_r - sigma_l) * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
            * special.gamma(2 / alpha) / special.gamma(1 / alpha))
    return alpha, float(mean), sigma_l, sigma_r


def _aggd_features(x):
    alpha, mean, sl, sr = aggd_fit(x)
    return [alpha, mean, sl ** 2, sr ** 2]


def _ggd_features(x):
    alpha, _, sl, sr = aggd_fit(x)
    return [alpha, (sl ** 2 + sr ** 2) / 2.0]


# ------------------------------------------------------------------ 对外接口
def brisque18(img):
    """18 维：MSCN 的 GGD(2) + 四方向乘积的 AGGD(4×4)。"""
    coefs = pair_products(mscn_coefficients(img))
    feats = _ggd_features(coefs.pop("mscn"))
    for c in coefs.values():
        feats.extend(_aggd_features(c))
    return np.asarray(feats, dtype=np.float32)


def mscn_features(img, scales=(1,), chroma=False):
    """按尺度与通道组合 18 维特征。"""
    channels = [to_gray(img)]
    if chroma:
        _, cb, cr = rgb_to_ycbcr(img)
        channels.extend([cb, cr])

    feats = []
    for scale in scales:
        for ch in channels:
            feats.extend(brisque18(resize_block(ch, scale)))
    return np.asarray(feats, dtype=np.float32)
