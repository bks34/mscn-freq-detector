"""特征拼接（对应论文 3.1.4 节）。

cfg 字段：
    scales : 下采样倍率，如 (1,) 或 (1, 2, 4)
    chroma : 是否在 Cb / Cr 上重复 MSCN 统计
    freq   : 是否拼接 40 维频域特征
"""

from __future__ import division, absolute_import, print_function

import numpy as np

from .mscn import mscn_features
from .freq import freq_features


def build_feature_vector(img, cfg):
    parts = [mscn_features(img, scales=tuple(cfg.get("scales", (1,))),
                           chroma=bool(cfg.get("chroma", False)))]
    if cfg.get("freq", False):
        parts.append(freq_features(img))
    return np.concatenate(parts).astype(np.float32)


def feature_dim(cfg, image_shape=(32, 32, 3)):
    """用占位图计算特征维度。"""
    return int(build_feature_vector(np.zeros(image_shape, dtype=np.float32), cfg).shape[0])
