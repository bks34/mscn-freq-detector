"""全局配置：路径、数据集、攻击与特征。

所有路径都相对于项目根目录，无需修改。
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT, "data")            # 数据集
CKPT_DIR = os.path.join(ROOT, "checkpoints")     # CNN 权重
ADV_DIR = os.path.join(ROOT, "adv_data")         # 对抗样本 .npy
RESULT_DIR = os.path.join(ROOT, "results")       # 检测结果 CSV 与特征缓存

for _d in (DATA_DIR, CKPT_DIR, ADV_DIR, RESULT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- 数据集
# 论文使用 CIFAR-10（主实验）与 ImageNet 子集（验证）
DATASETS = ["cifar", "imagenet"]
NUM_CLASSES = {"cifar": 10, "imagenet": 1000}
INPUT_SIZE = {"cifar": 32, "imagenet": 224}

# ---------------------------------------------------------------- 攻击
# 命名规则：<attack>_<eps>；eps 为 [0,1] 尺度（8/255≈0.03125）
ATTACKS = {
    "cifar": [
        "fgsm_0.03125", "bim_0.03125", "pgd_0.03125",
        "cw_10", "df_0",
    ],
    "imagenet": [
        "fgsm_0.03125", "bim_0.03125", "pgd_0.03125", "cw_10",
    ],
}

ATTACK_PARAMS = {
    "fgsm": dict(),
    "bim":  dict(steps=10),
    "pgd":  dict(steps=40),
    "cw":   dict(c=1.0, steps=200, lr=0.01),
    "df":   dict(steps=50),
}

# 训练检测器时使用的“已知攻击”
TRAIN_ATTACK = {"cifar": "pgd_0.03125", "imagenet": "pgd_0.03125"}

# ---------------------------------------------------------------- 特征
# nss   : 18 维（单尺度灰度 MSCN + GGD/AGGD）
# fusion: 202 维（多尺度 + 色度 + 频域）
FEATURE_CFG = {
    "nss":    dict(scales=(1,), chroma=False, freq=False),
    "fusion": dict(scales=(1, 2, 4), chroma=True, freq=True),
}

# ---------------------------------------------------------------- 评测
TARGET_FPR = 0.05      # 固定假正率
TRAIN_RATIO = 0.7      # 干净样本划分比例
SEED = 42
