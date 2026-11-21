"""融合特征检测器（论文 4.2 / 4.3 节的 Fusion-202）。

用法：
    python detect_fusion.py -d=cifar --limit=1000 --train-attack=pgd_0.03125
    python detect_fusion.py -d=cifar --limit=1000 --freq-mode=none --tag=abl
"""

from __future__ import division, absolute_import, print_function

import argparse

from common.detector import run
from features.fusion import feature_dim
from setup_paths import DATASETS, FEATURE_CFG, TARGET_FPR, TRAIN_ATTACK


def main(args):
    assert args.dataset in DATASETS, f"dataset 必须是 {DATASETS}"
    cfg = dict(FEATURE_CFG["fusion"])
    cfg["freq"] = args.freq_mode == "both"

    dim = feature_dim(cfg, image_shape=(32, 32, 3) if args.dataset == "cifar" else (224, 224, 3))
    print(f"[cfg] {cfg}\n[cfg] feature dim = {dim}")

    run(dataset=args.dataset, cfg=cfg, tag=args.tag,
        train_attack=args.train_attack or TRAIN_ATTACK[args.dataset],
        fpr=args.fpr, limit=args.limit,
        force_features=args.force, tune_svm=not args.no_tune)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSCN + frequency fusion detector.")
    parser.add_argument("-d", "--dataset", required=True, choices=DATASETS)
    parser.add_argument("--tag", default="fusion", help="结果目录名")
    parser.add_argument("--freq-mode", default="both", choices=["both", "none"],
                        help="both=202 维融合特征，none=仅 162 维 MSCN")
    parser.add_argument("--train-attack", default=None)
    parser.add_argument("--fpr", default=TARGET_FPR, type=float)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-tune", action="store_true")
    main(parser.parse_args())
