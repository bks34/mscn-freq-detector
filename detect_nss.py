"""NSS 基线检测器（论文 4.2 / 4.3 节的 NSS-18）。

用法：python detect_nss.py -d=cifar --limit=1000 --train-attack=pgd_0.03125
"""

from __future__ import division, absolute_import, print_function

import argparse

from common.detector import run
from setup_paths import DATASETS, FEATURE_CFG, TARGET_FPR, TRAIN_ATTACK


def main(args):
    assert args.dataset in DATASETS, f"dataset 必须是 {DATASETS}"
    cfg = dict(FEATURE_CFG["nss"])
    run(dataset=args.dataset, cfg=cfg, tag="nss",
        train_attack=args.train_attack or TRAIN_ATTACK[args.dataset],
        fpr=args.fpr, limit=args.limit,
        force_features=args.force, tune_svm=not args.no_tune)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSS baseline detector (MSCN + AGGD + SVM).")
    parser.add_argument("-d", "--dataset", required=True, choices=DATASETS)
    parser.add_argument("--train-attack", default=None, help="训练检测器用的攻击，默认 PGD")
    parser.add_argument("--fpr", default=TARGET_FPR, type=float)
    parser.add_argument("--limit", default=None, type=int, help="只用测试集前 N 张")
    parser.add_argument("--force", action="store_true", help="强制重算特征")
    parser.add_argument("--no-tune", action="store_true", help="跳过 SVM 网格搜索")
    main(parser.parse_args())
