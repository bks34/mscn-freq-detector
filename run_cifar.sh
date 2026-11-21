#!/usr/bin/env bash
# CIFAR-10 主实验：训练 → 5 种攻击 → NSS-18 与 Fusion-202
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python
LIMIT=1000

echo "== 0/4 准备 CIFAR-10 数据 =="
$PY tools/prepare_cifar.py

echo "== 1/4 训练 ResNet-18 =="
$PY train_cnn_base.py -d=cifar -e=30 -b=128 --lr=0.1

echo "== 2/4 生成对抗样本 =="
for a in fgsm_0.03125 bim_0.03125 pgd_0.03125 cw_10 df_0; do
  $PY generate_adv.py -d=cifar -a="$a" --max-samples=$LIMIT --batch-size=100 --force
done

echo "== 3/4 检测器 =="
$PY detect_nss.py    -d=cifar --limit=$LIMIT --train-attack=pgd_0.03125 --force
$PY detect_fusion.py -d=cifar --limit=$LIMIT --train-attack=pgd_0.03125 --force
$PY collect_results.py --tags=nss,fusion
echo "完成：results/nss/all_cifar.csv、results/fusion/all_cifar.csv"
