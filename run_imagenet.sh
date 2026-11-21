#!/usr/bin/env bash
# ImageNet 验证：预训练 ResNet-50 + 1000 张子集 + 4 种攻击
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python
LIMIT=1000

echo "== 0/3 parquet -> npz =="
$PY tools/prepare_imagenet.py --n $LIMIT

echo "== 1/3 生成对抗样本 =="
for a in fgsm_0.03125 bim_0.03125 pgd_0.03125 cw_10; do
  $PY generate_adv.py -d=imagenet -a="$a" --max-samples=$LIMIT --batch-size=32 --force
done

echo "== 2/3 检测器 =="
$PY detect_nss.py    -d=imagenet --limit=$LIMIT --train-attack=pgd_0.03125 --force
$PY detect_fusion.py -d=imagenet --limit=$LIMIT --train-attack=pgd_0.03125 --force

echo "== 3/3 汇总 =="
$PY collect_results.py --tags=nss,fusion
echo "完成：results/nss/all_imagenet.csv、results/fusion/all_imagenet.csv"
