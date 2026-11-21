# MSCN + 频域融合的对抗样本检测

本仓库是课程报告 **《基于 MSCN 与频域特征融合的对抗样本检测及其能力边界》** 的实验代码。
在固定假正率的评测设置下，比较 18 维 NSS 特征与 202 维多尺度—色度—频域融合特征，
在 CIFAR-10（主实验）与 ImageNet 子集（验证）上评估对 L$\infty$ 与 L2 攻击的检测能力。

## Note

代码结构参考了综述论文的官方实现 [detectors_review](https://github.com/aldahdooh/detectors_review)，
MSCN 特征实现参考了 NSS 原始工作 [Kherchouche et al., IJCNN 2020](https://hal.archives-ouvertes.fr/hal-03003468)。

## Requirement

1. Python 3.12
2. PyTorch 2.15（CUDA 13.2）
3. torchattacks 3.5
4. scikit-learn、PyWavelets、PyArrow、Pillow、tqdm、matplotlib

```bash
uv sync          # 或 pip install -e .
```

## setup_paths

`setup_paths.py` 集中管理路径、数据集、攻击列表、特征配置与评测参数（FPR、划分比例、随机种子）。
所有路径均为项目内相对路径，无需修改。

## Data preparation

两个脚本都会在文件缺失时自动下载（CIFAR 取自 Gitee 镜像，ImageNet 取自 HuggingFace），
下载后转换为 npz 供训练与检测直接使用：

```bash
python tools/prepare_cifar.py                  # -> data/cifar10.npz
python tools/prepare_imagenet.py --n 1000      # -> data/imagenet.npz（约 530 MB 原文件）
```

若网络不可用，也可手动准备：

- CIFAR-10：下载 `train-00000-of-00001.parquet` 与 `test-00000-of-00001.parquet`
  到 `data/cifar10_parquet/`（[Gitee 镜像](https://gitee.com/hf-datasets/uoft-cs-cifar10)）；
- ImageNet：下载 `train-00000-of-00014.parquet` 到 `data/imagenet/`
  （[HuggingFace: mrm8488/ImageNet1K-val](https://huggingface.co/datasets/mrm8488/ImageNet1K-val)）。

## Train CNN

CIFAR-10 训练 ResNet-18：

```bash
python train_cnn_base.py -d=cifar -e=30 -b=128 --lr=0.1
```

ImageNet 直接使用 torchvision 预训练 ResNet-50（`IMAGENET1K_V1`），无需训练。

## Generate adversarial examples

```bash
python generate_adv.py -d=cifar                     # 生成 setup_paths.ATTACKS['cifar'] 中的全部攻击
python generate_adv.py -d=cifar -a=pgd_0.03125      # 只生成一种
```

产物：`adv_data/{dataset}_{attack}.npy`（float32, [0,1], (N,H,W,C)）。
攻击使用 torchattacks；如需增删攻击，请同步修改 `setup_paths.py` 中的 `ATTACKS`。

## Run detectors

```bash
python detect_nss.py    -d=cifar --limit=1000 --train-attack=pgd_0.03125   # NSS-18
python detect_fusion.py -d=cifar --limit=1000 --train-attack=pgd_0.03125   # Fusion-202
```

结果为每个攻击一个 CSV：`results/<tag>/{dataset}_{attack}.csv`。

## Collect results

```bash
python collect_results.py --tags=nss,fusion
```

生成 `results/<tag>/all_{dataset}.csv` 与 `all.csv`，列为 `DR/FPR/AUC`。

## Figures

生成论文中的可视化图（默认输出到 `figures/`）：

```bash
python tools/make_figures.py --dataset imagenet            # 图 3/图 4
python tools/make_figures.py --dataset cifar               # 需要 CIFAR 示例时
python tools/make_figures.py --out ../path/to/figures      # 指定输出目录
```

## Reproduce

```bash
bash run_cifar.sh        # CIFAR-10 全流程（约 30 分钟）
bash run_imagenet.sh     # ImageNet 验证（约 35 分钟）
```


所有输入与输出（`data/`、`adv_data/`、`results/`、`checkpoints/`、`figures/`）都在本目录内，
不需要外部路径；只有 ImageNet 预训练权重会由 torchvision 下载到 `~/.cache/torch/`。

## Project structure

```
code-pytorch/
├── setup_paths.py            # 路径 / 数据集 / 攻击 / 特征配置
├── common/
│   ├── data.py               # CIFAR-10 与 ImageNet 子集加载
│   ├── detector.py           # 特征缓存 + 划分 + SVM + 评测
│   └── util.py               # 指标 / CSV / 随机种子
├── features/
│   ├── mscn.py               # MSCN + GGD/AGGD（18 维）
│   ├── freq.py               # DWT 子带统计 + 径向谱（40 维）
│   └── fusion.py             # 特征拼接（202 维）
├── models/cnn.py             # CIFAR ResNet-18 / ImageNet 预训练 ResNet-50
├── tools/
│   ├── prepare_cifar.py      # parquet -> npz
│   ├── prepare_imagenet.py   # parquet -> npz（1000 张）
│   └── make_figures.py       # 生成论文图 3/图 4
├── train_cnn_base.py         # 训练 CIFAR 分类器
├── generate_adv.py           # 生成对抗样本
├── detect_nss.py             # NSS-18 检测器
├── detect_fusion.py          # Fusion-202 检测器
├── collect_results.py        # 汇总 CSV
├── run_cifar.sh / run_imagenet.sh
├── figures/                  # make_figures.py 的输出
└── data/ adv_data/ results/ checkpoints/
```
