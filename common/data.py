"""数据集加载（CIFAR-10 与 ImageNet 子集）。

CIFAR-10：优先使用 tools/prepare_cifar.py 生成的 data/cifar10.npz，
          没有时回退到 torchvision 自动下载。
ImageNet：使用 tools/prepare_imagenet.py 生成的 data/imagenet.npz（1000 张）。
"""

from __future__ import division, absolute_import, print_function

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from setup_paths import DATA_DIR

CIFAR_NPZ = os.path.join(DATA_DIR, "cifar10.npz")
IMAGENET_NPZ = os.path.join(DATA_DIR, "imagenet.npz")


class ArrayDataset(Dataset):
    """内存中的 uint8 图像数组。"""

    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = Image.fromarray(self.x[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.y[i])


def _tf(train=False, dataset=None):
    """训练集对 CIFAR 做轻量增强；输入保持 [0,1]，不做标准化。"""
    if train and dataset == "cifar":
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.ToTensor()


def get_datasets(dataset):
    if dataset == "cifar":
        if os.path.isfile(CIFAR_NPZ):
            d = np.load(CIFAR_NPZ)
            tr = ArrayDataset(d["x_train"], d["y_train"], transform=_tf(True, dataset))
            te = ArrayDataset(d["x_test"], d["y_test"], transform=_tf())
        else:
            tr = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                  transform=_tf(True, dataset))
            te = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                  transform=_tf())
    elif dataset == "imagenet":
        if not os.path.isfile(IMAGENET_NPZ):
            raise FileNotFoundError(
                f"找不到 {IMAGENET_NPZ}，请先运行 tools/prepare_imagenet.py")
        d = np.load(IMAGENET_NPZ)
        tr = ArrayDataset(d["x"], d["y"], transform=_tf())
        te = ArrayDataset(d["x"], d["y"], transform=_tf())
    else:
        raise ValueError(f"不支持的数据集：{dataset}")
    return tr, te


def get_loaders(dataset, batch_size=128, num_workers=2, train_subset=None):
    tr, te = get_datasets(dataset)
    if train_subset is not None:
        tr = Subset(tr, range(min(train_subset, len(tr))))
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(te, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def to_numpy(images):
    """Tensor -> float32 ndarray：(N,C,H,W)->(N,H,W,C)，(C,H,W)->(H,W,C)。"""
    if isinstance(images, torch.Tensor):
        arr = images.detach().cpu().numpy()
    else:
        arr = np.asarray(images)
    if arr.ndim == 4:
        if arr.shape[1] in (1, 3) and arr.shape[1] < arr.shape[-1]:
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
    return arr.astype(np.float32)


def get_test_arrays(dataset, limit=None):
    """返回 (images, labels)，images 为 float32 [0,1]。"""
    _, te = get_datasets(dataset)
    n = len(te) if limit is None else min(limit, len(te))
    imgs, labels = [], []
    for i in range(n):
        x, y = te[i]
        imgs.append(to_numpy(x))
        labels.append(int(y))
    return np.stack(imgs).astype(np.float32), np.asarray(labels, dtype=np.int64)
