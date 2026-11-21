"""分类器：CIFAR-10 用 ResNet-18，ImageNet 用预训练 ResNet-50。"""

from __future__ import division, absolute_import, print_function

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from setup_paths import CKPT_DIR, NUM_CLASSES


# ------------------------------------------------------------------ ResNet-18 (CIFAR)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18CIFAR(nn.Module):
    """CIFAR 版 ResNet-18：3×3 stem、无最大池化。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


class _Normalized(nn.Module):
    """把 [0,1] 输入按 ImageNet 均值方差归一化后再送入预训练网络。"""

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


# ------------------------------------------------------------------ 工厂
def build_model(dataset):
    if dataset == "cifar":
        return ResNet18CIFAR(NUM_CLASSES["cifar"])
    if dataset == "imagenet":
        import torchvision.models as tvm
        return tvm.resnet50(weights=None)
    raise ValueError(f"不支持的数据集：{dataset}")


def ckpt_path(dataset):
    return os.path.join(CKPT_DIR, f"cnn_{dataset}.pt")


def save_model(model, dataset):
    torch.save(model.state_dict(), ckpt_path(dataset))


def load_model(dataset, device="cpu"):
    if dataset == "imagenet":
        # 直接使用 torchvision 预训练权重；外层归一化保证攻击仍在 [0,1] 空间
        import torchvision.models as tvm
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        model = _Normalized(base, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return model.to(device).eval()

    model = build_model(dataset)
    path = ckpt_path(dataset)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到权重 {path}，请先运行 train_cnn_base.py -d={dataset}")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()
