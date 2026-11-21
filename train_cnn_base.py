"""训练 CIFAR-10 的 ResNet-18 基线分类器。

用法：python train_cnn_base.py -d=cifar -e=30 -b=128 --lr=0.1
ImageNet 使用 torchvision 预训练权重，无需训练。
"""

from __future__ import division, absolute_import, print_function

import argparse
import time

import torch
import torch.nn as nn

from common.data import get_loaders
from common.util import get_device, set_seed
from models.cnn import build_model, save_model
from setup_paths import DATASETS, SEED


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total


def main(args):
    assert args.dataset in DATASETS, f"dataset 必须是 {DATASETS}"
    assert args.dataset == "cifar", "ImageNet 使用预训练权重，无需训练"
    set_seed(SEED)
    device = get_device()
    print(f"[device] {device}")

    train_loader, test_loader = get_loaders(
        args.dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, train_subset=args.train_subset)

    model = build_model(args.dataset).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
        scheduler.step()

        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch:3d}/{args.epochs}  loss={running / len(train_loader.dataset):.4f}  "
              f"test_acc={acc * 100:.2f}%  ({time.time() - t0:.1f}s)")
        if acc > best:
            best = acc
            save_model(model, args.dataset)
            print(f"  -> saved checkpoint {args.dataset} (best {best * 100:.2f}%)")

    print(f"[done] best test acc = {best * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 ResNet-18 baseline.")
    parser.add_argument("-d", "--dataset", required=True, choices=DATASETS)
    parser.add_argument("-e", "--epochs", default=30, type=int)
    parser.add_argument("-b", "--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--train-subset", default=None, type=int,
                        help="只用训练集前 N 张（快速冒烟测试）")
    main(parser.parse_args())
