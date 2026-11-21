"""把 HF/Gitee 镜像的 CIFAR-10 parquet 转成 npz，供 common/data.py 直接加载。

用法：python tools/prepare_cifar.py
产物：data/cifar10.npz（x_train, y_train, x_test, y_test，uint8）
"""

from __future__ import division, absolute_import, print_function

import io
import os
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "cifar10_parquet")
OUT = os.path.join(ROOT, "data", "cifar10.npz")

URLS = {
    "train": "https://gitee.com/hf-datasets/uoft-cs-cifar10/raw/main/plain_text/train-00000-of-00001.parquet",
    "test": "https://gitee.com/hf-datasets/uoft-cs-cifar10/raw/main/plain_text/test-00000-of-00001.parquet",
}


def ensure_file(url, path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)


def load_split(path):
    table = pq.read_table(path)
    xs, ys = [], []
    for row in table.to_pylist():
        img = row["img"]
        arr = np.asarray(Image.open(io.BytesIO(img["bytes"])).convert("RGB"), dtype=np.uint8)
        xs.append(arr)
        ys.append(int(row["label"]))
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def main():
    for split in ("train", "test"):
        ensure_file(URLS[split], os.path.join(SRC, f"{split}.parquet"))
    x_tr, y_tr = load_split(os.path.join(SRC, "train.parquet"))
    x_te, y_te = load_split(os.path.join(SRC, "test.parquet"))
    print("train", x_tr.shape, "test", x_te.shape)
    np.savez_compressed(OUT, x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
    print("saved", OUT, os.path.getsize(OUT) // 1024 // 1024, "MB")


if __name__ == "__main__":
    main()
