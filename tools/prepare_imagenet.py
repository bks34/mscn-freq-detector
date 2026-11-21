"""把 ImageNet-1k 的 parquet 转成 npz（均匀抽样 N 张，统一缩放到 224×224）。

parquet 按类别排序，若直接取前 N 张只会覆盖少数类别；这里按全局步长抽样，
尽量覆盖更多类别。

用法：python tools/prepare_imagenet.py [--src ...] [--n 1000]
产物：data/imagenet.npz（x: uint8 [N,224,224,3]，y: int64 [N]）
"""

from __future__ import division, absolute_import, print_function

import argparse
import io
import os
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = ("https://huggingface.co/datasets/mrm8488/ImageNet1K-val/resolve/main/"
       "data/train-00000-of-00014.parquet")


def find_col(names, candidates):
    for c in candidates:
        if c in names:
            return c
    raise KeyError(f"找不到列，候选 {candidates}，实际 {names}")


def main(args):
    if not os.path.isfile(args.src):
        os.makedirs(os.path.dirname(args.src), exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, args.src)
    pf = pq.ParquetFile(args.src)
    total = pf.metadata.num_rows
    step = max(1, total // args.n)

    xs = np.zeros((args.n, 224, 224, 3), dtype=np.uint8)
    ys = np.zeros(args.n, dtype=np.int64)
    picked, base = 0, 0

    for gi in range(pf.num_row_groups):
        table = pf.read_row_group(gi)
        names = table.column_names
        img_col = find_col(names, ["image", "img", "image_bytes"])
        lab_col = find_col(names, ["label", "labels", "class"])
        rows = table.to_pylist()
        for j, r in enumerate(rows):
            if (base + j) % step != 0 or picked >= args.n:
                continue
            img = r[img_col]
            if isinstance(img, dict):
                img = img.get("bytes") or img.get("path")
            im = Image.open(io.BytesIO(img)) if isinstance(img, bytes) else Image.open(img)
            im = im.convert("RGB")
            # 与 torchvision 评测一致：短边缩放到 256，再中心裁剪 224
            w, h = im.size
            scale = 256.0 / min(w, h)
            im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
            left, top = (im.width - 224) // 2, (im.height - 224) // 2
            im = im.crop((left, top, left + 224, top + 224))
            xs[picked] = np.asarray(im, dtype=np.uint8)
            ys[picked] = int(r[lab_col])
            picked += 1
        base += table.num_rows
        if picked >= args.n:
            break

    xs, ys = xs[:picked], ys[:picked]
    out = os.path.join(ROOT, "data", "imagenet.npz")
    np.savez_compressed(out, x=xs, y=ys)
    print(f"saved {out}: {xs.shape}, classes={len(np.unique(ys))}, "
          f"size={os.path.getsize(out)//1024//1024} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ImageNet parquet to npz.")
    parser.add_argument("--src", default=os.path.join(ROOT, "data", "imagenet", "train-00000.parquet"))
    parser.add_argument("--n", default=1000, type=int)
    main(parser.parse_args())
