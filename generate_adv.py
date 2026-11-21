"""生成对抗样本（对应综述代码的 generate_adv.py，改用 torchattacks）。

用法：
    python generate_adv.py -d=cifar                  # 生成 setup_paths.ATTACKS['cifar'] 里的全部攻击
    python generate_adv.py -d=cifar -a=fgsm_0.03125  # 只生成一种
    python generate_adv.py -d=cifar --max-samples=2000

产物：adv_data/{dataset}_{attack}.npy  （float32, [0,1], (N,H,W,C)）
      adv_data/{dataset}_{attack}.json （记录 N 与参数，供检测脚本对齐）
"""

from __future__ import division, absolute_import, print_function

import argparse
import json
import os

import numpy as np
import torch

from common.data import to_numpy
from common.util import get_device, set_seed, write_json
from models.cnn import load_model
from setup_paths import ADV_DIR, ATTACK_PARAMS, ATTACKS, DATASETS, SEED


def parse_attack(name):
    """'fgsm_0.03125' -> ('fgsm', 0.03125)；'cw_0' -> ('cw', 0.0)"""
    parts = name.rsplit("_", 1)
    return parts[0], float(parts[1]) if len(parts) == 2 else 0.0


def build_attack(name, eps, model):
    import torchattacks as ta

    kind, _ = parse_attack(name)
    p = dict(ATTACK_PARAMS.get(kind, {}))
    if kind == "fgsm":
        return ta.FGSM(model, eps=eps)
    if kind == "bim":
        return ta.BIM(model, eps=eps, alpha=eps / 4, steps=p.get("steps", 10))
    if kind == "pgd":
        return ta.PGD(model, eps=eps, alpha=eps / 4, steps=p.get("steps", 40), random_start=True)
    if kind == "cw":
        # 命名 cw_<c>：如 cw_10 表示 c=10 的更强 C&W；cw_0 退回默认 c=1
        c = eps if eps > 0 else p.get("c", 1.0)
        return ta.CW(model, c=c, kappa=0, steps=p.get("steps", 200), lr=p.get("lr", 0.01))
    if kind == "df":
        return ta.DeepFool(model, steps=p.get("steps", 50), overshoot=0.02)
    raise ValueError(f"未知攻击：{kind}")


def to_tensor(images):
    """(N,H,W,C) 或 (N,H,W) -> (N,C,H,W) tensor"""
    if images.ndim == 4:
        t = torch.from_numpy(np.transpose(images, (0, 3, 1, 2))).float()
    else:
        t = torch.from_numpy(images[:, None]).float()
    return t


def main(args):
    assert args.dataset in DATASETS
    set_seed(SEED)
    device = get_device()

    model = load_model(args.dataset, device=device)
    from common.data import get_test_arrays
    images, labels = get_test_arrays(args.dataset, limit=args.max_samples)
    x = to_tensor(images).to(device)
    y = torch.from_numpy(labels).long().to(device)
    print(f"[data] {args.dataset}: {x.shape}")

    attacks = ATTACKS[args.dataset] if not args.attack else [args.attack]
    for name in attacks:
        out_npy = os.path.join(ADV_DIR, f"{args.dataset}_{name}.npy")
        if os.path.isfile(out_npy) and not args.force:
            print(f"[skip] {out_npy} already exists (use --force)")
            continue
        kind, eps = parse_attack(name)
        print(f"[attack] {name} (kind={kind}, eps={eps})")
        atk = build_attack(name, eps, model)

        advs = []
        bs = args.batch_size
        for i in range(0, x.size(0), bs):
            xb, yb = x[i:i + bs], y[i:i + bs]
            adv = atk(xb, yb)
            advs.append(adv.detach().cpu())
        adv = torch.cat(advs).clamp(0, 1)
        arr = to_numpy(adv).astype(np.float32)
        np.save(out_npy, arr)
        write_json(out_npy.replace(".npy", ".json"),
                   {"dataset": args.dataset, "attack": name, "n": int(arr.shape[0]),
                    "shape": list(arr.shape), "eps": eps})
        print(f"  -> {out_npy}  {arr.shape}")

    # 保存与攻击文件对齐的测试标签，避免检测阶段重新取数时错位
    np.save(os.path.join(ADV_DIR, f"{args.dataset}_test_labels.npy"), labels)
    print("[done]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate adversarial examples with torchattacks.")
    parser.add_argument("-d", "--dataset", required=True, choices=DATASETS)
    parser.add_argument("-a", "--attack", default=None, help="如 fgsm_0.03125；默认生成全部")
    parser.add_argument("--max-samples", default=None, type=int, help="只取测试集前 N 张（快速实验）")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
