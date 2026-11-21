"""生成论文用的可视化图（MSCN 差异图与频域对比图）。

用法：
    python tools/make_figures.py --dataset imagenet   # 默认
    python tools/make_figures.py --dataset cifar

输出：figures/mscn_vis.pdf、figures/freq_vis.pdf（可用 --out 指定其他目录）
"""

from __future__ import division, absolute_import, print_function

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from features.freq import radial_power_spectrum, spectrum_features
from features.mscn import mscn_coefficients, to_gray
from models.cnn import load_model

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 6
plt.rcParams["axes.titlesize"] = 6
plt.rcParams["axes.labelsize"] = 6
plt.rcParams["xtick.labelsize"] = 5
plt.rcParams["ytick.labelsize"] = 5
plt.rcParams["legend.fontsize"] = 4.5

OUT_DIR = os.path.join(ROOT, "figures")

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
ATTACKS = {
    "cifar": ["fgsm_0.03125", "bim_0.03125", "pgd_0.03125", "cw_10"],
    "imagenet": ["fgsm_0.03125", "bim_0.03125", "pgd_0.03125", "cw_10"],
}
NAMES = ["FGSM", "BIM", "PGD", "C&W"]


def class_names(dataset):
    if dataset == "cifar":
        return CIFAR_CLASSES
    import torchvision.models as tvm
    return tvm.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


def load_data(dataset):
    if dataset == "cifar":
        x = np.load(os.path.join(ROOT, "data", "cifar10.npz"))["x_test"][:1000]
        y = np.load(os.path.join(ROOT, "adv_data", "cifar_test_labels.npy"))[:1000]
        prefix = "cifar"
    else:
        d = np.load(os.path.join(ROOT, "data", "imagenet.npz"))
        x, y = d["x"][:1000], d["y"][:1000]
        prefix = "imagenet"
    x = x.astype(np.float32) / 255.0
    adv = {a: np.load(os.path.join(ROOT, "adv_data", f"{prefix}_{a}.npy")).astype(np.float32)
           for a in ATTACKS[dataset]}
    return x, y, adv


def predict(model, imgs, bs=32):
    t = torch.from_numpy(np.transpose(imgs, (0, 3, 1, 2))).float()
    with torch.no_grad():
        return torch.cat([model(t[i:i + bs]).argmax(1)
                          for i in range(0, len(t), bs)]).numpy()


def mscn_figure(x, adv, pc, pa, y, names, path, dataset):
    attacks = ATTACKS[dataset]
    mask = (pc == y)
    for a in attacks:
        mask &= (pa[a] != y)
    cand = np.where(mask)[0]
    scores = [np.mean(np.abs(mscn_coefficients(adv[attacks[0]][i])
                             - mscn_coefficients(x[i]))) for i in cand[:150]]
    idx = int(cand[int(np.argmax(scores))])

    imgs = [x[idx]] + [adv[a][idx] for a in attacks]
    titles = ["(a) 原图", "(b) FGSM", "(c) BIM", "(d) PGD", "(e) C&W"]
    clean_mscn = mscn_coefficients(x[idx])

    fig = plt.figure(figsize=(5.6, 4.3) if dataset == "imagenet" else (5.6, 3.8))
    gs = GridSpec(3, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.06],
                  hspace=0.08, wspace=0.06)
    for j, (im, nm) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
        ax.set_title(nm, fontsize=6)
        ax.axis("off")
        cls = names[pc[idx]] if j == 0 else names[pa[attacks[j - 1]][idx]]
        ax.text(0.5, -0.06, cls, transform=ax.transAxes, ha="center", fontsize=5.5)

        ax2 = fig.add_subplot(gs[1, j])
        mm = mscn_coefficients(im)
        im2 = ax2.imshow(mm, cmap="gray", vmin=-3, vmax=3, interpolation="bilinear")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[2, j])
        d = np.abs(mm - clean_mscn) if j > 0 else np.zeros_like(mm)
        im3 = ax3.imshow(d, cmap="hot", vmin=0, vmax=2.5, interpolation="bilinear")
        ax3.axis("off")

    for r, lab in enumerate(["输入图像", "MSCN 系数 $\\hat{I}$", "$|\\Delta\\hat{I}|$"]):
        fig.text(0.015, 0.82 - r * 0.31, lab, rotation=90, va="center", fontsize=6)
    fig.colorbar(im2, cax=fig.add_subplot(gs[1, 5]))
    fig.colorbar(im3, cax=fig.add_subplot(gs[2, 5]))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return idx


def freq_figure(x, adv, path, dataset, n=300):
    attacks = ATTACKS[dataset]
    n_bins = 32 if dataset == "imagenet" else 16

    def logps(im):
        F = np.abs(np.fft.fftshift(np.fft.fft2(to_gray(im)))) ** 2
        return np.log10(F + 1e-12)

    P_clean = np.mean([logps(im) for im in x[:n]], axis=0)

    def radial_logps(imgs):
        acc, cnt = None, 0
        for im in imgs[:n]:
            F = np.abs(np.fft.fftshift(np.fft.fft2(to_gray(im)))) ** 2
            h, w = F.shape
            cy, cx = h // 2, w // 2
            yy, xx = np.indices((h, w))
            r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            bins = np.linspace(0, min(cy, cx), n_bins + 1)
            idx = np.digitize(r.ravel(), bins) - 1
            p = F.ravel()
            vals = []
            for b in range(n_bins):
                m = idx == b
                vals.append(np.log10(p[m].mean() + 1e-12) if m.any() else np.nan)
            vals = np.asarray(vals)
            acc = vals if acc is None else acc + vals
            cnt += 1
        return acc / cnt

    r_bins = np.linspace(0, 1, n_bins + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    lp_clean = radial_logps(x)
    lp_diff = {nm: radial_logps(adv[k]) - lp_clean for k, nm in zip(attacks, NAMES)}

    def avg_rad(imgs):
        ps = []
        for im in imgs[:n]:
            r, p = radial_power_spectrum(to_gray(im), n_bins=n_bins)
            ps.append(p / (p.sum() + 1e-12))
        return r, np.mean(ps, axis=0)

    def stats(imgs):
        b, s = [], []
        for im in imgs[:n]:
            bb, _, _, ss = spectrum_features(im)
            b.append(bb)
            s.append(ss)
        return np.mean(b), np.mean(s)

    b0, s0 = stats(x)
    rel = {}
    for k, nm in zip(attacks, NAMES):
        b, s = stats(adv[k])
        rel[nm] = ((b - b0) / abs(b0) * 100, (s - s0) / s0 * 100)

    fig = plt.figure(figsize=(5.6, 4.3))
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(P_clean, cmap="magma")
    ax.set_title("(a) 原图对数功率谱")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    for nm in NAMES:
        ax.plot(r_centers[1:], lp_diff[nm][1:], "o-", ms=3, label=nm)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("归一化径向频率")
    ax.set_ylabel("$\\Delta \\log_{10} P$")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("(b) 径向差分谱（相对原图）")

    ax = fig.add_subplot(gs[1, 0])
    r, p = avg_rad(x)
    ax.loglog(r[1:], p[1:], label="原图", lw=2)
    for k, nm in zip(attacks, NAMES):
        r, p = avg_rad(adv[k])
        ax.loglog(r[1:], p[1:], ls="--", label=nm)
    ax.set_xlabel("径向频率 r")
    ax.set_ylabel("归一化功率")
    ax.legend()
    ax.set_title("(c) 平均径向功率谱")

    ax = fig.add_subplot(gs[1, 1])
    xx = np.arange(len(NAMES))
    w = 0.35
    ax.bar(xx - w / 2, [rel[n][0] for n in NAMES], w, label="$\\beta$ 相对变化")
    ax.bar(xx + w / 2, [rel[n][1] for n in NAMES], w, label="SF 相对变化")
    ax.set_xticks(xx)
    ax.set_xticklabels(NAMES)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("相对变化 (%)")
    ax.legend()
    ax.set_title("(d) 谱斜率与平坦度的变化")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return rel


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--dataset", default="imagenet", choices=["imagenet", "cifar"])
    parser.add_argument("--n", default=300, type=int, help="频域统计平均的图像数")
    parser.add_argument("--out", default=None, help="输出目录（默认 figures/）")
    args = parser.parse_args()

    out_dir = args.out or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    x, y, adv = load_data(args.dataset)
    names = class_names(args.dataset)
    model = load_model(args.dataset)
    pc = predict(model, x)
    pa = {a: predict(model, adv[a]) for a in ATTACKS[args.dataset]}

    idx = mscn_figure(x, adv, pc, pa, y, names,
                      os.path.join(out_dir, "mscn_vis.pdf"), args.dataset)
    rel = freq_figure(x, adv, os.path.join(out_dir, "freq_vis.pdf"),
                      args.dataset, n=args.n)
    print(f"sample index = {idx}")
    for k, v in rel.items():
        print(f"  {k:5s} beta {v[0]:+.1f}%   SF {v[1]:+.1f}%")


if __name__ == "__main__":
    main()
