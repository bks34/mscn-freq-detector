"""检测器公共流程：特征提取（带缓存）→ 划分 → SVM → 按固定 FPR 评测。

与综述代码 detect_nss.py 的协议对齐，但修掉了两个问题：
  1. 干净特征严格按 70/30 划分，评测不再用到训练样本（原代码有泄漏）；
  2. 阈值取自“训练集干净样本”的分位数，而不是测试集。
"""

from __future__ import division, absolute_import, print_function

import hashlib
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm

from common.util import (dr_at_fpr, ensure_dir, evalulate_detection_test, get_device,
                         set_seed, threshold_at_fpr, write_result_csv)
from features.fusion import build_feature_vector
from setup_paths import ADV_DIR, RESULT_DIR, SEED, TARGET_FPR, TRAIN_RATIO


# ------------------------------------------------------------------ 缓存
def cfg_hash(cfg):
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def feature_path(dataset, tag, kind, attack=None, cfg=None):
    """缓存文件名带配置哈希，避免换特征配置后误用旧缓存。"""
    folder = ensure_dir(os.path.join(RESULT_DIR, "features"))
    key = f"{dataset}_{tag}_{kind}" + (f"_{attack}" if attack else "")
    if cfg is not None:
        key += "_" + cfg_hash(cfg)
    return os.path.join(folder, key + ".npy")


def extract(images, cfg, desc="features"):
    feats = np.zeros((len(images), 0), dtype=np.float32)
    for i, img in enumerate(tqdm(images, desc=desc, ncols=80)):
        f = build_feature_vector(img, cfg)[None, :]
        if i == 0:
            feats = np.zeros((len(images), f.shape[1]), dtype=np.float32)
        feats[i] = f
    return feats


def get_features(images, cfg, path, desc="features", force=False):
    if os.path.isfile(path) and not force:
        arr = np.load(path)
        if arr.shape[0] == len(images):
            return arr
    arr = extract(images, cfg, desc=desc)
    np.save(path, arr)
    return arr


# ------------------------------------------------------------------ SVM
def fit_svm(X, y, seed=SEED, tune=True):
    """小网格搜索 + 3 折交叉验证；样本少时退化为固定参数。"""
    if not tune or len(y) < 200:
        clf = SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced")
        clf.fit(X, y)
        return clf

    best = (-1.0, None)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    for C in (0.1, 1.0, 10.0, 100.0):
        for gamma in ("scale", 0.01, 0.1):
            scores = []
            for tr_idx, va_idx in skf.split(X, y):
                clf = SVC(C=C, kernel="rbf", gamma=gamma, class_weight="balanced")
                clf.fit(X[tr_idx], y[tr_idx])
                pred = clf.predict(X[va_idx])
                scores.append((pred == y[va_idx]).mean())
            mean_acc = float(np.mean(scores))
            if mean_acc > best[0]:
                best = (mean_acc, dict(C=C, gamma=gamma))
    clf = SVC(**best[1], kernel="rbf", class_weight="balanced")
    clf.fit(X, y)
    print(f"  [svm] best params {best[1]}  cv_acc={best[0]:.4f}")
    return clf


# ------------------------------------------------------------------ 主流程
def run(dataset, cfg, tag, train_attack, fpr=TARGET_FPR, limit=None,
        force_features=False, tune_svm=True, attacks=None):
    set_seed(SEED)
    import torch
    from common.data import get_test_arrays
    from models.cnn import load_model

    print(f"[{tag}] dataset={dataset} train_attack={train_attack} fpr={fpr}")
    device = get_device()
    model = load_model(dataset, device=device)

    images, labels = get_test_arrays(dataset, limit=limit)
    print(f"  test images: {images.shape}")

    # ---- 干净样本预测（224×224 时单批不能太大，避免显存不足）
    bs = 64 if dataset == "imagenet" else 256
    with torch.no_grad():
        x = torch.from_numpy(np.transpose(images, (0, 3, 1, 2)) if images.ndim == 4
                             else images[:, None]).float().to(device)
        logits = []
        for i in range(0, x.size(0), bs):
            logits.append(model(x[i:i + bs]).argmax(1).cpu())
        preds_clean = torch.cat(logits).numpy()

    correct = np.where(preds_clean == labels)[0]
    print(f"  clean correct: {len(correct)}/{len(labels)}")

    # ---- 干净特征
    f_clean = get_features(images, cfg, feature_path(dataset, tag, "clean", cfg=cfg),
                           desc=f"{tag}: clean", force=force_features)

    # ---- 训练攻击的对抗特征
    adv_attack = os.path.join(ADV_DIR, f"{dataset}_{train_attack}.npy")
    if not os.path.isfile(adv_attack):
        raise FileNotFoundError(f"缺少 {adv_attack}，请先运行 generate_adv.py -d={dataset}")
    adv_train_imgs = np.load(adv_attack)
    if len(adv_train_imgs) < len(images):
        raise ValueError(f"{adv_attack} 只有 {len(adv_train_imgs)} 张，少于测试集 {len(images)} 张；"
                         f"请统一 generate_adv.py 与检测脚本的 --limit")
    f_adv_train = get_features(adv_train_imgs[:len(images)], cfg,
                               feature_path(dataset, tag, "adv", train_attack, cfg=cfg),
                               desc=f"{tag}: {train_attack}", force=force_features)

    # ---- 划分（在“干净分类正确”的样本上）
    rng = np.random.RandomState(SEED)
    idx = correct.copy()
    rng.shuffle(idx)
    n_train = int(len(idx) * TRAIN_RATIO)
    tr_idx, te_idx = idx[:n_train], idx[n_train:]

    X_tr = np.concatenate([f_clean[tr_idx], f_adv_train[tr_idx]])
    y_tr = np.concatenate([np.zeros(len(tr_idx)), np.ones(len(tr_idx))])
    clf = fit_svm(X_tr, y_tr, tune=tune_svm)
    thr = threshold_at_fpr(clf.decision_function(f_clean[tr_idx]), fpr)
    print(f"  threshold @FPR={fpr}: {thr:.4f}")

    # ---- 逐攻击评测
    from setup_paths import ATTACKS
    attacks = attacks or ATTACKS[dataset]
    out_dir = ensure_dir(os.path.join(RESULT_DIR, tag))
    summary = []

    for attack in attacks:
        path = os.path.join(ADV_DIR, f"{dataset}_{attack}.npy")
        if not os.path.isfile(path):
            print(f"  [skip] missing {path}")
            continue
        adv_imgs = np.load(path)[:len(images)]
        f_adv = get_features(adv_imgs, cfg, feature_path(dataset, tag, "adv", attack, cfg=cfg),
                             desc=f"{tag}: {attack}", force=force_features)

        with torch.no_grad():
            xa = torch.from_numpy(np.transpose(adv_imgs, (0, 3, 1, 2)) if adv_imgs.ndim == 4
                                  else adv_imgs[:, None]).float().to(device)
            pa = []
            for i in range(0, xa.size(0), bs):
                pa.append(model(xa[i:i + bs]).argmax(1).cpu())
            preds_adv = torch.cat(pa).numpy()

        rows = []
        for ttype in ("all", "success"):
            if ttype == "all":
                sel = te_idx
            else:
                sel = te_idx[preds_adv[te_idx] != labels[te_idx]]
            if len(sel) == 0:
                continue
            clean_test = f_clean[sel]
            adv_test = f_adv[sel]
            s_clean = clf.decision_function(clean_test)
            s_adv = clf.decision_function(adv_test)
            _, dr, auc, real_fpr = dr_at_fpr(s_clean, s_adv, fpr)
            pred = np.concatenate([s_clean > thr, s_adv > thr]).astype(int)
            y = np.concatenate([np.zeros(len(sel)), np.ones(len(sel))]).astype(int)
            acc, tpr, fpr_v, tp, ap, fp, an = evalulate_detection_test(y, pred)
            rows.append(dict(type=ttype, nsamples=len(sel),
                             acc_suc=float(np.mean(preds_adv[sel] != labels[sel])),
                             acc=acc, tpr=dr, fpr=real_fpr, tp=tp, ap=ap,
                             fb=len(sel) - tp, an=an, auc=auc))
            print(f"  {attack:16s} [{ttype:7s}] n={len(sel):4d}  "
                  f"DR@FPR{int(fpr * 100)}={dr * 100:6.2f}%  AUC={auc * 100:6.2f}%  "
                  f"cleanFPR={real_fpr * 100:5.2f}%")

        write_result_csv(os.path.join(out_dir, f"{dataset}_{attack}.csv"), rows)
        summary.extend([dict(attack=attack, **r) for r in rows])

    if summary:
        write_result_csv(os.path.join(out_dir, f"summary_{dataset}.csv"), summary)
    print(f"[{tag}] done -> {out_dir}")
