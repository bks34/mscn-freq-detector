"""通用工具：随机种子、设备、指标与 CSV 读写。"""

from __future__ import division, absolute_import, print_function

import csv
import json
import os
import random

import numpy as np

FIELDNAMES = ["type", "nsamples", "acc_suc", "acc", "tpr", "fpr",
              "tp", "ap", "fb", "an", "auc"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device(prefer_gpu=True):
    import torch
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_tpr_fpr(true_labels, pred_labels):
    true_labels = np.asarray(true_labels).astype(bool)
    pred_labels = np.asarray(pred_labels).astype(bool)
    tp = int(np.sum(true_labels & pred_labels))
    an = int(np.sum(true_labels))
    ap = int(np.sum(pred_labels))
    fp = ap - tp
    fn = an - tp
    tpr = tp / an if an else 0.0
    fpr = fp / (fp + (len(true_labels) - an)) if (len(true_labels) - an) else 0.0
    return tpr, fpr, tp, ap, fp, fn


def evalulate_detection_test(y_true, y_pred):
    """返回 (acc, tpr, fpr, tp, ap, fp, an)。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    acc = float((y_true == y_pred).mean())
    tpr, fpr, tp, ap, fp, fn = get_tpr_fpr(y_true, y_pred)
    return acc, tpr, fpr, tp, ap, fp, int(np.sum(y_true))


def auc_score(y_true, scores):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y_true).astype(int), np.asarray(scores)))


def threshold_at_fpr(clean_scores, fpr=0.05):
    """取干净样本分数的 (1-fpr) 分位数作为阈值。"""
    return float(np.quantile(np.asarray(clean_scores), 1.0 - fpr))


def dr_at_fpr(clean_scores, adv_scores, fpr=0.05):
    """返回 (thr, dr, auc, real_fpr)。"""
    thr = threshold_at_fpr(clean_scores, fpr)
    dr = float(np.mean(np.asarray(adv_scores) > thr))
    real_fpr = float(np.mean(np.asarray(clean_scores) > thr))
    y = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(adv_scores))])
    s = np.concatenate([clean_scores, adv_scores])
    return thr, dr, auc_score(y, s), real_fpr


def write_result_csv(path, rows):
    ensure_dir(os.path.dirname(path))
    extra = []
    for row in rows:
        for k in row:
            if k not in FIELDNAMES and k not in extra:
                extra.append(k)
    fields = FIELDNAMES + extra
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def read_result_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
