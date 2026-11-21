"""汇总结果（对应综述代码的 collect_results_from_csv.py）。

把 results/<tag>/summary_<dataset>.csv 合并成一张“每个攻击一行”的宽表：
    results/<tag>/all_<dataset>.csv
以及跨数据集的总表：
    results/<tag>/all.csv
"""

from __future__ import division, absolute_import, print_function

import argparse
import csv
import os

from common.util import ensure_dir, read_result_csv
from setup_paths import DATASETS, RESULT_DIR


def collect_one(dataset, tag):
    path = os.path.join(RESULT_DIR, tag, f"summary_{dataset}.csv")
    if not os.path.isfile(path):
        return []
    rows = read_result_csv(path)
    out = {}
    for r in rows:
        attack = r.get("attack", "")
        key = (attack, r.get("type", "all"))
        out[key] = {
            "dataset": dataset,
            "attack": attack,
            "type": r.get("type", ""),
            "nsamples": r.get("nsamples", ""),
            "DR": float(r["tpr"]) if r.get("tpr") else "",
            "FPR": float(r["fpr"]) if r.get("fpr") else "",
            "AUC": float(r["auc"]) if r.get("auc") else "",
        }
    return list(out.values())


def write_csv(path, rows):
    ensure_dir(os.path.dirname(path))
    if not rows:
        print(f"[skip] nothing to write: {path}")
        return
    fields = ["dataset", "attack", "type", "nsamples", "DR", "FPR", "AUC"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[ok] {path}  ({len(rows)} rows)")


def main(args):
    tags = args.tags.split(",") if args.tags else ["nss", "mscn_freq"]
    for tag in tags:
        all_rows = []
        for ds in DATASETS:
            rows = collect_one(ds, tag)
            if rows:
                write_csv(os.path.join(RESULT_DIR, tag, f"all_{ds}.csv"), rows)
                all_rows.extend(rows)
        if all_rows:
            write_csv(os.path.join(RESULT_DIR, tag, "all.csv"), all_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect per-attack CSVs into summary tables.")
    parser.add_argument("--tags", default=None, help="逗号分隔，如 nss,mscn_freq")
    main(parser.parse_args())
