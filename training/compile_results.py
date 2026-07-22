"""Compile all training-run results into training/results/all_runs.csv.

Handles both:
  - Older results format: {"test_metrics": {...}, "config": {...}}
  - New results format:   {"test_default_t0.5": {...}, "test_tuned": {...}}
"""
import csv
import json
from pathlib import Path
# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

RUN_ROOTS = [
    config.RESULTS_DIR,      # new runs (head_seed*)
]

FIELDS = ["run_tag","source_dir","split_kind",
          "test_bal_acc","test_bal_acc_tuned","threshold_used","test_macro_f1",
          "test_roc_auc","test_pr_auc",
          "test_absent_recall","test_absent_precision",
          "test_present_recall","test_present_precision",
          "epochs","config_summary"]


def _pull(d, keys, default=None):
    for k in keys:
        if k in d: return d[k]
    return default


def load_result(path: Path):
    try:
        r = json.load(path.open())
    except Exception:
        return None
    cfg = r.get("config", {}) or {}
    tag = cfg.get("run_tag") or path.parent.name
    ep = len(r.get("history", []))

    # New format
    if "test_default_t0.5" in r:
        d = r["test_default_t0.5"]
        t = r.get("test_tuned", d)
        seed = r.get("seed") or cfg.get("seed")
        return {
            "run_tag": tag,
            "source_dir": str(path.parent),
            "split_kind": "composition-aware fixed splits",
            "test_bal_acc": round(d["balanced_acc"], 4),
            "test_bal_acc_tuned": round(t["balanced_acc"], 4),
            "threshold_used": round(t["t"], 3),
            "test_macro_f1": round(d.get("macro_f1", 0), 4),
            "test_roc_auc": round(d.get("roc_auc", 0), 4),
            "test_pr_auc": round(d.get("pr_auc", 0), 4),
            "test_absent_recall": round(d["rec_absent"], 4),
            "test_absent_precision": round(d["prec_absent"], 4),
            "test_present_recall": round(d["rec_present"], 4),
            "test_present_precision": round(d["prec_present"], 4),
            "epochs": ep,
            "config_summary": f"seed={seed}",
        }
    # Older format
    tm = r.get("test_metrics")
    if not tm: return None
    cs_bits = []
    for k in ("head","use_all_data","seed"):
        if k in cfg: cs_bits.append(f"{k}={cfg[k]}")
    return {
        "run_tag": tag,
        "source_dir": str(path.parent),
        "split_kind": "older / free-form",
        "test_bal_acc": round(tm.get("balanced_acc", 0), 4),
        "test_bal_acc_tuned": "",
        "threshold_used": 0.5,
        "test_macro_f1": round(tm.get("macro_f1", 0), 4),
        "test_roc_auc": round(tm.get("roc_auc", 0), 4),
        "test_pr_auc": round(tm.get("pr_auc", 0), 4),
        "test_absent_recall": round(tm.get("rec_absent", 0), 4),
        "test_absent_precision": round(tm.get("prec_absent", 0), 4),
        "test_present_recall": round(tm.get("rec_present", 0), 4),
        "test_present_precision": round(tm.get("prec_present", 0), 4),
        "epochs": ep,
        "config_summary": ";".join(cs_bits) or f"epochs={ep}",
    }


def main():
    rows = []
    for root in RUN_ROOTS:
        if not root.exists(): continue
        for sub in sorted(root.iterdir()):
            r = sub / "results.json"
            if not r.exists(): continue
            rec = load_result(r)
            if rec: rows.append(rec)
    print(f"found {len(rows)} runs")

    out = config.RESULTS_DIR / "all_runs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")

    rows.sort(key=lambda r: -r["test_bal_acc"])
    print(f"\ntop 10 by test_bal_acc:")
    for r in rows[:10]:
        tuned = f" tuned={r['test_bal_acc_tuned']}" if r['test_bal_acc_tuned'] != '' else ""
        print(f"  {r['test_bal_acc']:.4f}{tuned}  roc={r['test_roc_auc']:.3f}  {r['run_tag']}")


if __name__ == "__main__":
    main()
