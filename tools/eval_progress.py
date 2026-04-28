"""
Eval moi checkpoint co san cua mot model -> in bang mAP qua time.
Dung de chuan doan hoi tu.

Luu y: tu khi co auto-cleanup, intermediate checkpoint cu se bi xoa khi co
intermediate moi luu. Vi vay neu muon track convergence, can set
SOLVER.CHECKPOINT_PERIOD nho hon de luu nhieu mid-train hon, hoac tat
auto-cleanup tam thoi.

Dung:
  python tools/eval_progress.py source_only_c4
  python tools/eval_progress.py source_only_c4 --split source_test
  python tools/eval_progress.py triplet_c4 --plot
"""
import argparse
import glob
import os
import re
import subprocess
import sys


CFG_MAP = {
    "source_only_c4":  "configs/da_faster_rcnn/fusionda_source_only_c4.yaml",
    "source_only_fpn": "configs/da_faster_rcnn/fusionda_source_only_fpn.yaml",
    "triplet_c4":      "configs/da_faster_rcnn/fusionda_triplet_c4.yaml",
    "triplet_fpn":     "configs/da_faster_rcnn/fusionda_triplet_fpn.yaml",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", choices=list(CFG_MAP.keys()))
    ap.add_argument("--split", default="target_test", choices=["target_test", "source_test"])
    ap.add_argument("--plot", action="store_true", help="plot mAP curve voi matplotlib")
    args = ap.parse_args()

    cfg = CFG_MAP[args.tag]
    ckpts = sorted(glob.glob(f"output/{args.tag}/model_*.pth"))
    if not ckpts:
        print(f"[ERROR] khong co checkpoint trong output/{args.tag}/", file=sys.stderr)
        sys.exit(1)

    ds = f"fusionda_{args.split}_cocostyle"
    gt = f"datasets/annotations/{args.split}.json"

    rows = []
    for ckpt in ckpts:
        name = os.path.basename(ckpt).replace(".pth", "").replace("model_", "")
        try:
            iteration = int(name)
        except ValueError:
            iteration = -1  # 'final'
        out_dir = f"output/{args.tag}/inference_progress/{name}_{args.split}"
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{args.tag}] eval ckpt={name} on {args.split}...")
        result = subprocess.run([
            "python", "tools/eval_standalone.py",
            "--config-file", cfg,
            "--weight", ckpt,
            "--test-dataset", ds,
            "--gt-json", gt,
            "--out-dir", out_dir,
        ], capture_output=True, text=True)

        out = result.stdout + result.stderr
        m_all = re.search(r"mAP \(0\.5:0\.95\):\s*([\d.]+)", out)
        m_50  = re.search(r"mAP50\s*:\s*([\d.]+)", out)
        m_75  = re.search(r"mAP75\s*:\s*([\d.]+)", out)
        if m_all and m_50:
            rows.append({
                "iter": iteration,
                "name": name,
                "mAP":   float(m_all.group(1)),
                "mAP50": float(m_50.group(1)),
                "mAP75": float(m_75.group(1)) if m_75 else 0.0,
            })
        else:
            print(f"  [WARN] khong parse duoc mAP, return code = {result.returncode}")

    # Print bang
    print("\n" + "=" * 60)
    print(f"  {args.tag}  ON  {args.split}")
    print("=" * 60)
    print(f"{'iter':<10} {'mAP':>8} {'mAP50':>8} {'mAP75':>8}")
    print("-" * 40)
    for r in sorted(rows, key=lambda x: x["iter"] if x["iter"] >= 0 else 9999999):
        print(f"{r['name']:<10} {r['mAP']:>8.2f} {r['mAP50']:>8.2f} {r['mAP75']:>8.2f}")

    # Phan tich
    if len(rows) >= 2:
        rows_sorted = sorted(rows, key=lambda x: x["iter"] if x["iter"] >= 0 else 9999999)
        last = rows_sorted[-1]["mAP50"]
        prev = rows_sorted[-2]["mAP50"]
        delta = last - prev
        print("\n--- Phan tich ---")
        if abs(delta) < 0.5:
            print(f"  delta_mAP50 = {delta:+.2f}  -> DA HOI TU (chenh < 0.5 AP)")
        elif delta > 0.5:
            print(f"  delta_mAP50 = {delta:+.2f}  -> con TANG -> chua hoi tu, can train them")
        else:
            print(f"  delta_mAP50 = {delta:+.2f}  -> dang TUT -> overfit, ckpt tot nhat la {rows_sorted[-2]['name']}")

    # Plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            rs = sorted(rows, key=lambda x: x["iter"] if x["iter"] >= 0 else 9999999)
            xs = [r["iter"] if r["iter"] >= 0 else (rs[-2]["iter"] + 7000 if len(rs) > 1 else 99999) for r in rs]
            plt.figure(figsize=(8, 5))
            plt.plot(xs, [r["mAP"]   for r in rs], "o-", label="mAP")
            plt.plot(xs, [r["mAP50"] for r in rs], "s-", label="mAP50")
            plt.xlabel("iteration")
            plt.ylabel("AP (%)")
            plt.title(f"{args.tag} on {args.split}")
            plt.legend(); plt.grid(True)
            out = f"output/{args.tag}/progress_{args.split}.png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            print(f"\n[SAVED] {out}")
        except ImportError:
            print("[WARN] matplotlib khong cai, bo qua plot")


if __name__ == "__main__":
    main()
