"""Fine-tunes the Tiny-ImageNet diffusion model with the standard directional CLIP loss and with the
optimal-transport (Sinkhorn) directional loss on the same train/test images, then scores the edits.

    python benchmarks/compare_ot.py            # runs everything, writes benchmarks/results_ot.json + assets/
"""
import argparse
import json
import os
import subprocess
import sys
import time
from glob import glob, escape as glob_escape

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from benchmarks.eval_edits import EditScorer, find_test_pairs  # noqa: E402

PROMPTS = [("photo", "watercolor art"), ("photo", "pencil sketch"), ("photo", "painting by Gogh")]


def slug(text):
    return text.replace(" ", "_")


def run_dir_for(exp):
    dirs = glob(os.path.join(glob_escape(ROOT), glob_escape(exp) + "_FT_*"))
    assert len(dirs) == 1, (exp, dirs)
    return dirs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="direction,ot,direction+ot")
    parser.add_argument("--lambda_ot", type=float, default=1.0)
    # the OT gradient is ~2.4x smaller than the directional one at the start of fine-tuning (measured on
    # Tiny-ImageNet val images); Adam makes the absolute scale irrelevant for OT alone, but not in the sum
    parser.add_argument("--lambda_ot_hybrid", type=float, default=2.5)
    parser.add_argument("--t_0", type=int, default=500)
    parser.add_argument("--n_train_img", type=int, default=40)
    parser.add_argument("--n_test_img", type=int, default=16)
    parser.add_argument("--n_iter", type=int, default=5)
    parser.add_argument("--lr", type=float, default=4e-6)
    parser.add_argument("--figure_rows", type=int, default=6)
    parser.add_argument("--out", default="benchmarks/results_ot.json")
    args = parser.parse_args()

    methods = {
        "direction": dict(lambda_direction=1.0, lambda_ot=0.0),
        "ot": dict(lambda_direction=0.0, lambda_ot=args.lambda_ot),
        "direction+ot": dict(lambda_direction=1.0, lambda_ot=args.lambda_ot_hybrid),
    }
    methods = {k: methods[k] for k in args.methods.split(",")}
    n_precomp = max(args.n_train_img, args.n_test_img)

    results = json.load(open(args.out)) if os.path.exists(args.out) else {"runs": []}
    done = {(r["trg_txt"], r["method"]) for r in results["runs"]}
    scorer = EditScorer(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))

    for src_txt, trg_txt in PROMPTS:
        for method, weights in methods.items():
            if (trg_txt, method) in done:
                continue
            exp = os.path.join("runs", f"cmp_{slug(trg_txt)}_{method.replace('+', '_')}")
            cmd = [sys.executable, "main.py", "--config", "tiny_imagenet.yml", "--clip_finetune_eff",
                   "--exp", exp, "--src_txts", src_txt, "--trg_txts", trg_txt,
                   "--t_0", str(args.t_0), "--n_inv_step", "40", "--n_train_step", "6", "--n_test_step", "40",
                   "--n_precomp_img", str(n_precomp), "--n_train_img", str(args.n_train_img),
                   "--n_test_img", str(args.n_test_img), "--n_iter", str(args.n_iter),
                   "--lr_clip_finetune", str(args.lr), "--clip_loss_w", "3", "--l1_loss_w", "0", "--id_loss_w", "0",
                   "--lambda_direction", str(weights["lambda_direction"]), "--lambda_ot", str(weights["lambda_ot"])]
            print(" ".join(cmd), flush=True)
            start = time.time()
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            wall = time.time() - start
            log = proc.stdout + proc.stderr
            train_times = [float(l.split("takes ")[1].rstrip("s")) for l in log.splitlines() if "Training for 1 image takes" in l]
            if proc.returncode != 0 or not train_times:
                raise RuntimeError(f"{exp} failed:\n{log[-3000:]}")

            run_dir = run_dir_for(exp)
            with open(os.path.join(run_dir, "train.log"), "w") as f:
                f.write(log)
            src, edit, it = find_test_pairs(os.path.join(run_dir, "image_samples"), trg_txt, n_test_step=40)
            summary, _ = scorer.score(src, edit, src_txt, trg_txt)
            results["runs"].append({
                "src_txt": src_txt, "trg_txt": trg_txt, "method": method, **weights, "run_dir": os.path.relpath(run_dir, ROOT),
                "it": it, "n_test_images": len(src), "wall_s": wall,
                "train_s_per_image": float(np.median(train_times)), **summary,
            })
            json.dump(results, open(args.out, "w"), indent=2)
            print(json.dumps(results["runs"][-1]), flush=True)

    results["settings"] = vars(args)
    add_paired_stats(results, scorer)
    json.dump(results, open(args.out, "w"), indent=2)
    write_table(results)
    for src_txt, trg_txt in PROMPTS:
        write_figure(results, trg_txt, list(methods), args.figure_rows)


def add_paired_stats(results, scorer, baseline="direction"):
    """Every method edits the same test images, so compare per image: mean difference vs. the baseline
    with a 95% t-interval (n = number of test images)."""
    from scipy import stats

    per_image = {}
    for r in results["runs"]:
        src, edit, _ = find_test_pairs(os.path.join(ROOT, r["run_dir"], "image_samples"), r["trg_txt"], r["it"], 40)
        per_image[(r["trg_txt"], r["method"])] = scorer.score(src, edit, r["src_txt"], r["trg_txt"])[1]
    for r in results["runs"]:
        if r["method"] == baseline:
            continue
        base, ours = per_image[(r["trg_txt"], baseline)], per_image[(r["trg_txt"], r["method"])]
        r["paired_vs_" + baseline] = {}
        for metric in ("clip_direction", "clip_target_gain", "lpips"):
            d = (ours[metric] - base[metric]).numpy()
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            r["paired_vs_" + baseline][metric] = {"mean": float(d.mean()), "ci95": float(half),
                                                  "p": float(stats.ttest_rel(ours[metric], base[metric]).pvalue)}


def write_table(results):
    lines = ["| Edit | Method | CLIP direction ↑ | CLIP target gain ↑ | LPIPS ↓ | L1 ↓ | Train s/img |",
             "|---|---|---|---|---|---|---|"]
    order = {trg: i for i, (_, trg) in enumerate(PROMPTS)}
    methods = ["direction", "ot", "direction+ot"]
    for r in sorted(results["runs"], key=lambda r: (order[r["trg_txt"]], methods.index(r["method"]))):
        lines.append(f"| {r['src_txt']} → {r['trg_txt']} | {r['method']} | {r['clip_direction']['mean']:.3f} | "
                     f"{r['clip_target_gain']['mean']:+.3f} | {r['lpips']['mean']:.3f} | {r['l1']['mean']:.3f} | "
                     f"{r['train_s_per_image']:.2f} |")
    with open(os.path.join(ROOT, "benchmarks", "results_ot.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


_VAL = None


def tiny_imagenet_class(png_path):
    """Class name of a saved test image, found by matching it against the labelled Tiny-ImageNet val set."""
    global _VAL
    if _VAL is None:
        root = os.path.join(ROOT, "data", "tiny-imagenet-200")
        rows = [line.split("\t")[:2] for line in open(os.path.join(root, "val", "val_annotations.txt"))]
        names = dict(line.rstrip("\n").split("\t") for line in open(os.path.join(root, "words.txt")))
        pixels = np.stack([np.asarray(Image.open(os.path.join(root, "val", "images", f)).convert("RGB"), dtype=np.int16)
                           for f, _ in rows])
        _VAL = (pixels.reshape(len(rows), -1), [names[w].split(",")[0] for _, w in rows])
    x = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.int16).reshape(-1)
    return _VAL[1][int(np.abs(_VAL[0] - x).sum(1).argmin())]


def write_figure(results, trg_txt, methods, n_rows, scale=2):
    """One row per test image, one image per class: source | each method."""
    runs = {r["method"]: r for r in results["runs"] if r["trg_txt"] == trg_txt}
    columns = []
    for i, method in enumerate(methods):
        src, edit, _ = find_test_pairs(os.path.join(ROOT, runs[method]["run_dir"], "image_samples"), trg_txt,
                                       runs[method]["it"], 40)
        if i == 0:
            columns.append(src)
        columns.append(edit)
    keep, classes = [], []
    for k, path in enumerate(columns[0]):
        name = tiny_imagenet_class(path)
        if name not in classes:
            keep.append(k)
            classes.append(name)
    keep, classes = keep[:n_rows], classes[:n_rows]
    columns = [[paths[k] for k in keep] for paths in columns]
    results.setdefault("figure_classes", {})[trg_txt] = classes
    json.dump(results, open(os.path.join(ROOT, "benchmarks", "results_ot.json"), "w"), indent=2)
    rows = len(keep)
    tile, pad = 64 * scale, 4
    canvas = Image.new("RGB", (len(columns) * (tile + pad) - pad, rows * (tile + pad) - pad), "white")
    for c, paths in enumerate(columns):
        for r in range(rows):
            img = Image.open(paths[r]).convert("RGB").resize((tile, tile), Image.NEAREST)
            canvas.paste(img, (c * (tile + pad), r * (tile + pad)))
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    canvas.save(os.path.join(ROOT, "assets", f"ot_comparison_{slug(trg_txt)}.png"), optimize=True)


if __name__ == "__main__":
    main()
