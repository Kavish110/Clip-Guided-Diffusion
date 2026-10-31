"""Scores the test-set edits written by `main.py --clip_finetune(_eff)`.

Uses a CLIP model *different* from the one used as the training loss (ViT-B/32 vs ViT-B/16),
so a method cannot score well just by over-fitting the guidance model.

    python benchmarks/eval_edits.py --run_dir runs/<exp> --src_txt photo --trg_txt "watercolor art"
"""
import argparse
import json
import os
import re
import sys
from glob import glob, escape as glob_escape

import clip
import lpips
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_templates import imagenet_templates  # noqa: E402


class EditScorer:
    def __init__(self, device, clip_model="ViT-B/32"):
        self.device = device
        self.model, self.preprocess = clip.load(clip_model, device=device)
        self.model = self.model.float().eval()
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def text_features(self, text):
        tokens = clip.tokenize([t.format(text) for t in imagenet_templates]).to(self.device)
        f = self.model.encode_text(tokens).float()
        f = f / f.norm(dim=-1, keepdim=True)
        f = f.mean(0)
        return f / f.norm()

    @torch.no_grad()
    def image_features(self, paths):
        x = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in paths]).to(self.device)
        f = self.model.encode_image(x).float()
        return f / f.norm(dim=-1, keepdim=True)

    @staticmethod
    def to_tensor(paths):
        x = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 127.5 - 1 for p in paths])
        return torch.from_numpy(x).permute(0, 3, 1, 2)

    @torch.no_grad()
    def score(self, src_paths, edit_paths, src_txt, trg_txt):
        src_f, edit_f = self.image_features(src_paths), self.image_features(edit_paths)
        t_src, t_trg = self.text_features(src_txt), self.text_features(trg_txt)
        text_dir = (t_trg - t_src) / (t_trg - t_src).norm()
        img_dir = edit_f - src_f
        img_dir = img_dir / img_dir.norm(dim=-1, keepdim=True)
        src_x, edit_x = self.to_tensor(src_paths).to(self.device), self.to_tensor(edit_paths).to(self.device)
        per_image = {
            "clip_direction": (img_dir @ text_dir).cpu(),
            "clip_target": (edit_f @ t_trg).cpu(),
            "clip_target_gain": (edit_f @ t_trg - src_f @ t_trg).cpu(),
            "lpips": self.lpips(src_x, edit_x).flatten().cpu(),
            "l1": (src_x - edit_x).abs().mean((1, 2, 3)).cpu(),
        }
        return {k: {"mean": v.mean().item(), "std": v.std().item()} for k, v in per_image.items()}, per_image


def find_test_pairs(image_folder, trg_txt, it=None, n_test_step=None):
    """(original, edited) image paths for the test set at outer iteration `it` (default: last)."""
    trg = trg_txt.replace(" ", "_")
    pattern = re.compile(rf"test_(\d+)_2_clip_{re.escape(trg)}_(\d+)_ngen(\d+)\.png$")
    found = {}
    # run directories embed the prompt list, e.g. "..._['watercolor art']_...", so escape glob brackets
    for path in glob(os.path.join(glob_escape(image_folder), "test_*_2_clip_*.png")):
        m = pattern.search(os.path.basename(path))
        if m and (n_test_step is None or int(m.group(3)) == n_test_step):
            found.setdefault(int(m.group(2)), {})[int(m.group(1))] = path
    if not found:
        raise FileNotFoundError(f"no test edits for '{trg_txt}' in {image_folder}")
    it = max(found) if it is None else it
    steps = sorted(found[it])
    return ([os.path.join(image_folder, f"test_{k}_0_orig.png") for k in steps],
            [found[it][k] for k in steps], it)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--src_txt", required=True)
    parser.add_argument("--trg_txt", required=True)
    parser.add_argument("--it", type=int, default=None)
    parser.add_argument("--n_test_step", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    scorer = EditScorer(device)
    src, edit, it = find_test_pairs(os.path.join(args.run_dir, "image_samples"), args.trg_txt, args.it, args.n_test_step)
    summary, _ = scorer.score(src, edit, args.src_txt, args.trg_txt)
    print(json.dumps({"run_dir": args.run_dir, "it": it, "n_images": len(src), **summary}, indent=2))


if __name__ == "__main__":
    main()
