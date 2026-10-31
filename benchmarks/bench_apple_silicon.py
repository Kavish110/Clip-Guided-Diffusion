"""Benchmarks the diffusion UNet (guided-diffusion 64x64, 296M params) and a CLIP fine-tuning step
under the Apple Silicon optimisations in this repo.

    python benchmarks/bench_apple_silicon.py --out benchmarks/results_apple_silicon.json
"""
import argparse
import contextlib
import json
import os
import platform
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.improved_ddpm.unet as unet  # noqa: E402
from models.improved_ddpm.script_util import i_DDPM  # noqa: E402
from utils.diffusion_utils import get_beta_schedule, denoising_step  # noqa: E402


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, warmup=2, iters=10):
    for _ in range(warmup):
        fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - start) / iters


def autocast(device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def bench_config(name, device, sdpa=True, amp_dtype=None, channels_last=False, batch=1, iters=10, clip_model=None,
                 attn_recompute=False):
    unet.USE_SDPA = sdpa
    model = i_DDPM("TINY_IMAGENET")
    model.load_state_dict(torch.load("pretrained/64x64_diffusion.pt", map_location="cpu"))
    if attn_recompute:
        # reproduces the original AttentionBlock, which always gradient-checkpointed itself
        for m in model.modules():
            if isinstance(m, unet.AttentionBlock):
                m.use_checkpoint = True
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    betas = torch.from_numpy(get_beta_schedule(beta_start=1e-4, beta_end=0.02, num_diffusion_timesteps=1000,
                                               beta_schedule="cosine")).float().to(device)
    x = torch.randn(batch, 3, 64, 64, device=device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    t = torch.full((batch,), 400.0, device=device)
    t_next = torch.full((batch,), 300.0, device=device)
    res = {"name": name, "device": device.type, "sdpa": sdpa, "amp": str(amp_dtype).replace("torch.", ""),
           "channels_last": channels_last, "attn_recompute": attn_recompute, "batch": batch}

    # 1) inference: one DDIM step (UNet forward + update)
    model.eval()

    def infer():
        with torch.no_grad(), autocast(device, amp_dtype):
            return denoising_step(x, t=t, t_next=t_next, models=model, logvars=None, b=betas,
                                  sampling_type="ddim", learn_sigma=True)
    res["ddim_step_ms"] = 1000 * timeit(infer, device, iters=iters)

    # 2) GPU-efficient fine-tuning step: UNet fwd + CLIP directional loss + backward + Adam
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-9)
    x0 = torch.randn(batch, 3, 64, 64, device=device).clamp(-1, 1)

    def train():
        opt.zero_grad(set_to_none=True)
        with autocast(device, amp_dtype):
            _, x0_t = denoising_step(x, t=t, t_next=t_next, models=model, logvars=None, b=betas,
                                     sampling_type="ddim", learn_sigma=True, out_x0_t=True)
            loss = (2 - clip_model(x0, "photo", x0_t.float(), "watercolor art")) / 2 if clip_model else x0_t.square().mean()
            loss = -torch.log(loss) if clip_model else loss
        loss.backward()
        opt.step()
    res["finetune_step_ms"] = 1000 * timeit(train, device, warmup=2, iters=max(3, iters // 2))

    # numerical drift w.r.t. fp32 einsum reference on the same device
    del opt
    model.eval()
    return model, res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/results_apple_silicon.json")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--skip_cpu", action="store_true")
    args = parser.parse_args()

    from losses.clip_loss import CLIPLoss
    torch.manual_seed(0)
    mps = torch.device("mps")
    cpu = torch.device("cpu")
    clip_mps = CLIPLoss(mps, lambda_direction=1, clip_model="ViT-B/16")
    # same weights, but with CLIP's original Conv2d patch embedding (slow MPS backward)
    clip_mps_conv = CLIPLoss(mps, lambda_direction=1, clip_model="ViT-B/16")
    patch = clip_mps_conv.model.visual.conv1
    conv = torch.nn.Conv2d(3, patch.weight.shape[0], patch.patch, stride=patch.patch, bias=False).to(mps)
    conv.weight = patch.weight
    clip_mps_conv.model.visual.conv1 = conv
    clip_cpu = None if args.skip_cpu else CLIPLoss(cpu, lambda_direction=1, clip_model="ViT-B/16")

    # each MPS row adds one change on top of the previous one (channels_last / autocast rows branch off the last kept config)
    configs = []
    for batch in (1, 8):
        if not args.skip_cpu:
            configs.append(dict(name="original: CPU fp32", device=cpu, sdpa=False,
                                attn_recompute=True, clip_model=clip_cpu, iters=3, batch=batch))
        configs += [
            dict(name="MPS fp32 (naive port)", device=mps, sdpa=False, attn_recompute=True,
                 clip_model=clip_mps_conv, batch=batch),
            dict(name="+ no forced attention recompute", device=mps, sdpa=False, clip_model=clip_mps_conv, batch=batch),
            dict(name="+ SDPA attention", device=mps, sdpa=True, clip_model=clip_mps_conv, batch=batch),
            dict(name="+ CLIP patch-embed as matmul", device=mps, sdpa=True, batch=batch),
            dict(name="  + channels_last (rejected)", device=mps, sdpa=True, channels_last=True, batch=batch),
            dict(name="  + bf16 autocast", device=mps, sdpa=True, amp_dtype=torch.bfloat16, batch=batch),
            dict(name="  + fp16 autocast", device=mps, sdpa=True, amp_dtype=torch.float16, batch=batch),
        ]

    # reference output for numerical error of each config
    unet.USE_SDPA = False
    ref_model = i_DDPM("TINY_IMAGENET")
    ref_model.load_state_dict(torch.load("pretrained/64x64_diffusion.pt", map_location="cpu"))
    ref_model.eval()
    gen = torch.Generator().manual_seed(1)
    probe_x = torch.randn(2, 3, 64, 64, generator=gen)
    probe_t = torch.full((2,), 400.0)
    with torch.no_grad():
        ref_out = ref_model(probe_x, probe_t)
    del ref_model

    results = []
    for cfg in configs:
        cfg = dict(cfg)
        cfg.setdefault("clip_model", clip_mps)
        cfg.setdefault("iters", args.iters)
        model, res = bench_config(**cfg)
        dev = cfg["device"]
        x = probe_x.to(dev)
        if cfg.get("channels_last"):
            x = x.to(memory_format=torch.channels_last)
        with torch.no_grad(), autocast(dev, cfg.get("amp_dtype")):
            out = model(x, probe_t.to(dev)).float().cpu()
        res["max_abs_err_vs_cpu_fp32"] = (out - ref_out).abs().max().item()
        print(json.dumps(res))
        results.append(res)
        del model
        if dev.type == "mps":
            torch.mps.empty_cache()

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    meta = {"chip": chip, "macos": platform.mac_ver()[0], "torch": torch.__version__, "python": platform.python_version()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
