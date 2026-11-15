"""Audit a frozen ImageNet-64 EDM denoiser on labeled validation images."""

import argparse
import json
import math
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from audit_baselines import lattice_predictors
from download_edm import CHECKPOINT, SYNSETS
from floor_core import cifar_rounding_risk_bound


FIELDS = ("dsm", "stein", "rounding", "soft_discrete",
          "dsm_minus_rounding", "dsm_minus_soft_discrete")


def load_validation(manifest_path, synsets_path, limit):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["dataset"], manifest["split"]) != ("ILSVRC/imagenet-1k", "validation"):
        raise ValueError("Expected an ILSVRC ImageNet validation manifest")
    validation_synsets = synsets_path.read_text(encoding="utf-8").split()
    classes = sorted(set(validation_synsets))
    if len(validation_synsets) != 50_000 or len(classes) != 1000:
        raise ValueError("Expected 50,000 validation synsets spanning 1,000 classes")
    class_indices = {wnid: index for index, wnid in enumerate(classes)}
    rows = manifest["images"]
    if limit is not None:
        if limit > len(rows):
            raise ValueError(f"Requested {limit} images but manifest has only {len(rows)}")
        rows = rows[:limit]
    if len(rows) < 2:
        raise ValueError("At least two images are needed to estimate standard errors")
    images, labels, ids = [], [], []
    for row in rows:
        match = re.fullmatch(
            r"ILSVRC2012_val_(\d{8})(?:_(n\d{8}))?\.[^.]+",
            row["source_name"], re.IGNORECASE,
        )
        if not match or not 1 <= int(match[1]) <= len(validation_synsets):
            raise ValueError(f"Invalid ImageNet validation filename: {row['source_name']}")
        image_id = int(match[1])
        wnid = validation_synsets[image_id - 1]
        if match[2] and match[2].lower() != wnid:
            raise ValueError(
                f"Synset in {row['source_name']} disagrees with validation mapping ({wnid})"
            )
        labels.append(class_indices[wnid])
        ids.append(image_id)
        with Image.open(manifest_path.parent / row["file"]) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            crop = min(width, height)
            left, top = (width - crop) // 2, (height - crop) // 2
            resized = rgb.crop((left, top, left + crop, top + crop)).resize(
                (64, 64), Image.Resampling.LANCZOS
            )
            images.append(np.asarray(resized))
    if len(set(ids)) != len(ids):
        raise ValueError("Validation subset contains duplicate image IDs")
    pixels = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    return pixels, torch.tensor(labels), ids, manifest["revision"]


def load_network(edm_dir, model_dir, device):
    edm_dir = edm_dir.resolve()
    if not (edm_dir / "torch_utils" / "persistence.py").is_file():
        raise ValueError(f"Missing NVlabs EDM checkout at {edm_dir}")
    sys.path.insert(0, str(edm_dir))
    # EDM pickle loading executes serialized Python. Use the official checkpoint.
    with (model_dir / CHECKPOINT).open("rb") as f:
        network = pickle.load(f)["ema"]
    if (network.img_resolution, network.img_channels, network.label_dim) != (64, 3, 1000):
        raise ValueError("Unexpected ImageNet-64 model dimensions")
    return network.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def evaluate(network, pixels, labels, grid, batch_size, device, seed, fp16):
    noise = torch.Generator(device="cpu").manual_seed(seed)
    values = np.empty((len(grid), len(pixels), len(FIELDS)))
    for node, lam in enumerate(grid):
        sigma = math.exp(-lam / 2)
        for start in range(0, len(pixels), batch_size):
            stop = start + batch_size
            y = pixels[start:stop].to(device=device, dtype=torch.float32) / 127.5 - 1
            cls = F.one_hot(labels[start:stop], num_classes=1000).to(device, dtype=torch.float32)
            eps = torch.randn(y.shape, generator=noise).to(device)
            x = y + sigma * eps
            denoised = network(x, torch.full((len(y),), sigma, device=device), cls,
                               force_fp32=not fp16)
            prediction = (x.double() - denoised.double()) / sigma
            hard, soft = lattice_predictors(x, sigma, radius=max(2, math.ceil(8 * 127.5 * sigma)))
            dsm, rounding, discrete = [
                (eps - predictor).square().flatten(1).mean(1)
                for predictor in (prediction, hard, soft)
            ]
            stein = 1 - (eps * prediction).flatten(1).mean(1)
            values[node, start:stop] = torch.stack(
                (dsm, stein, rounding, discrete, dsm - rounding, dsm - discrete), dim=1
            ).cpu().numpy()
        print(f"{node + 1}/{len(grid)}  sigma={sigma:.6g}  DSM/dim={values[node, :, 0].mean():.4f}",
              flush=True)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/imagenet_val_512/manifest.json"))
    parser.add_argument("--edm-dir", type=Path, default=Path("models/edm"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--n-grid", type=int, default=9)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true", help="allow EDM's native FP16 forward")
    args = parser.parse_args()
    
    if args.batch < 1 or args.n_grid < 2 or (args.limit is not None and args.limit < 2):
        parser.error("need a positive batch size, at least two grid points and at least two images")
    
    if not 0 < args.sigma_min < args.sigma_max <= 0.01:
        parser.error("require 0 < sigma-min < sigma-max <= 0.01 for the lattice-tail audit")
    
    device = torch.device(args.device)
    
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")
    
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")

    pixels, labels, ids, revision = load_validation(args.manifest, args.model_dir / SYNSETS, args.limit)
    grid = np.linspace(-2 * math.log(args.sigma_max), -2 * math.log(args.sigma_min), args.n_grid)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    start_time = time.perf_counter()
    network = load_network(args.edm_dir, args.model_dir, device)
    values = evaluate(network, pixels, labels, grid, args.batch, device, args.seed, args.fp16)
    
    # Integrate each image first so paired differences retain cross-node covariance.
    integrated = 0.5 * np.trapezoid(values, grid, axis=0)
    means = integrated.mean(axis=0)
    se = integrated.std(axis=0, ddof=1) / math.sqrt(len(pixels))
    
    summary = {name: {"mean": float(mean), "se": float(err)}
               for name, mean, err in zip(FIELDS, means, se)}
    
    bound_grid = np.linspace(grid[0], grid[-1], 1001)
    
    rounding_bound = 0.5 * np.trapezoid(cifar_rounding_risk_bound(bound_grid, dimension=1), bound_grid)
   
    peak_gib = torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else None
    
    result = {
        "checkpoint": CHECKPOINT, "dataset_revision": revision, "validation_ids": ids,
        "edm_labels": labels.tolist(), "seed": args.seed, "fp16_requested": args.fp16,
        "log_snr": grid.tolist(), "sigma": np.exp(-grid / 2).tolist(),
        "fields": FIELDS, "curve_mean_per_dim": values.mean(axis=1).tolist(),
        "se_scope": "Paired across validation images and sampled noise. Excludes quadrature error.",
        "integrated_nats_per_dim": summary,
        "rounding_bound_quadrature_nats_per_dim": float(rounding_bound),
        "peak_cuda_reserved_gib": peak_gib, "elapsed_seconds": time.perf_counter() - start_time,
    }
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    
    print(json.dumps(summary, indent=2))
    
    if peak_gib is not None:
        print(f"Peak CUDA reserved memory: {peak_gib:.2f} GiB")
    
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
