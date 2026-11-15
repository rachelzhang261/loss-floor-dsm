# Image denoising audits

The current manuscript uses a 240k-step CIFAR-10 network and a frozen
class-conditional ImageNet-64 EDM denoiser. Commands below run from this directory.
The results are already included; data and model weights are not bundled.

## CIFAR-10

CIFAR-10 is downloaded by torchvision when needed and scaled to its 8-bit lattice
in `[-1,1]`, without dequantization or augmentation. The wide VP training range is
log-SNR `[-8,13.8]`. Train the first 60k steps with:

```sh
python train.py --config wide --steps 60000 --out runs/wide
```

The original 60k checkpoint did not contain Adam state, so the reported extension
started with fresh Adam state and warmup. The script restores optimizer state
when present. For a fresh reproduction of the same extension protocol, create a
copy of the 60k checkpoint without the `optimizer` and `warmup_start` keys:

```sh
python -c "import torch; p='runs/wide/step060000.pt'; c=torch.load(p,map_location='cpu',weights_only=True); c.pop('optimizer',None); c.pop('warmup_start',None); torch.save(c,'runs/wide/step060000_weights_only.pt')"
python train.py --resume runs/wide/step060000_weights_only.pt --steps 240000 --out runs/wide_extended
```

Use the resulting checkpoint path consistently in the following commands.
The examples name the saved-run convention:

```sh
python evaluate.py --ckpt runs/wide_extended/step240000.pt --n-grid 219 --seed 1 --out results/reproduced_audit.json
python audit_baselines.py --ckpt runs/wide_extended/step240000.pt --out results/reproduced_baselines.json
python make_audit_figure.py --results results --out ../figures/fig_cifar_rebuttal.pdf
```

The last command redraws the paper from the supplied original records, not the
new `reproduced_*.json` files. It reads the two specific filenames documented in
`results/README.md`. The paired baseline audit uses 400 held-out image/noise draws
per node and seed 2027. The independent Jacobian audit uses `--seed 1`, one probe,
and 219 nodes. Standard errors describe evaluation sampling conditional on the
trained checkpoint; they do not include training-run variability or plug-in bias.

`floor_core.py` also contains a finite empirical-posterior calculation. Its
distribution is the finite training set, which is different from the held-out
population used in the image audits. `reanalyze.py` can summarize a saved
evaluation with explicit `--analysis` and `--out` arguments.

## ImageNet-64

The public checkpoint is `edm-imagenet-64x64-cond-adm.pkl`. Download the official
checkpoint and label mapping, and obtain the external code at the commit used
by the working evaluation environment:

```sh
python download_edm.py --out models
git clone https://github.com/NVlabs/edm.git models/edm
git -C models/edm checkout 008a4e5316c8e3bfe61a62f874bddba254295afb
```

Obtain access to `ILSVRC/imagenet-1k` on Hugging Face and authenticate with
`hf auth login`. Recreate the 512-image validation subset using the recorded
dataset revision and subset-sampling seed:

```sh
python download_imagenet.py --out data/imagenet_val_512 --count 512 --seed 42 --revision 49e2ee26f3810fb5a7536bbf732a7b07389a47b5
python evaluate_imagenet_edm.py --manifest data/imagenet_val_512/manifest.json --edm-dir models/edm --model-dir models --n-grid 9 --sigma-min 0.002 --sigma-max 0.003 --seed 2027 --out results/reproduced_imagenet.json
```

The download helper requires a new output directory, and the evaluator requires
a new output filename. The original result records all 512 validation IDs and
class indices. Inputs are center-cropped and resized with Lanczos to 64x64. The
default evaluation uses FP32; do not enable `--fp16` when reproducing the stated
protocol. Each image's differences are integrated before standard errors are
computed across images, preserving cross-node covariance.

The noise predictor uses the class label. Hard and soft references ignore it,
which is allowed for a reference-risk upper bound. The supplied result supports
the 6.9% paired improvement on the tested noise band, not a full-range floor or
a claim about generated samples.
