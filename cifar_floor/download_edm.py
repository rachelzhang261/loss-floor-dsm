"""Download NVIDIA's EDM ImageNet-64 checkpoint and validation synsets."""

import argparse
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


CHECKPOINT = "edm-imagenet-64x64-cond-adm.pkl"
CHECKPOINT_URL = f"https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/{CHECKPOINT}"
SYNSETS = "imagenet_2012_validation_synset_labels.txt"
SYNSETS_URL = f"https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/{SYNSETS}"


def download(url, destination):
    if destination.exists():
        print(f"Already present: {destination}")
        return
    with NamedTemporaryFile(dir=destination.parent, delete=False) as scratch:
        temporary_path = Path(scratch.name)
    digest = hashlib.sha256()
    try:
        with urlopen(url, timeout=60) as response, temporary_path.open("wb") as out:
            expected = response.headers.get("Content-Length")
            if response.headers.get_content_type() in {"text/html", "application/json"}:
                raise ValueError(f"Unexpected response type from {url}")
            total = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                out.write(block)
                digest.update(block)
                total += len(block)
                if total % (64 * 1024**2) == 0:
                    print(f"Downloaded {total / 1024**2:.0f} MiB", flush=True)
        if not total or (expected is not None and total != int(expected)):
            raise ValueError(f"Incomplete download from {url}: received {total} bytes")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved {destination} ({total / 1024**2:.1f} MiB), SHA-256 {digest.hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    download(CHECKPOINT_URL, args.out / CHECKPOINT)
    download(SYNSETS_URL, args.out / SYNSETS)


if __name__ == "__main__":
    main()
