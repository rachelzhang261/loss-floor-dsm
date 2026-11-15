"""Stream a labeled subset of ILSVRC ImageNet validation images."""

import argparse
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory


DATASET = "ILSVRC/imagenet-1k"
SPLIT = "validation"
SHUFFLE_BUFFER = 1024


def save_subset(rows, output, count, revision, seed):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        subset = Path(scratch) / "subset"
        images = subset / "images"
        images.mkdir(parents=True)
        records = []
        for index, row in enumerate(rows.take(count), start=1):
            image = row["image"]
            data = image["bytes"]
            if not data:
                raise ValueError(f"Image {index} did not contain embedded bytes")

            source = PurePosixPath((image["path"] or "").replace("\\", "/")).name
            suffix = PurePosixPath(source).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}:
                suffix = ".img"
            name = f"{index:05d}{suffix}"
            (images / name).write_bytes(data)
            records.append({
                "file": f"images/{name}",
                "source_name": source,
                "label": int(row["label"]),
            })
            if index % 50 == 0 or index == count:
                print(f"Saved {index}/{count} images", flush=True)

        if len(records) != count:
            raise ValueError(f"Requested {count} images but received {len(records)}")

        manifest = {
            "dataset": DATASET,
            "split": SPLIT,
            "revision": revision,
            "seed": seed,
            "shuffle_buffer": SHUFFLE_BUFFER if seed is not None else None,
            "images": records,
        }
        (subset / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        subset.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--count", type=int, default=16, help="number of images to save")
    parser.add_argument("--seed", type=int, help="shuffle with this seed before taking the subset")
    parser.add_argument("--revision", default="main", help="Hugging Face branch, tag, or commit")
    args = parser.parse_args()
    if not 1 <= args.count <= 50_000:
        parser.error("--count must be between 1 and 50000")
    output = args.out.resolve()
    if output.exists():
        parser.error(f"output directory already exists: {output}")

    from datasets import Image, load_dataset
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, LocalTokenNotFoundError

    try:
        revision = HfApi().dataset_info(DATASET, revision=args.revision, token=True).sha
        rows = load_dataset(
            DATASET, split=SPLIT, revision=revision, streaming=True, token=True
        ).cast_column("image", Image(decode=False))
        if args.seed is not None:
            rows = rows.shuffle(seed=args.seed, buffer_size=SHUFFLE_BUFFER)
        save_subset(rows, output, args.count, revision, args.seed)
    except (GatedRepoError, LocalTokenNotFoundError) as exc:
        parser.exit(1, f"{exc}\nAccept access at https://huggingface.co/datasets/{DATASET} "
                       "and run 'hf auth login' with the same account.\n")

    print(f"Wrote {args.count} images and manifest.json to {output}")


if __name__ == "__main__":
    main()
