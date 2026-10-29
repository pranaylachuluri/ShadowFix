"""Count ISTD triplets and save preview strips (shadow | mask | shadow-free).

    python inspect_dataset.py --data-root data --split train --num-samples 5 --output-dir dataset_preview
"""
import argparse
import random
from pathlib import Path

from PIL import Image

from shadowfix.data import list_triplets, load_mask, load_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--output-dir", default="dataset_preview")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    records = list_triplets(args.data_root, args.split)
    with_masks = sum(r[1] is not None for r in records)
    print(f"Found {len(records)} triplets ({with_masks} with masks) in the {args.split} split.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for a, b, c in random.Random(args.seed).sample(records, min(args.num_samples, len(records))):
        shadow, target = load_rgb(a, 256), load_rgb(c, 256)
        mask = load_mask(b, 256, like=shadow).convert("RGB")
        strip = Image.new("RGB", (256 * 3 + 16, 256), "white")
        for i, img in enumerate((shadow, mask, target)):
            strip.paste(img, (i * (256 + 8), 0))
        strip.save(out / f"{a.stem}.png")
    print(f"Saved previews to: {out}")


if __name__ == "__main__":
    main()
