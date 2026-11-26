"""Portable training, validation and held-out evaluation for StripContext."""
import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image

from models import VARIANTS, build_model
from numeric import batch_indices, model_loss

ROOT = Path(__file__).resolve().parent
DATASETS = json.loads((ROOT / "configs/datasets.json").read_text())
RECIPE = json.loads((ROOT / "configs/recipe.json").read_text())


class Images:
    def __init__(self, dataset, split, directory):
        self.name = dataset
        self.config = DATASETS[dataset]
        self.root = directory
        with (ROOT / "splits" / (dataset + ".csv")).open(encoding="utf-8", newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]

    def verify(self):
        for r in self.rows:
            for key in ("image", "mask"):
                if not r[key]:
                    continue
                path = self.root / r[key]
                with path.open("rb") as f:
                    digest = hashlib.file_digest(f, "sha256").hexdigest()
                if digest != r[key + "_sha256"]:
                    raise ValueError(f"File differs from split manifest: {path}")

    def sample(self, row):
        size = tuple(reversed(self.config["input_h_w"]))
        with Image.open(self.root / row["image"]) as image:
            rgb = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR))
        if row["mask"]:
            with Image.open(self.root / row["mask"]) as image:
                if self.name == "MSD":
                    pix = np.asarray(image.convert("RGB").resize(size, Image.Resampling.NEAREST))
                    mask = np.any(pix != 0, axis=-1)
                else:
                    mask = np.asarray(image.convert("L").resize(size, Image.Resampling.NEAREST)) > 0
        else:
            mask = np.zeros(size[::-1], dtype=bool)
        if bool(mask.any()) != bool(int(row["positive"])):
            raise ValueError(f"Foreground disagrees with the manifest for {row['image']}")
        return rgb.transpose(2, 0, 1), mask

    def batch(self, indices, device, generator=None):
        images, masks = zip(*(self.sample(self.rows[i]) for i in indices))
        x = torch.from_numpy(np.stack(images)).to(device, torch.float32).div_(255)
        y = torch.from_numpy(np.stack(masks)).to(device, torch.long)
        if generator is not None:
            for axis in (2, 3):
                chosen = (torch.rand(len(indices), generator=generator) < RECIPE["flip_prob"]).to(device)
                x = torch.where(chosen[:, None, None, None], x.flip(axis), x)
                y = torch.where(chosen[:, None, None], y.flip(axis - 1), y)
        mu = x.new_tensor(self.config["mean"])[None, :, None, None]
        sigma = x.new_tensor(self.config["std"])[None, :, None, None]
        return (x - mu) / sigma, y


def measures(counts, dataset):
    cfg = DATASETS[dataset]
    pos = [r for r in counts if r["tp"] + r["fn"]]
    normal = [r for r in counts if not r["tp"] + r["fn"]]
    limit = cfg["small_area"] - int(not cfg["small_upper_inclusive"])
    small = [r for r in pos if r["tp"] + r["fn"] <= limit]
    primary = pos if dataset == "MSD" else counts
    tp, fp, fn = [sum(r[k] for r in primary) for k in ("tp", "fp", "fn")]
    ratio = lambda a, b: a / b if b else None
    n_pixels = int(np.prod(cfg["input_h_w"]))
    return dict(IoU=ratio(tp, tp + fp + fn), Dice=ratio(2 * tp, 2 * tp + fp + fn),
                P=ratio(tp, tp + fp), R=ratio(tp, tp + fn),
                smallR=ratio(sum(r["tp"] for r in small), sum(r["tp"] + r["fn"] for r in small)),
                normalFPR=ratio(sum(r["fp"] for r in normal), len(normal) * n_pixels),
                misses=sum(r["tp"] == 0 for r in pos), small_misses=sum(r["tp"] == 0 for r in small),
                normal_FP_images=sum(r["fp"] > 0 for r in normal) if normal else None)


@torch.inference_mode()
def evaluate(model, data, device, test=False):
    model.eval()
    counts = []
    n = len(data.rows)
    batch = RECIPE["batch_size"]
    batches = [list(range(i, min(i + batch, n))) for i in range(0, n, batch)] if test else batch_indices(n, batch)
    for ids in batches:
        x, y = data.batch(ids, device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = model(x)[0].argmax(1).bool()
        gt = y.bool()
        count = torch.stack([(pred & gt).sum((1, 2)), (pred & ~gt).sum((1, 2)), (~pred & gt).sum((1, 2))], 1)
        for idx, (tp, fp, fn) in zip(ids, count.tolist()):
            counts.append(dict(image=data.rows[idx]["image"], tp=tp, fp=fp, fn=fn))
    return measures(counts, data.name), counts


def save_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def train(model, args, device, identity):
    train_set = Images(args.dataset, "train", args.data)
    valid_set = Images(args.dataset, "val", args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "config.json", dict(**identity, recipe=RECIPE, dataset_config=DATASETS[args.dataset]))
    train_set.verify()
    valid_set.verify()
    gen = torch.Generator().manual_seed(args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=RECIPE["lr"], betas=tuple(RECIPE["betas"]),
                           eps=RECIPE["eps"], weight_decay=RECIPE["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=RECIPE["epochs"], eta_min=RECIPE["eta_min"])
    best, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, RECIPE["epochs"] + 1):
        model.train()
        for ids in batch_indices(len(train_set.rows), RECIPE["batch_size"], gen, True):
            x, y = train_set.batch(ids, device, gen)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.bfloat16):
                heads = model(x)
            loss = model_loss("fastsdnet", heads, y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            opt.step()
        score = evaluate(model, valid_set, device)[0]["IoU"]
        sched.step()
        if score > best + RECIPE["checkpoint_tolerance"]:
            best, best_epoch = score, epoch
            tmp = args.output / "best.tmp"
            torch.save(dict(**identity, epoch=epoch, model_state=model.state_dict()), tmp)
            tmp.replace(args.output / "best.pt")
        history.append(dict(epoch=epoch, val_IoU=score, best_epoch=best_epoch))
        save_json(args.output / "history.json", history)
        print(f"Epoch {epoch}/200   validation IoU {score:.6f}   best epoch {best_epoch}", flush=True)
    save_json(args.output / "complete.json", dict(**identity, best_epoch=best_epoch, epochs=len(history)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["train", "test"])
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--data", type=Path, required=True, help="Root containing the files named in the dataset split CSV")
    p.add_argument("--model", choices=VARIANTS, default="FSD_StripContext")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True, help="A new directory for training; its matching directory for testing")
    args = p.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error("The published training/evaluation recipe requires a CUDA GPU with BF16 support")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    identity = dict(model=args.model, seed=args.seed, dataset=args.dataset)
    model = build_model(args.model, args.seed).cuda()
    if args.action == "train":
        train(model, args, "cuda", identity)
        return
    record = json.loads((args.output / "complete.json").read_text())
    if record["epochs"] != RECIPE["epochs"] or any(record[k] != v for k, v in identity.items()):
        raise ValueError("Completed run does not match the requested evaluation")
    ckpt = torch.load(args.output / "best.pt", weights_only=True, map_location="cpu")
    if ckpt["epoch"] != record["best_epoch"] or any(ckpt[k] != v for k,v in identity.items()):
        raise ValueError("Checkpoint and run metadata differ")
    model.load_state_dict(ckpt["model_state"], strict=True)
    data = Images(args.dataset, "test", args.data)
    data.verify()
    metrics, counts = evaluate(model, data, "cuda", test=True)
    save_json(args.output / "test.json", dict(**identity, metrics=metrics, per_image=counts))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

