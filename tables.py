"""Regenerate the paper's nine tables and paired descriptive statistics, offline."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parent
MAIN = ["FastSDNet", "FSD_E3SPM", "FSD_E3CoordAtt", "FSD_StripContext"]
ABLATION = ["FastSDNet", "FSD_StripLocal", "FSD_StripAxes", "FSD_StripLocalRow",
            "FSD_StripLocalColumn", "FSD_StripContext"]
PLACES = ["FastSDNet", "FSD_StripAtE1", "FSD_StripAtE2", "FSD_StripContext", "FSD_StripAtE2E3"]
OVERLAP = ["IoU", "Dice", "P", "R", "smallR"]
ERRORS = ["normalFPR", "misses", "small_misses", "normal_FP_images"]


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(values, digits):
    return f"{mean(values):.{digits}f} ± {stdev(values):.{digits}f}"


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = math.floor(index)
    return values[lo] + (values[math.ceil(index)] - values[lo]) * (index - lo)


def main(output):
    output.mkdir(exist_ok=True, parents=True)

    def write(name, header, rows):
        with (output / (name + ".csv")).open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    splits = []
    config = json.loads((ROOT / "configs/datasets.json").read_text())
    for dataset, spec in config.items():
        rows = load_csv(ROOT / "splits" / (dataset + ".csv"))
        counts = Counter(r["split"] for r in rows)
        test = counts["test"]
        if dataset == "MSD":
            pos = sum(int(r["positive"]) for r in rows if r["split"] == "test")
            test = f"{pos} + {test-pos} normal"
        splits.append([dataset, counts["train"], counts["val"], test, "x".join(map(str, spec["input_h_w"]))])
    write("table1", ["dataset", "train", "val", "test", "HxW"], splits)

    groups = defaultdict(dict)
    for row in load_csv(ROOT / "results/per_seed.csv"):
        group = groups[(row["study"], row["model"])]
        if row["seed"] in group:
            raise ValueError(f"Duplicate seed within a group: {row}")
        group[row["seed"]] = {k: float(row[k]) if row[k] else None for k in OVERLAP + ERRORS}

    def to_units(key, val):
        return val * 100 if key in OVERLAP + ["normalFPR"] else val

    def summary(study, model, keys):
        cells = []
        for key in keys:
            values = [row[key] for row in groups[study, model].values()]
            if all(v is None for v in values):
                cells.append("—")
            else:
                cells.append(summarize([to_units(key, v) for v in values], 6 if key == "normalFPR" else 2))
        return cells

    for table, study, models, keys in [(2, "main4", MAIN, OVERLAP), (3, "main4", MAIN, ERRORS),
                                        (4, "internal5", ABLATION, OVERLAP), (5, "internal5", ABLATION, ERRORS)]:
        write(f"table{table}", ["model", *keys], [[m, *summary(study, m, keys)] for m in models])
    costs = json.loads((ROOT / "results/conv_cost.json").read_text())
    metrics = ["IoU", "Dice", "smallR"]
    write("table6", ["model", *metrics, "parameters", "conv_MACs_M"],
          [[m, *summary("positions3", m, metrics), costs[m][0], f"{costs[m][1]/1e6:.6f}"] for m in PLACES])
    for table, keys in [(7, OVERLAP), (8, ERRORS)]:
        write(f"table{table}", ["dataset", "model", *keys],
              [[ds, m, *summary(ds, m, keys)] for ds in ("KSDD1", "SD900", "MSD")
               for m in ("FastSDNet", "FSD_StripContext")])
    timing = json.loads((ROOT / "results/forward_timings.json").read_text())
    rows = []
    for m in MAIN:
        blocks = [r for r in timing if r["model"] == m and r["batch"] == 1]
        times = [t for b in blocks for t in b["wall_ms"]]
        assert len(blocks) == 4 and len(times) == 800
        rows.append([m, costs[m][0], f"{costs[m][1]/1e6:.6f}", f"{median(times):.3f}",
                     f"{percentile(times,.95):.3f}", f"{max(b['single_peak']['allocated_MiB'] for b in blocks):.3f}"])
    write("table9", ["model","parameters","conv_MACs_M","median_ms","P95_ms","peak_allocated_MiB"], rows)

    diffs = []
    for (study, model), other in groups.items():
        if model == "FSD_StripContext":
            continue
        full = groups[study, "FSD_StripContext"]
        assert full.keys() == other.keys()
        for metric in OVERLAP + ERRORS:
            pairs = [(full[s][metric], other[s][metric]) for s in full]
            if all(a is None and b is None for a, b in pairs):
                continue
            delta = [to_units(metric, a-b) for a,b in pairs]
            diffs.append([study, model, metric, len(delta), mean(delta), stdev(delta)])
    write("paired_differences", ["study","comparator","metric","n","full_minus_comparator_mean","sample_SD"], diffs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("generated"))
    main(parser.parse_args().output)

