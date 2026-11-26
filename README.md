# StripContext

Research materials for **StripContext: Fusing Local and Directional Context for Surface Defect Segmentation**. StripContext adds a local depthwise filter and two direction-aware strip branches at the E3 decoder input of FastSDNet. The zero-initialized output projection preserves the input feature at initialization.

## Recreate Tables 1–9

From the repository root, run the following command with Python 3.12. This step uses only the standard library and the saved experimental results.

```bash
python tables.py --output generated
```

Open `generated/table1.csv` through `generated/table9.csv`. The additional file `paired_differences.csv` matches models by training seed before calculating differences. Ratio differences are in percentage points. Count differences are in images. The output uses the sample standard deviation across matched runs.

| Paper tables | Inputs and experimental unit |
| --- | --- |
| 1 | Fixed image-level split lists and input resolutions |
| 2–3 | Four methods compared on KSDD2, using the four shared training seeds |
| 4–5 | Branch ablation on KSDD2, five matched seeds |
| 6 | Placement ablation on KSDD2, three matched seeds |
| 7–8 | Separate training on KSDD1 fold 0, SD900 and MSD, three matched seeds per dataset |
| 9 | Parameter and convolution counts, four groups of measured batch-one forward passes on one RTX 5060 Laptop GPU |

The per-run values in `results/per_seed.csv` store proportions on a 0–1 scale. Empty cells denote unavailable measurements. Tables convert proportions to percentages. Some study groups reuse the same runs, including the earlier eight-seed baseline comparison. Analyze each group separately. The four-method comparison shares seeds 1261985776, 1927119363, 42 and 2026. Test-set observations informed architecture exploration, so the comparisons are descriptive.

## Train and evaluate

The reference environment used Python 3.12.7, PyTorch 2.9.1 with CUDA 12.8, NumPy 2.2.6 and Pillow 11.3.0. The following installation commands target Windows or Linux with an NVIDIA GPU and BF16 support. Use a fresh Python environment.

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Obtain the selected dataset from its source in `configs/data_sources.json`. Set the data directory to the parent of the paths listed in the corresponding `splits/*.csv`. For KSDD2 this directory contains `train/` and `test/`. The runner checks file hashes against the supplied split. The package contains file lists and checksums. Images and model weights are obtained or produced separately.

```bash
python experiment.py train --dataset KSDD2 --data ./data/KSDD2 --model FSD_StripContext --seed 42 --output runs/e3_42
python experiment.py train --dataset KSDD2 --data ./data/KSDD2 --model FastSDNet --seed 42 --output runs/base_42
python experiment.py test --dataset KSDD2 --data ./data/KSDD2 --model FSD_StripContext --seed 42 --output runs/e3_42
python experiment.py test --dataset KSDD2 --data ./data/KSDD2 --model FastSDNet --seed 42 --output runs/base_42
```

For a matched experiment, complete training for its model–seed pairs before invoking held-out evaluation. Each training command requires a new output directory. It trains one network for 200 epochs, writes validation history and saves the checkpoint with the highest pooled validation IoU, retaining the earlier epoch within a tolerance of 1e-8. Testing uses that checkpoint and writes per-image TP, FP and FN counts together with pooled metrics to `test.json`. The portable runner reads source images directly. Exact numerical agreement of a new full training run can depend on hardware and software.

The other dataset identifiers are `KSDD1`, `SD900` and `MSD`. For these datasets the paper compares FastSDNet with FSD_StripContext. Small-defect thresholds and normalization constants are recorded per dataset. On MSD the 213 defective test images determine foreground metrics. The 20 additional normal images determine false alarms. SD900 has no normal test images, so its normal-image metrics are undefined.

## Files

- `models.py` builds the baseline, the full E3 module, SPM and Coordinate Attention comparators, four branch ablations and three alternative placements. SPM and CA adapt the cited modules to E3.
- `numeric.py` implements the four-head cross-entropy/Dice loss and minibatch construction. The training recipe is in `configs/recipe.json`.
- `splits/` lists image paths, source mask paths, split assignments and SHA-256 hashes. An empty mask path on MSD denotes an unannotated defect-free image with an all-background target.
- `results/figure3_cases.csv` records the shared crop coordinates and full-image confusion counts for the three post hoc examples shown in Figure 3. Coordinates refer to resized inputs, with right and bottom bounds exclusive.
- `results/forward_timings.json` contains the original synchronized forward timings and grouped throughput measurements. Table 9 reports GPU-resident four-head forward passes, including the auxiliary heads. Memory is peak allocated MiB. The MAC count covers convolutions only.
- `third_party/` preserves the upstream implementations, revisions and individual license notices for Fast-SDNet, SPNet and Coordinate Attention. Dataset publishers provide access and usage terms at the linked sources.

