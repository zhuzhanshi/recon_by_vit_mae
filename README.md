# Line-scan anomaly pipeline (unified)

## Overview
This project follows a reconstruction-based anomaly detection paradigm, where only normal samples are required for training.
This project implements an end-to-end pipeline for line-scan image anomaly detection using Masked Autoencoders (MAE) combined with Vision Transformers (ViT). It is designed primarily for industrial inspection scenarios, such as detecting defects on train undercarriages. The pipeline covers all stages from data preparation through MAE pretraining, decoder finetuning, inference with postprocessing, and visualization of results.

Typical pipeline stages include:  
- Preparing and organizing data  
- Pretraining the MAE model on unlabeled data  
- Finetuning the decoder on labeled or synthetic data  
- Running inference to detect anomalies  
- Visualizing and analyzing outputs  

---

## Environment Setup

Create the conda environment and activate it:

```
conda env create -f environment.yml
conda activate te_project
```

If conda is not available, install dependencies via pip fallback:

```
pip install -r requirements.txt
```

---

## Data Protocol (Spec v2)

The pipeline expects a `roots.txt` file listing directories, each containing JPG images. Each line in `roots.txt` specifies one root directory with image files. This design allows flexible grouping of datasets across multiple folders.

Filenames follow the pattern:

```
[vehicle_id]_10X[camera_id][series].jpg
```

- `vehicle_id`: integer from 1 to 8  
- `camera_id`: integer from 1 to 9  
- `series`: a zero-padded sequence number (e.g., 001..N), ordered left to right  

---

## Configuration (config-first philosophy)

All hyperparameters and settings are managed in the `configs/default.yaml` file. Command-line flags override configuration values when provided, enabling flexible experimentation without modifying config files directly.

---

## End-to-End Workflow

### Synthetic data generation

Generate synthetic datasets for training or testing:

```
python scripts/gen_synth_data.py --root synthetic_root --series 12 --runs 2
```

This creates synthetic image series under the specified root directory.

### MAE pretraining

Pretrain the MAE model using combined train and validation roots. This step learns general image representations without labels:

```
python scripts/train_mae.py --roots-txt synthetic_root/roots.txt --config configs/default.yaml --output outputs/pretrain
```

### Decoder finetuning

Finetune the decoder head on unlabeled normal data (or optionally synthetic normal data). This stage adapts the reconstruction behavior of the pretrained MAE to the target domain, enabling anomaly detection via reconstruction residuals without requiring explicit anomaly labels.

```
python scripts/finetune_decoder.py --roots-txt synthetic_root/roots.txt --pretrained outputs/pretrain/mae_pretrained.pth --config configs/default.yaml --output outputs/finetune
```

### Inference & postprocess

Run inference on new data using the finetuned model. Postprocessing settings from the config are applied automatically:

```
python -m src.runner infer --roots-txt synthetic_root/roots.txt --mode mae --ckpt outputs/finetune/mae_finetuned.pth --config configs/default.yaml --output outputs/infer_mae
```

Inference outputs include:  
- `H_global.npy` (heatmap data)  
- `bboxes.json` (detected bounding boxes)  
- `overlay_heatmap.jpg` (visual heatmap overlay)  
- `overlay_bboxes.jpg` (visual bounding box overlay)  

---

## Experiment Management (exp_name timestamp suffix)

Set `train.exp_name_time_suffix` in the config to automatically append a timestamp suffix to experiment names. The resolved experiment name is saved to:

```
outputs/<resolved_exp_name>/resolved_exp_name.txt
```

This helps organize outputs from multiple runs systematically.

---

## Monitoring & Debugging

### TensorBoard

Launch TensorBoard to monitor training and validation metrics:

```
tensorboard --logdir runs
```

Quick smoke test for TensorBoard setup:

```
python -c "from torch.utils.tensorboard import SummaryWriter; w=SummaryWriter('runs/_smoke'); w.add_scalar('x',1,0); w.close(); print('tb ok')"
```

### Validation residual statistics

During validation, residual quantiles such as P95, P99, and P99.5 are logged based on the residual used for inference (e.g., `|gray - recon|`). These statistics are saved to TensorBoard and to JSON files like:

```
outputs/<exp_name>/val_stats/epoch_XXXX_residual_quantiles.json
```

Use these quantile values to select an appropriate `postprocess.quantile` threshold, improving anomaly detection sensitivity instead of guessing.

### Model inspection

Inspect model architectures and parameter counts using:

```
python scripts/inspect_model.py --config configs/default.yaml --preset tiny
python scripts/inspect_model.py --config configs/default.yaml --preset vitb
```

Comparing the tiny and ViT-Base (vitb) presets helps understand trade-offs between model size, complexity, and performance.

(Optional: install `torchinfo` for detailed layer-wise summaries.)

---

## Pretrained MAE Initialization

Configure MAE initialization from a pretrained checkpoint in `configs/default.yaml`:

```yaml
mae:
  init_from_pretrained: true
  pretrained_ckpt: /path/to/mae_vitb16.pth
```

An initialization report is saved to:

```
outputs/<exp_name>/init_report.json
```

Download the official MAE pretrained checkpoint:

```
curl -L -o mae_pretrain_vit_base.pth https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
```

---

## CLI Utilities

Build index and compute group statistics:

```
python -m src.runner build-index --roots-txt /path/to/roots.txt --config configs/default.yaml
```

Run demo inference on a single group:

```
python -m src.runner demo-infer --roots-txt /path/to/roots.txt --dir-path /path/to/dir --vehicle 1 --camera 1 --config configs/default.yaml
```

Run inference on all groups listed in `roots.txt`:

```
python -m src.runner infer --roots-txt /path/to/roots.txt --config configs/default.yaml
```

---

## Notes / Design Assumptions

- Training crops are 512x512 pixels, aligned to patch size `P` (default 16).  
- Inference uses sliding windows with stride 256 and weighted fusion for smooth predictions.  
- Stitching is done in series order with no overlap; total width `W_total = s_max * W_block`.
