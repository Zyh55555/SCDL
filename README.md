# Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation

This repository contains the official implementation of **Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation (SCDL)**.

## Acknowledgement

This project is built upon and extends parts of the codebase from [GALoss](https://github.com/cicailalala/GALoss), developed for the ECCV 2024 paper **Gradient-Aware for Class-Imbalanced Semi-supervised Medical Image Segmentation**.  
We sincerely thank the original authors for making their code publicly available.

In this repository, the original framework has been further modified and extended to support our SCDL method, including the corresponding training and inference pipelines.

## Overview

This repository provides training and inference code for semi-supervised medical image segmentation on datasets such as **Synapse** and **AMOS**.

Supported evaluation entry points in this folder include:

- `inference_Synapse_MagicNet.py`
- `inference_Synapse_CPS.py`
- `inference_AMOS_MagicNet.py`
- `inference_AMOS_CPS.py`

All inference scripts assume commands are executed from the `SCDL/` directory and require at least one visible CUDA GPU.

## Environment

Recommended base packages:

- Python 3.7.11
- torch 1.10.0
- torchvision 0.11.0
- opencv-python 4.1.1.26
- numpy 1.20.3
- h5py 3.7.0
- scipy 1.7.1

The inference utilities in this folder also require:

- `timm`
- `einops`
- `nibabel`
- `medpy`
- `scikit-image`
- `tqdm`
- `surface-distance`

Example Linux setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision opencv-python numpy h5py scipy timm einops nibabel medpy scikit-image tqdm surface-distance matplotlib
```

## Data Preparation

### Synapse

- Download the processed Synapse data used by MagicNet.
- Place it under `./data/Synapse/`.
- The inference scripts evaluate the fixed test list `['0004', '0007', '0010', '0033', '0035', '0036']`.

### AMOS

- Download the processed AMOS data and place it under `./data/AMOS/`.
- This repository already includes the split files under `./data/amos_splits/`.
- The AMOS inference scripts read the test IDs from `./data/amos_splits/test.txt`.

A typical local layout is:

```text
SCDL/
|-- data/
|   |-- Synapse/
|   |-- AMOS/
|   `-- amos_splits/
|-- model/
|-- inference_Synapse_MagicNet.py
|-- inference_Synapse_CPS.py
|-- inference_AMOS_MagicNet.py
`-- inference_AMOS_CPS.py
```

## Important Notes Before Running Inference

- Run all commands from the `SCDL/` directory.
- If you do not pass checkpoint paths, the scripts will automatically search the default experiment directory under `./model/`.
- `inference_Synapse_MagicNet.py` supports two checkpoint layouts.
- One layout is a combined checkpoint that already contains both `model` and `scdl`.
- The other layout is a model-only checkpoint together with a separate `--scdl_path`.
- For CPS inference, either pass both `--model_path_A` and `--model_path_B`, or pass neither and let the script auto-discover a matched pair.
- `inference_AMOS_MagicNet.py` has an absolute default `root_path` in the source code. On a normal Linux machine you should always pass your own `--root_path ./data/AMOS/`.

## Inference Commands

### 1. Synapse MagicNet

Use this when evaluating the MagicNet-style model with SCDL prior.

Explicit checkpoint path:

```bash
cd /path/to/SCDL
CUDA_VISIBLE_DEVICES=0 python inference_Synapse_MagicNet.py \
  --root_path ./data/Synapse \
  --save_path ./model \
  --labelnum 4 \
  --seed 1337 \
  --model_path ./model/Synapse_MagicNet_GA_4labeled_seed_1337/iter_xxx_best.pth
```

Automatic checkpoint discovery:

```bash
CUDA_VISIBLE_DEVICES=0 python inference_Synapse_MagicNet.py \
  --root_path ./data/Synapse \
  --save_path ./model \
  --labelnum 4 \
  --seed 1337
```

This searches the latest `*_best.pth` under:

```text
./model/Synapse_MagicNet_GA_<labelnum>labeled_seed_<seed>/
```

### 2. Synapse CPS

Use this when evaluating the two-network CPS model.

Explicit checkpoint paths:

```bash
cd /path/to/SCDL
CUDA_VISIBLE_DEVICES=0 python inference_Synapse_CPS.py \
  --root_path ./data/Synapse \
  --save_path ./model \
  --labelnum 4 \
  --seed 1337 \
  --model_path_A ./model/Synapse_CPS_GA_4labeled_seed_1337/iter_xxx_best_A.pth \
  --model_path_B ./model/Synapse_CPS_GA_4labeled_seed_1337/iter_xxx_best_B.pth
```

### 3. AMOS MagicNet

Use this when evaluating the AMOS MagicNet variant.

Explicit checkpoint path:

```bash
cd /path/to/SCDL
CUDA_VISIBLE_DEVICES=0 python inference_AMOS_MagicNet.py \
  --root_path ./data/AMOS \
  --save_path ./model \
  --labelnum 10 \
  --seed 1337 \
  --model_path ./model/AMOS_MagicNet_GA_10labeled/iter_xxx_best.pth
```

### 4. AMOS CPS

Use this when evaluating the two-network CPS model on AMOS.

Explicit checkpoint paths:

```bash
cd /path/to/SCDL
CUDA_VISIBLE_DEVICES=0 python inference_AMOS_CPS.py \
  --root_path ./data/AMOS \
  --save_path ./model \
  --labelnum 10 \
  --seed 1337 \
  --model_path_A ./model/AMOS_CPS_10labeled/iter_xxx_best_A.pth \
  --model_path_B ./model/AMOS_CPS_10labeled/iter_xxx_best_B.pth
```

## Output Files

After inference, the scripts write results into the corresponding experiment directory under `./model/`.

Common output files include:

- `log_total_metric.txt`: formatted validation metrics
- `metric_final_<dataset>_<exp>.npy`: raw metric array

For example, Synapse MagicNet with `labelnum=4` and `seed=1337` writes into:

```text
./model/Synapse_MagicNet_GA_4labeled_seed_1337/
```

## References

This repository is related to or built upon the following projects:

- [GALoss](https://github.com/cicailalala/GALoss)
- [MagicNet](https://github.com/DeepMed-Lab-ECNU/MagicNet)
- [DHC](https://github.com/xmed-lab/DHC)

## Citation

If you find this repository useful, please consider citing our paper below.

### SCDL

```bibtex
@misc{su2026semanticclassdistributionlearning,
      title={Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation}, 
      author={Yingxue Su and Yiheng Zhong and Keying Zhu and Zimu Zhang and Zhuoru Zhang and Yifang Wang and Yuxin Zhang and Jingxin Liu},
      year={2026},
      eprint={2603.05202},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.05202}, 
}