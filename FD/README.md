# Training and Evaluation of Finite Differences (FD)

This directory contains scripts to **train and evaluate neural antiderivatives using finite differences**.  
Experiments can be run across different dimensions (1D, 2D, 3D), signal types (analytic vs. real data), and integration orders.

---

## Key Notes
- Integration order convention:  
  - `--order 0` in FD corresponds to first-order integration in AD/Integral.  
  - So FD order *k* = AD/Integral order *(k+1)*.  
- Use `--blur 1` to enable blur-compensated supervision.  
- For **both blurred and unblurred training**, the signal is passed using `--monte_carlo`.  
  - If unblurred: pass the raw data (image, motion, geometry, etc.).  
  - If blurred: pass the precomputed blurred `.npy` ground truth from `fd_blurred_gts`.  
- Evaluation is split into **blur** and **non-blur** versions (`*_blur.py` vs. standard).  

---

## Scripts Overview

### Training
- `train_1d.py` — 1D analytic functions (Ackley, Gaussians, hyperrectangles).  
- `train_1d_motion.py` — 1D real motion-capture sequences.  
- `train_2d.py` — 2D analytic signals or natural images.  
- `train_2d_envmap.py` — 2D environment maps for rendering tasks.  
- `train_3d.py` — 3D analytic signals or signed distance functions (SDFs).  

### Evaluation
- `derivative_*_analytic.py`, `derivative_*_all.py` — Non-blur evaluation.  
- `derivative_*_analytic_blur.py`, `derivative_*_all_blur.py` — Blur-compensated evaluation.  
- `eval.py`, `eval_blur.py` — Convolution evaluation (non-blur / blur).  

---

## Running Experiments

### 1D Analytic (Ackley, order 0 = first order FD)
```bash
python train_1d.py \
  --summary "logs" \
  --experiment_name "fd_1d_ackley_order0" \
  --batch 1024 \
  --num-steps 2000 \
  --num_channels 256 \
  --num_layers 4 \
  --schedule_gamma 0.5 \
  --schedule_step 10000 \
  --pe 4 \
  --learn_rate 1e-3 \
  --workers 12 \
  --norm_exp 0 \
  --monte_carlo "../../data/analytic/ackley.npy" \
  --order 0
```

### 1D Motion Data (blurred, order 1)
```bash
python train_1d_motion.py \
  --summary "logs" \
  --experiment_name "fd_1d_motion_subject0_blur_order0" \
  --batch 1024 \
  --num-steps 1000 \
  --num_channels 256 \
  --num_layers 4 \
  --schedule_gamma 0.5 \
  --schedule_step 10000 \
  --pe 4 \
  --learn_rate 1e-3 \
  --workers 12 \
  --norm_exp 0 \
  --monte_carlo "../../data/fd_blurred_gts/motion_mc_order=1/subject_0_motion1d_order_1_minimal_0.04_samples_100000.npy" \
  --order 0 \
  --blur 1
```

### 2D Image (DIV2K sample, unblurred, order 0)
```bash
python train_2d.py \
  --summary "logs" \
  --experiment_name "fd_2d_image_0008_order0" \
  --batch 512 \
  --num-steps 1000 \
  --num_channels 256 \
  --num_layers 4 \
  --schedule_gamma 0.5 \
  --schedule_step 100000 \
  --pe 4 \
  --learn_rate 1e-3 \
  --workers 12 \
  --norm_exp 0 \
  --monte_carlo "../../data/images/0008.png" \
  --order 0
```

### 2D Environment Map (blurred, order 0)
```bash
python train_2d_envmap.py \
  --summary "logs" \
  --experiment_name "fd_2d_envmap_studio_blur_order0" \
  --batch 512 \
  --num-steps 2000 \
  --num_channels 256 \
  --num_layers 4 \
  --schedule_gamma 0.5 \
  --schedule_step 100000 \
  --pe 4 \
  --learn_rate 1e-3 \
  --workers 12 \
  --norm_exp 0 \
  --monte_carlo "../../data/fd_blurred_gts/envmap_mc_order=0/dikhololo_night_1k_2d_order_0_minimal_0.04_samples_200000.npy" \
  --order 0 \
  --blur 1
```

### 3D SDF (Stanford Bunny, unblurred, order 0)
```bash
python train_3d.py \
  --summary "logs" \
  --experiment_name "fd_3d_sdf_bunny_order0" \
  --batch 256 \
  --num-steps 2000 \
  --num_channels 256 \
  --num_layers 4 \
  --schedule_gamma 0.5 \
  --schedule_step 100000 \
  --pe 4 \
  --learn_rate 1e-3 \
  --workers 12 \
  --norm_exp 0 \
  --monte_carlo "../../data/geometry/Stanford_bunny.ply" \
  --order 0
```

---

## Evaluation

The repository provides two main evaluation types:

### Reconstruction (`derivative_*.py`)

Use these scripts to evaluate derivative-based reconstructions. Append `_blur.py` to the end of these files to evaluate the blurred version.

- `derivative_1d_motion_all.py` → motion signals  
- `derivative_2d_all.py` → images  
- `derivative_3d_all.py` → 3D geometry  
- `derivative_*d_analytic.py` → analytic signals

Run directly, e.g.:

```bash
python derivative_1d_motion_all.py
python derivative_2d_all.py
python derivative_3d_all.py
```

### Convolution (`eval.py`)

Used to evaluate a particular model trained with FD for a convolution task. Run eval.py or eval_blur.py depending on the method.

```bash
python eval.py --model_path="../models/FD-Blur/1d/subject_0_order=2.pth"
python eval.py --model_path="../models/FD-Noblur/2d/0085_order=1_25.pth"
python eval.py --model_path="../models/FD-Noblur/3d/Stanford_bunny_order=0_scale=25.pth"
```

```bash
python eval_blur.py --model_path="../models/FD-Blur/1d/subject_10_motion1d_order_0_minimal_0.04_samples_100000.npy_order=0.pth"
python eval_blur.py --model_path="../models/FD-Blur/1d/subject_10_motion1d_order_0_minimal_0.04_samples_100000.npy_order=0.pth"
python eval_blur.py --model_path="../models/FD-Blur/3d/Thingi10K_815483_3d_order_0_0.04_samples_10000_order=0.pth"
```
