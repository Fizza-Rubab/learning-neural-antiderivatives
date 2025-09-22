# Training and Evaluation of Automatic Differentiation

This directory contains scripts to **train and evaluate neural antiderivatives using Automatic Differentiation (AD)**.  
Experiments can be run across different dimensions (1D, 2D, 3D), signal types (analytic vs. real data), and integration orders.

---

## 📂 Scripts Overview

### Training
- `train_1d.py` — 1D analytic functions (Ackley, Gaussians, hyperrectangles).  
- `train_1d_motion.py` — 1D real motion-capture sequences.  
- `train_2d.py` — 2D analytic signals or natural images.  
- `train_2d_envmap.py` — 2D environment maps for rendering tasks.  
- `train_3d.py` — 3D analytic signals or signed distance functions (SDFs).  

### Evaluation
- `derivative_1d_analytic.py`, `derivative_1d_motion_all.py` — Evaluate 1D analytic / motion data.  
- `derivative_2d_analytic.py`, `derivative_2d_all.py` — Evaluate 2D analytic / images.  
- `derivative_3d_analytic.py`, `derivative_3d_all.py` — Evaluate 3D analytic / SDFs.  
- `eval.py` — Evaluate convolution task. (Ground truth need to be generated).  

---

## 🚀 Running Experiments

Below are examples for training and evaluating with AD supervision.  
Use descriptive experiment names:  
`{dimension}_{datatype}_{signal/order}` (e.g., `2d_image_0008_order1`).  

---

### 🔹 1D
#### 1D Analytic (Ackley function, order 1)
```bash
python train_1d.py \
  --summary "logs" \
  --experiment_name "1d_ackley_order1" \
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
  --analytic "ackley" \
  --order 1
```

Change --`order` to 2 or 3 for higher-order integration.

#### Motion Data (Subject 0, order 1)

```bash
python train_1d_motion.py \
  --summary "logs" \
  --experiment_name "1d_motion_subject0_order1" \
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
  --pose "../../data/motion/subject_0.txt" \
  --order 1
```

### 🔹 2D

#### Analytic Function (Hyperrectangle, order 2)

```bash
python train_2d.py \
  --summary "logs" \
  --experiment_name "2d_hyperrect_order2" \
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
  --analytic "hr" \
  --order 2
```

#### Real Image (DIV2K sample, order 1)
```bash
python train_2d.py \
  --summary "logs" \
  --experiment_name "2d_image_0008_order1" \
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
  --image "../../data/images/0008.png" \
  --order 1
```

#### Environment Map

```bash
python train_2d_envmap.py \
  --summary "logs" \
  --experiment_name "2d_envmap_studio_order1" \
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
  --image "../../data/envmap/dikhololo_night_1k.exr" \
  --order 1
```

### 🔹 3D

#### Analytic Function (Gaussian, order 2)

```bash
python train_3d.py \
  --summary "logs" \
  --experiment_name "3d_gaussian_order2" \
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
  --analytic "gm" \
  --order 2
```

#### Real Geometry
```bash
python train_3d.py \
  --summary "logs" \
  --experiment_name "3d_sdf_chair_order1" \
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
  --object "../../data/geometry/Stanford_bunny.ply" \
  --order 1
```