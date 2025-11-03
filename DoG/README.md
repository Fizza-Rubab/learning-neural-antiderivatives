# Training and Evaluation of Derivative of Gaussians (DoG)

This directory contains scripts to **train and evaluate neural antiderivatives using smooth estimators** (Derivative of Gaussian – DoG).  
Experiments can be run across different dimensions (1D, 2D, 3D), signal types (analytic vs. real data), and blur configurations (blur / no-blur).

---

## Scripts Overview
For all scripts below, first `cd` into AutoDiff folder. The training scripts are in experiments subfolder.

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

## Evaluation

The repository provides two main evaluation types:

### Reconstruction (`derivative_*.py`)

Use these scripts to evaluate derivative-based reconstructions:

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

Used to evaluate a particular model trained with DoG for a convolution task. Depending on the path it infers whether to blur or not and what is the scale and order

```bash
python eval.py --model_path="../models/DoG-Blur/1d/subject_1_motion1d_order_2_gaussian_0.03_samples_100000.npy_1d_scale_0.01_order_0_gpu.pth"
python eval.py --model_path="../models/DoG-Noblur/2d/0105_2d_scale_0.01_order_0.pth"
python eval.py --model_path="../models/DoG-Blur/3d/ShapeNet_04468005_1fff38d54059eb1465547fd38c0cec46_3d_order_2_0.6_samples_30000_3d_scale_0.2_order_0_gpu.pth"
```
