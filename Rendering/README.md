# Rendering

This subfolder contains code for using **neural antiderivatives in glossy illumination rendering**.

---

### Ground Truth Generation
To generate reference ground truths, run:
```bash
python convolution_2d.py
```

### Individual Methods

Each rendering method is implemented as:
```bash
convolution_deferred_<method_name>.py
```
where <method_name> can be one of the supported approaches (e.g., DoG, FD, AutoDiff, Reduction, etc.).

### Additional Resources
- `brdf/` — contains BRDF textures and dirac checkpoints used in rendering.
- `meshes/` — store additional 3D mesh models here if you wish to test new geometry.