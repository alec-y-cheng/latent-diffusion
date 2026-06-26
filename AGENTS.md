# AGENTS.md

## Project Context

This repository contains latent diffusion and surrogate modeling experiments for urban wind comfort prediction.

### References

Professor/instructor example code can be found in the `references/` folder. Treat these scripts as the primary reference for data loading, training, inference, visualization, and import logic. When unsure, follow patterns from `references/` before creating new implementations.

---

## Data Formats

Datasets are typically stored as `.h5` (preferred) or `.npz` (smaller experiments).

Common datasets:

* `one_target_dataset.h5`
* `one_target_did_dataset.h5`

`one_target_did_dataset.h5` extends the standard dataset with additional Directionally Integrated Distance (DID) channels corresponding to the 8 wind directions.

---

## Inputs

Standard inputs contain 8 channels:

  Input X, shape (8, H, W)

  0: X coordinate
  1: Y coordinate
  2: Z_relative
  3: SDF
  4: Bldg_height
  5: U/Uref
  6: dir_sin
  7: dir_cos




Inputs are normalized before training. Most models operate on tensors of shape:

`X = [C, H, W]`

---

## Outputs

Historically, models predicted a single wake-deficit field:

`Delta_U = (Mag_U - U_ref) / U_ref`

Actual wind magnitude can be reconstructed as:

`Mag_U = (Prediction * U_ref) + U_ref`

Current experiments may predict multiple outputs simultaneously:

  Target Y, shape (4, H, W)

  0: floor speed / mag_U
  1: floor turbulence / k
  2: roof speed / mag_U_roof
  3: roof turbulence / k_roof

Do not assume outputs are single-channel.

---

## Methodology

Most experiments use SDF and/or DID geometry representations as a proxy for CFD flow fields. Models are often adapted from image-generation architectures (UNets, ViTs, diffusion models, transformers, etc.) and augmented with physics-informed losses and evaluation metrics.

Common evaluation metrics:

* RMSE
* MAE
* MAPE
* SSIM
* R²
* Gradient Correlation (GradCorr)
* Inference Time

When evaluating models, prioritize physically meaningful metrics (especially GradCorr and flow structure preservation) rather than pixel-wise image quality alone.

Don't wait for long commands to finish
