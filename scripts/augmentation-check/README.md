# Augmentation Consistency Check

This folder checks whether an LDM prediction is stable under geometry-preserving input transforms. For each sample, it predicts the original condition, predicts transformed versions of that same condition, transforms those augmented predictions back to the original orientation, and compares them against the original prediction.

The key idea is not ground-truth accuracy. It is equivariance consistency:

```text
f(T(x)) transformed back should approximately match f(x)
```

For scalar output channels, the script applies the inverse spatial transform before comparing. For UK condition channels, flips and rotations also update local coordinate channels `4/5` and wind-vector channels `6/7`, matching the dataset convention used by `ldm.data.cfd_data.CFDConditionalDataset`.

Outputs are written inside each selected experiment folder:

- `augmentation_consistency/per_sample_consistency.csv`
- `augmentation_consistency/summary_consistency.csv`
- `augmentation_consistency/examples/*.png`

Run a quick smoke test:

```bash
MAX_SAMPLES=20 NUM_EXAMPLES=4 sbatch scripts/augmentation-check/run_augmentation_consistency.slurm
```

Run the full training split:

```bash
sbatch scripts/augmentation-check/run_augmentation_consistency.slurm
```

Useful overrides:

```bash
FILTER=uk_grad_corr_low MAX_SAMPLES=100 STEPS=25 sbatch scripts/augmentation-check/run_augmentation_consistency.slurm
SPLIT=validation MAX_SAMPLES=0 sbatch scripts/augmentation-check/run_augmentation_consistency.slurm
```
