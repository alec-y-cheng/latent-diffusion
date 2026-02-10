"""Quick test script to verify metrics display layout."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_H, IMG_W = 504, 504

# Simulate data
np.random.seed(42)
y_true = np.random.rand(IMG_H, IMG_W) * 10
y_pred = y_true + np.random.randn(IMG_H, IMG_W) * 0.5

diff = y_pred - y_true

# --- Create Circular Domain Mask (exclude corners) ---
H, W = y_true.shape
center_y, center_x = H // 2, W // 2
radius = min(H, W) // 2 - 5
Y_coords, X_coords = np.ogrid[:H, :W]
dist_from_center = np.sqrt((X_coords - center_x)**2 + (Y_coords - center_y)**2)
domain_mask = dist_from_center < radius

# --- Compute Metrics (within circular domain only) ---
diff_masked = diff[domain_mask]
abs_diff_masked = np.abs(diff_masked)

mae = np.mean(abs_diff_masked)
rmse = np.sqrt(np.mean(diff_masked**2))
max_err = np.max(abs_diff_masked)

# MAPE
gt_masked = y_true[domain_mask]
gt_abs = np.abs(gt_masked)
valid_for_mape = gt_abs > 0.1
mape = np.mean(abs_diff_masked[valid_for_mape] / gt_abs[valid_for_mape]) * 100.0

# SSIM
try:
    from skimage.metrics import structural_similarity as ssim_func
    data_range = max(np.max(y_true), np.max(y_pred)) - min(np.min(y_true), np.min(y_pred))
    ssim_val = ssim_func(y_true, y_pred, data_range=data_range)
except ImportError:
    ssim_val = -1.0

# Gradient Correlation
def compute_gradient_correlation(pred, true, dmask):
    pred_dx = np.diff(pred, axis=1, prepend=pred[:, :1])
    pred_dy = np.diff(pred, axis=0, prepend=pred[:1, :])
    true_dx = np.diff(true, axis=1, prepend=true[:, :1])
    true_dy = np.diff(true, axis=0, prepend=true[:1, :])
    
    pred_grad = np.concatenate([pred_dx.flatten(), pred_dy.flatten()])
    true_grad = np.concatenate([true_dx.flatten(), true_dy.flatten()])
    mask_flat = np.concatenate([dmask.flatten(), dmask.flatten()])
    pred_grad = pred_grad[mask_flat]
    true_grad = true_grad[mask_flat]
    
    return np.corrcoef(pred_grad, true_grad)[0, 1]

grad_corr = compute_gradient_correlation(y_pred, y_true, domain_mask)

# Helper to add border
def add_border(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

# --- Plot (4-panel layout) ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 1. Wind Input (simulated with quiver)
ax0 = axes[0]
step = 25
Y, X_grid = np.mgrid[0:IMG_H:step, 0:IMG_W:step]
U = np.ones_like(X_grid, dtype=float) * 0.7
V = np.ones_like(X_grid, dtype=float) * 0.7

# Mask outside circle
dist = np.sqrt((X_grid - center_x)**2 + (Y - center_y)**2)
valid = dist < radius
U[~valid] = np.nan
V[~valid] = np.nan

ax0.imshow(np.ones((IMG_H, IMG_W)), cmap='Greys_r', vmin=0, vmax=1, origin='lower')
ax0.quiver(X_grid, Y, U, V, color='red', scale=30, width=0.004)
ax0.set_title("Wind Input")
ax0.set_xlim(0, IMG_W)
ax0.set_ylim(0, IMG_H)
ax0.set_aspect('equal')
add_border(ax0)
# Add invisible colorbar for consistent panel sizing
sm = plt.cm.ScalarMappable(cmap='Greys', norm=plt.Normalize(0, 1))
cbar = plt.colorbar(sm, ax=ax0, fraction=0.046, pad=0.04)
cbar.ax.set_visible(False)

# 2. Ground Truth
ax1 = axes[1]
im1 = ax1.imshow(y_true, origin='lower', cmap='viridis')
ax1.set_title("Ground Truth")
add_border(ax1)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# 3. Prediction
ax2 = axes[2]
im2 = ax2.imshow(y_pred, origin='lower', cmap='viridis')
ax2.set_title("Prediction")
add_border(ax2)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# 4. Difference with metrics BELOW
ax3 = axes[3]
im3 = ax3.imshow(diff, cmap='RdBu', vmin=-2.0, vmax=2.0, origin='lower')
ax3.set_title("Diff (Pred - GT)")

# --- Display Metrics BELOW the image ---
ssim_str = f"{ssim_val:.3f}" if ssim_val >= 0 else "N/A"
metrics_line1 = f"MAE:{mae:.3f} | RMSE:{rmse:.3f} | MAPE:{mape:.1f}%"
metrics_line2 = f"SSIM:{ssim_str} | GradCorr:{grad_corr:.3f}"
metrics_text = f"{metrics_line1}\n{metrics_line2}"
ax3.set_xlabel(metrics_text, fontsize=7, family='monospace',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

add_border(ax3)
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("test_metrics_layout.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved test_metrics_layout.png")
print(f"Metrics: MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.1f}% SSIM={ssim_str} GradCorr={grad_corr:.3f}")
