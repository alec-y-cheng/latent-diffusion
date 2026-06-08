import torch
import torch.nn as nn

class GradCorrLoss(nn.Module):
    def __init__(self, per_channel=False):
        super().__init__()
        self.per_channel = per_channel

    def forward(self, pred, true, mask=None):
        """
        Compute Gradient Correlation Loss.
        Args:
            pred: Predicted image (B, C, H, W)
            true: Ground truth image (B, C, H, W)
        Returns:
            loss: 1 - Correlation Coefficient (Scalar)
        """
        b, c, _, _ = pred.shape

        pred_dy = torch.diff(pred, dim=2, prepend=pred[:, :, :1, :])
        true_dy = torch.diff(true, dim=2, prepend=true[:, :, :1, :])
        pred_dx = torch.diff(pred, dim=3, prepend=pred[:, :, :, :1])
        true_dx = torch.diff(true, dim=3, prepend=true[:, :, :, :1])

        if not self.per_channel and mask is None:
            pred_grad = torch.cat([pred_dy.reshape(b, -1), pred_dx.reshape(b, -1)], dim=1)
            true_grad = torch.cat([true_dy.reshape(b, -1), true_dx.reshape(b, -1)], dim=1)
            pred_mean = pred_grad.mean(dim=1, keepdim=True)
            true_mean = true_grad.mean(dim=1, keepdim=True)
            pred_c = pred_grad - pred_mean
            true_c = true_grad - true_mean
            denom = torch.sqrt((pred_c ** 2).sum(dim=1) * (true_c ** 2).sum(dim=1)) + 1e-5
            corr = (pred_c * true_c).sum(dim=1) / denom
            return (1.0 - corr).mean()

        pred_grad = torch.cat([pred_dy.flatten(2), pred_dx.flatten(2)], dim=2)
        true_grad = torch.cat([true_dy.flatten(2), true_dx.flatten(2)], dim=2)

        if mask is None:
            grad_mask = torch.ones_like(pred_grad)
        else:
            if mask.shape[1] == 1 and c != 1:
                mask = mask.expand(-1, c, -1, -1)
            grad_mask = torch.cat([mask.flatten(2), mask.flatten(2)], dim=2)

        weight_sum = grad_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        pred_mean = (pred_grad * grad_mask).sum(dim=2, keepdim=True) / weight_sum
        true_mean = (true_grad * grad_mask).sum(dim=2, keepdim=True) / weight_sum
        pred_c = pred_grad - pred_mean
        true_c = true_grad - true_mean
        pred_ss = (pred_c.pow(2) * grad_mask).sum(dim=2)
        true_ss = (true_c.pow(2) * grad_mask).sum(dim=2)
        denom = torch.sqrt(pred_ss * true_ss).clamp_min(1e-5)
        corr = (pred_c * true_c * grad_mask).sum(dim=2) / denom
        corr = torch.where(torch.isfinite(corr), corr, torch.zeros_like(corr))
        return (1.0 - corr).mean()
