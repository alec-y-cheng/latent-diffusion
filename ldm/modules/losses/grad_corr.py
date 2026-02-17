import torch
import torch.nn as nn

class GradCorrLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, true):
        """
        Compute Gradient Correlation Loss.
        Args:
            pred: Predicted image (B, C, H, W)
            true: Ground truth image (B, C, H, W)
        Returns:
            loss: 1 - Correlation Coefficient (Scalar)
        """
        b, c, h, w = pred.shape
        
        # Compute Gradients
        # dy: Diff along H (axis 2)
        pred_dy = torch.diff(pred, dim=2, prepend=pred[:, :, :1, :])
        true_dy = torch.diff(true, dim=2, prepend=true[:, :, :1, :])
        
        # dx: Diff along W (axis 3)
        pred_dx = torch.diff(pred, dim=3, prepend=pred[:, :, :, :1])
        true_dx = torch.diff(true, dim=3, prepend=true[:, :, :, :1])
        
        # Flatten gradients
        pred_grad = torch.cat([pred_dy.reshape(b, -1), pred_dx.reshape(b, -1)], dim=1)
        true_grad = torch.cat([true_dy.reshape(b, -1), true_dx.reshape(b, -1)], dim=1)
        
        # Compute Correlation per sample
        pred_mean = pred_grad.mean(dim=1, keepdim=True)
        true_mean = true_grad.mean(dim=1, keepdim=True)
        
        pred_c = pred_grad - pred_mean
        true_c = true_grad - true_mean
        
        pred_ss = (pred_c ** 2).sum(dim=1)
        true_ss = (true_c ** 2).sum(dim=1)
        
        denom = torch.sqrt(pred_ss * true_ss) + 1e-8
        
        cov = (pred_c * true_c).sum(dim=1)
        corr = cov / denom
        
        # Loss = 1 - Correlation (Maximize correlation)
        loss = 1.0 - corr
        
        return loss.mean()
