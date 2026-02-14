import torch
import torch.nn as nn

class GradCorrLoss(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug

    def compute_gradient_correlation(self, pred, target):
        """
        Compute Gradient Correlation between pred and target images.
        pred, target: (B, C, H, W)
        """
        # Ensure inputs are float
        pred = pred.float()
        target = target.float()
        
        # Compute Gradients in X and Y using standard difference
        # Similar to np.diff(axis=..., prepend=...)
        # dim 3 = W (x), dim 2 = H (y)
        
        # Dx
        # Append Ref padding or just diff
        # torch.diff was added in 1.8. Assuming environment has it.
        # If not, naive slice: tensor[..., 1:] - tensor[..., :-1]
        
        try:
            pred_dx = torch.diff(pred, dim=3, prepend=pred[..., :1])
            pred_dy = torch.diff(pred, dim=2, prepend=pred[..., :1, :])
            target_dx = torch.diff(target, dim=3, prepend=target[..., :1])
            target_dy = torch.diff(target, dim=2, prepend=target[..., :1, :])
        except AttributeError:
            # Fallback for older torch components if needed
            pred_dx = pred[..., 1:] - pred[..., :-1]
            # Pad to maintain shape? Or just concatenate flat.
            # User's numpy code: np.diff(pred, axis=1, prepend=pred[:, :1]) -> retains shape
            # We can replicate prepend logic manually for older torch
            pred_dx = torch.cat([pred[..., :1], pred[..., 1:]], dim=3) - torch.cat([pred[..., :1], pred[..., :-1]], dim=3)
            # Actually simpler: append first col/row
            # pred_dx = pred - torch.roll(pred, 1, dims=3)?
            # Let's stick to torch.diff if possible, or simple slice if not strict.
            # But wait, grad corr flattens everything anyway.
            # Let's implement the 'prepend' logic manually to be safe across versions.
            
            def safe_diff(x, dim):
                # prepend x[..., :1]
                if dim == 3: # W
                    prep = x[..., :1]
                elif dim == 2: # H
                    prep = x[..., :1, :]
                
                padded = torch.cat([prep, x], dim=dim)
                return padded.narrow(dim, 1, x.shape[dim]) - padded.narrow(dim, 0, x.shape[dim])

            pred_dx = safe_diff(pred, 3)
            pred_dy = safe_diff(pred, 2)
            target_dx = safe_diff(target, 3)
            target_dy = safe_diff(target, 2)

        # Flatten gradients per sample
        B = pred.shape[0]
        pred_grad = torch.cat([pred_dx.reshape(B, -1), pred_dy.reshape(B, -1)], dim=1)
        target_grad = torch.cat([target_dx.reshape(B, -1), target_dy.reshape(B, -1)], dim=1)
        
        # Compute Pearson Correlation
        # centered
        pred_mean = pred_grad.mean(dim=1, keepdim=True)
        target_mean = target_grad.mean(dim=1, keepdim=True)
        
        pred_centered = pred_grad - pred_mean
        target_centered = target_grad - target_mean
        
        # sum of products
        numerator = (pred_centered * target_centered).sum(dim=1)
        
        # sum of squares
        pred_ss = (pred_centered ** 2).sum(dim=1)
        target_ss = (target_centered ** 2).sum(dim=1)
        
        denominator = torch.sqrt(pred_ss * target_ss + 1e-8)
        
        corr = numerator / denominator
        
        return corr

    def forward(self, pred, target):
        """
        Return 1 - correlation (to minimize)
        """
        corr = self.compute_gradient_correlation(pred, target)
        # We want to maximize correlation (1.0), so minimize (1 - corr)
        loss = 1.0 - corr
        return loss.mean()
