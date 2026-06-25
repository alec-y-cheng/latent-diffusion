import torch
import torch.nn as nn
import torch.nn.functional as F


class Pix2PixGradientLoss(nn.Module):
    """Depthwise Sobel gradient matching adapted from the 4-channel Pix2PixHD model."""

    def __init__(self, channel_weights=None, eps=1e-6):
        super().__init__()
        if isinstance(channel_weights, str):
            channel_weights = [
                float(value) for value in channel_weights.split(":") if value
            ]
        self.channel_weights = channel_weights
        self.eps = eps

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]]
        ).unsqueeze(0) / 8.0
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0],
              [0.0, 0.0, 0.0],
              [1.0, 2.0, 1.0]]]
        ).unsqueeze(0) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _spatial_gradients(self, field):
        channels = field.shape[1]
        kernel_x = self.sobel_x.to(dtype=field.dtype).repeat(channels, 1, 1, 1)
        kernel_y = self.sobel_y.to(dtype=field.dtype).repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(field, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(field, kernel_y, padding=1, groups=channels)
        return grad_x, grad_y

    def _channel_mean(self, value, weight):
        if weight is None:
            return value.mean(dim=(0, 2, 3))
        if weight.shape[1] == 1 and value.shape[1] != 1:
            weight = weight.expand(-1, value.shape[1], -1, -1)
        numerator = (value * weight).sum(dim=(0, 2, 3))
        denominator = weight.sum(dim=(0, 2, 3)).clamp_min(self.eps)
        return numerator / denominator

    def _weighted_channels(self, values):
        if self.channel_weights is None:
            return values.mean()
        weights = values.new_tensor(self.channel_weights)
        if weights.numel() != values.numel():
            raise ValueError(
                f"Expected {values.numel()} Pix2Pix channel weights, got {weights.numel()}."
            )
        return (values * weights).sum() / weights.sum().clamp_min(self.eps)

    def forward(self, pred, target, mask=None, sample_mask=None):
        if sample_mask is not None:
            sample_mask = sample_mask.to(device=pred.device, dtype=pred.dtype)
            sample_mask = sample_mask.view(-1, 1, 1, 1)
            mask = sample_mask if mask is None else mask * sample_mask

        pred_x, pred_y = self._spatial_gradients(pred)
        target_x, target_y = self._spatial_gradients(target)
        error = (pred_x - target_x).abs() + (pred_y - target_y).abs()
        per_channel = self._channel_mean(error, mask)
        return self._weighted_channels(per_channel), per_channel
