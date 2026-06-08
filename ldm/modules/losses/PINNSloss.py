import torch
import torch.nn as nn
import torch.nn.functional as F


class PINNSLoss(nn.Module):
    def __init__(
        self,
        lambda_res=1.0,
        lambda_bc=1.0,
        lambda_smooth=0.25,
        lambda_range=0.05,
        lambda_masked_recon=0.0,
        lambda_roof_background=0.0,
        lambda_floor_background=0.0,
        turbulence_smooth_weight=1.0,
        roof_smooth_weight=1.0,
        sdf_channel=0,
        building_channel=1,
        wind_x_channel=6,
        wind_y_channel=7,
        building_mask_sharpness=20.0,
        eps=1e-6,
    ):
        super().__init__()
        self.lambda_res = lambda_res
        self.lambda_bc = lambda_bc
        self.lambda_smooth = lambda_smooth
        self.lambda_range = lambda_range
        self.lambda_masked_recon = lambda_masked_recon
        self.lambda_roof_background = lambda_roof_background
        self.lambda_floor_background = lambda_floor_background
        self.turbulence_smooth_weight = turbulence_smooth_weight
        self.roof_smooth_weight = roof_smooth_weight
        self.sdf_channel = sdf_channel
        self.building_channel = building_channel
        self.wind_x_channel = wind_x_channel
        self.wind_y_channel = wind_y_channel
        self.building_mask_sharpness = building_mask_sharpness
        self.eps = eps

    @staticmethod
    def _spatial_grads(x):
        dy = torch.diff(x, dim=2, prepend=x[:, :, :1, :])
        dx = torch.diff(x, dim=3, prepend=x[:, :, :, :1])
        return dy, dx

    @staticmethod
    def _weighted_mean(value, weight, eps):
        if weight.shape[1] == 1 and value.shape[1] != 1:
            weight = weight.expand(-1, value.shape[1], -1, -1)
        return (value * weight).sum() / weight.sum().clamp_min(eps)

    def _resize_cond(self, cond_pixel, height, width):
        if cond_pixel.shape[-2:] == (height, width):
            return cond_pixel
        return F.interpolate(cond_pixel, size=(height, width), mode="bilinear", align_corners=False)

    def _channel(self, cond, idx):
        if cond.shape[1] <= idx:
            return torch.zeros_like(cond[:, :1])
        return cond[:, idx:idx + 1]

    def _geometry_masks(self, cond):
        # Building height is normalized to [-1, 1]; zero raw height is normally
        # near -1, so this softly selects occupied footprint pixels.
        bldg = self._channel(cond, self.building_channel)
        solid = torch.sigmoid((bldg + 0.95) * self.building_mask_sharpness)

        sdf = self._channel(cond, self.sdf_channel)
        sdf_dy, sdf_dx = self._spatial_grads(sdf)
        sdf_edge = torch.sqrt(sdf_dx.pow(2) + sdf_dy.pow(2) + self.eps)
        sdf_edge = sdf_edge / sdf_edge.amax(dim=(2, 3), keepdim=True).clamp_min(self.eps)

        dilated = F.max_pool2d(solid, kernel_size=5, stride=1, padding=2)
        eroded = -F.max_pool2d(-solid, kernel_size=5, stride=1, padding=2)
        wall_band = (dilated - eroded).clamp(0.0, 1.0)

        boundary = torch.maximum(wall_band, sdf_edge * (1.0 - solid)).clamp(0.0, 1.0)
        fluid = (1.0 - solid).clamp(0.0, 1.0)
        far_fluid = (fluid * (1.0 - boundary)).clamp(0.0, 1.0)
        return solid, fluid, boundary, far_fluid

    def _target_masks(self, solid, fluid, channels):
        if channels != 4:
            return torch.ones(
                solid.shape[0], channels, solid.shape[2], solid.shape[3],
                device=solid.device, dtype=solid.dtype
            )
        return torch.cat([fluid, fluid, solid, solid], dim=1)

    def _channel_weighted_mean(self, value, weight):
        return self._channel_means(value, weight).mean()

    def _channel_means(self, value, weight):
        if weight.shape[1] == 1 and value.shape[1] != 1:
            weight = weight.expand(-1, value.shape[1], -1, -1)
        numer = (value * weight).sum(dim=(0, 2, 3))
        denom = weight.sum(dim=(0, 2, 3)).clamp_min(self.eps)
        return numer / denom

    def forward(self, pred_pixel, true_pixel, cond_pixel):
        """
        CFD physics-inspired regularizer for scalar speed/turbulence targets.

        This is intentionally condition-aware: it applies PDE-like smoothness in
        fluid regions and boundary matching near building geometry. Full
        Navier-Stokes residuals would require vector velocity/pressure fields,
        which these four scalar target channels do not provide.
        """
        _, _, h, w = pred_pixel.shape
        cond = self._resize_cond(cond_pixel, h, w)
        solid, fluid, boundary, far_fluid = self._geometry_masks(cond)
        target_masks = self._target_masks(solid, fluid, pred_pixel.shape[1])

        dy, dx = self._spatial_grads(pred_pixel)
        dy2, _ = self._spatial_grads(dy)
        _, dx2 = self._spatial_grads(dx)
        laplacian = dx2 + dy2

        wx = self._channel(cond, self.wind_x_channel)
        wy = self._channel(cond, self.wind_y_channel)
        wind_norm = torch.sqrt(wx.pow(2) + wy.pow(2) + self.eps)
        wx = wx / wind_norm
        wy = wy / wind_norm

        d_parallel = wx * dx + wy * dy
        d_cross = -wy * dx + wx * dy

        if pred_pixel.shape[1] == 4:
            smooth_channel_weights = pred_pixel.new_tensor([
                1.0,
                self.turbulence_smooth_weight,
                self.roof_smooth_weight,
                self.roof_smooth_weight * self.turbulence_smooth_weight,
            ]).view(1, 4, 1, 1)
        else:
            smooth_channel_weights = torch.ones_like(pred_pixel[:, :, :1, :1])

        l_res = self._channel_weighted_mean(laplacian.pow(2), target_masks)
        smooth_per_channel = self._channel_means(
            0.25 * d_parallel.pow(2) + d_cross.pow(2),
            target_masks,
        )
        smooth_weights = smooth_channel_weights.flatten()
        l_smooth = (
            smooth_per_channel * smooth_weights
        ).sum() / smooth_weights.sum().clamp_min(self.eps)
        l_bc = self._weighted_mean((pred_pixel - true_pixel).pow(2), boundary, self.eps)
        l_range = (F.relu(pred_pixel - 1.0).pow(2) + F.relu(-1.0 - pred_pixel).pow(2)).mean()
        l_masked_recon = self._channel_weighted_mean(
            (pred_pixel - true_pixel).pow(2),
            target_masks,
        )

        if pred_pixel.shape[1] >= 4:
            roof_background = fluid.expand(-1, 2, -1, -1)
            floor_background = solid.expand(-1, 2, -1, -1)
            l_roof_background = self._weighted_mean(
                (pred_pixel[:, 2:4] + 1.0).pow(2),
                roof_background,
                self.eps,
            )
            l_floor_background = self._weighted_mean(
                (pred_pixel[:, 0:2] + 1.0).pow(2),
                floor_background,
                self.eps,
            )
        else:
            l_roof_background = pred_pixel.new_zeros(())
            l_floor_background = pred_pixel.new_zeros(())

        total = (
            self.lambda_res * l_res
            + self.lambda_bc * l_bc
            + self.lambda_smooth * l_smooth
            + self.lambda_range * l_range
            + self.lambda_masked_recon * l_masked_recon
            + self.lambda_roof_background * l_roof_background
            + self.lambda_floor_background * l_floor_background
        )

        return total, {
            "loss_pinn_res": l_res,
            "loss_pinn_smooth": l_smooth,
            "loss_pinn_bc": l_bc,
            "loss_pinn_range": l_range,
            "loss_masked_recon": l_masked_recon,
            "loss_roof_background": l_roof_background,
            "loss_floor_background": l_floor_background,
        }
