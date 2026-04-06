import torch
from torch import nn
import torch.nn.functional as F

from networks.magicnet import Encoder, Decoder, FcLayer


class VNet_Magic_SCDL_Prior5(nn.Module):
    """
    VNet_Magic variant that injects an external SCDL prior only at the
    deepest decoder stage (bottleneck, a.k.a. x5).
    """

    def __init__(
        self,
        n_channels=1,
        n_classes=2,
        cube_size=32,
        patch_size=96,
        n_filters=16,
        normalization="instancenorm",
        has_dropout=False,
        has_residual=False,
        prior_ch=512,
        prior_weight=1.0,
    ):
        super().__init__()
        self.num_classes = n_classes
        self.prior_weight = float(prior_weight)

        self.encoder = Encoder(
            n_channels, n_classes, n_filters, normalization, has_dropout, has_residual
        )
        self.decoder = Decoder(
            n_channels, n_classes, n_filters, normalization, has_dropout, has_residual
        )

        # Project prior to bottleneck channels (16 * n_filters).
        self.prior_proj5 = nn.Conv3d(prior_ch, 16 * n_filters, kernel_size=1, bias=False)

        # Keep the same location head interface as original MagicNet.
        self.fc_layer = FcLayer(cube_size, patch_size)

    @staticmethod
    def combine_prior_and_samples(prior_3d, z_sampled_norm):
        """
        Combine SCDL prior_3d and z_sampled_norm into a single prior map.
        prior_3d:      [B, D, Dt, Ht, Wt]
        z_sampled_norm:[B, L, E, D]
        return:        [B, 2D, Dt, Ht, Wt]
        """
        if z_sampled_norm is None:
            return prior_3d
        if prior_3d is None:
            raise ValueError("prior_3d is required to reshape z_sampled_norm.")

        z_agg = z_sampled_norm.mean(dim=2)  # [B, L, D]
        B, L, D = z_agg.shape
        Dt, Ht, Wt = prior_3d.shape[-3:]
        if L != Dt * Ht * Wt:
            raise ValueError(
                "Token length L does not match prior_3d spatial size: "
                f"L={L}, Dt*Ht*Wt={Dt*Ht*Wt}."
            )
        z_agg_map = (
            z_agg.transpose(1, 2)
            .contiguous()
            .view(B, D, Dt, Ht, Wt)
        )
        return torch.cat([prior_3d, z_agg_map], dim=1)

    def _apply_prior_to_x5(self, features, prior):
        if (prior is None) or (self.prior_weight == 0.0):
            return features

        x1, x2, x3, x4, x5 = features
        if prior.shape[-3:] != x5.shape[-3:]:
            prior = F.interpolate(
                prior, size=x5.shape[-3:], mode="trilinear", align_corners=False
            )
        x5 = x5 + self.prior_proj5(prior) * self.prior_weight
        return [x1, x2, x3, x4, x5]

    def forward_prediction_head(self, feat):
        return self.decoder.out_conv(feat)

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_decoder(self, feat_list, prior=None):
        feat_list = self._apply_prior_to_x5(feat_list, prior)
        return self.decoder(feat_list)

    def forward(self, input, prior=None):
        features = self.encoder(input)
        features = self._apply_prior_to_x5(features, prior)
        out_seg, embedding = self.decoder(features)
        return out_seg, embedding


__all__ = ["VNet_Magic_SCDL_Prior5"]
