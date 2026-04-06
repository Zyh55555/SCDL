# from Models.unet import build_model as unet_build_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import t as StudentT

def mask_to_token_dist(y, num_classes, Ht, Wt):
    # y: [B,H,W] long
    B,H,W = y.shape
    y_oh = F.one_hot(y, num_classes=num_classes).permute(0,3,1,2).float()  # [B,C,H,W]
    # avg pool to token grid
    y_dist = F.interpolate(y_oh, size=(Ht, Wt), mode="area")
    y_dist = y_dist.clamp_min(0.0)
    y_dist = y_dist / (y_dist.sum(dim=1, keepdim=True).clamp_min(1e-8))

    return y_dist  # [B,C,Ht,Wt]


class SCDL(nn.Module):
    def __init__(self,
                 x_dim,
                 z_dim=256,
                 sample_num=50,
                 num_classes=14,
                 seed=1,
                 batch_size=16,
                 dist_loss_weight=0.1,
                 sup_weight=0.1,
                 sample_num_emb=2,
                 init_sigma_emb=0.05,
                 tau_e2p=0.2,
                 tau_p2e=0.2):   # >>> NEW <<<
        super(SCDL, self).__init__()

        self.sample_num = sample_num      # S: proxy sample per class
        self.num_classes = num_classes    # C
        self.seed = seed
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.dist_loss_weight = dist_loss_weight  # >>> NEW <<<
        self.sup_weight = sup_weight
        self.sample_num_emb = int(sample_num_emb)
        self.sigma_emb = nn.Parameter(init_sigma_emb * torch.ones(1, 1, 1, z_dim))
        self.tau_e2p=tau_e2p
        self.tau_p2e=tau_p2e

        # Encoder: input feature -> embedding
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, z_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim * 2, z_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim * 2, z_dim),
        )

        # Proxies: each class has μ and σ
        self.proxies = nn.Parameter(torch.empty([num_classes, z_dim * 2]))
        torch.nn.init.xavier_uniform_(self.proxies, gain=1.0)

    def gaussian_noise(self, shape):
        """Generate Gaussian noise for proxy sampling"""
        device = self.proxies.device
        # gen = torch.Generator(device=device).manual_seed(self.seed)
        return torch.normal(torch.zeros(*shape, device=device),
                            torch.ones(*shape, device=device))

    def encoder_result(self, x):
        """Forward through encoder"""
        return self.encoder(x)

    def encoder_proxies(self):
        """Return μ and σ for each class"""
        mu = self.proxies[:, :self.z_dim]                 # [C, D]
        sigma = F.softplus(self.proxies[:, self.z_dim:])  # [C, D]
        return mu, sigma

    def forward(self, x, y_dist_flat=None, pseudo_tok=None, x_class_tok=None, y_class_idx=None):
        """
        x: [B, L, D] patch-level input
        y_tok: [B, L] class labels for each patch
        pseudo_tok: [B, L] pseudo labels for each patch (if available)
        x_class_tok: Tensor [B, C_sel, D]
            Class-aware token embeddings.
            Obtained by cropping / masking x according to GT labels,
            where each token represents a semantic class–focused region
            (background suppressed).
        y_class_idx: LongTensor [B, C_sel]
            Class indices corresponding to x_class_tok.
            Specifies which semantic class each class-aware token belongs to.
            Used for class-level proxy alignment or semantic prior learning.
        """
        B, L, D = x.shape
        z = self.encoder_result(x)  # [B, L, D]

        # --------------------------------------------------
        # Proxy distributions
        # --------------------------------------------------
        mu_proxy, sigma_proxy = self.encoder_proxies()  # [C, D]
        # ============================================================
        # >>> NEW <<< Class-level semantic anchor loss (x_class_tok ↔ μ)
        # ============================================================
        class_anchor_loss = torch.tensor(0.0, device=x.device)

        if (x_class_tok is not None) and (y_class_idx is not None):
            # --------------------------------------------------
            # Normalize x_class_tok shape to [N_inst, D]
            # --------------------------------------------------
            if x_class_tok.dim() == 3:
                # [N_inst, Lc, D] -> first take the mean inside each class instance
                x_cls_feat = x_class_tok.mean(dim=1)   # [N_inst, D]
            else:
                # Already [N_inst, D]
                x_cls_feat = x_class_tok

            # Normalize (consistent with proxy)
            # x_cls_feat = F.normalize(x_cls_feat, dim=-1)   # [N_inst, D]
            z_cls = self.encoder_result(x_cls_feat)   # [N_inst, 256]

            z_cls = F.normalize(z_cls, dim=-1)       # [N_inst, 256]
            mu_norm = F.normalize(mu_proxy, dim=-1)        # [C, D]

            # --------------------------------------------------
            # Aggregate by class to obtain one semantic center per class
            # --------------------------------------------------
            unique_classes = torch.unique(y_class_idx)
            per_class_losses = []

            for c in unique_classes:
                mask = (y_class_idx == c)   # [N_inst]
                if mask.sum() == 0:
                    continue

                # Semantic center of this class
                # cls_center = x_cls_feat[mask].detach().mean(dim=0)   # [D]
                cls_center = z_cls[mask].detach().mean(dim=0)  # [256]
                cls_center = F.normalize(cls_center, dim=-1)
                # Corresponding proxy center
                mu_c = mu_norm[c]   # [D]

                # --------------------------------------------------
                # cosine anchor loss
                # --------------------------------------------------
                loss_c = 1.0 - torch.sum(cls_center * mu_c)
                per_class_losses.append(loss_c)

            if len(per_class_losses) > 0:
                class_anchor_loss = torch.stack(per_class_losses).mean()

        eps = self.gaussian_noise(
            [self.num_classes, self.sample_num, self.z_dim]
        )  # [C, S, D]

        z_proxy_sample = (
            mu_proxy.unsqueeze(1) +
            sigma_proxy.unsqueeze(1) * eps
        )  # [C, S, D]

        # Normalize embeddings and proxies
        z_norm = F.normalize(z, dim=-1)                 # [B, L, D]
        z_proxy_norm = F.normalize(z_proxy_sample, dim=-1)  # [C, S, D]

       # -----------------------------
        # >>> Added: embedding sampling
        # -----------------------------
        Dz = z.shape[-1]  # 256
        # -----------------------------
        # >>> Added: embedding sampling
        # -----------------------------
        E = self.sample_num_emb
        # eps_emb: [B, L, E, D]
        eps_emb = torch.randn(B, L, E, Dz, device=z.device)
        # Sampled embeddings: [B, L, E, D]
        z_sampled = z.unsqueeze(2) + eps_emb * self.sigma_emb
        # Normalize the E samples
        z_sampled_norm = F.normalize(z_sampled, dim=-1)

        # --------------------------------------------------
        # Similarity: embedding ↔ proxy samples
        # --------------------------------------------------
        # [B, C, S, L]
        att = torch.einsum('bld,csd->bcls', z_norm, z_proxy_norm)
        assert att.shape == (B, self.num_classes, L, self.sample_num), att.shape
        mu_norm = F.normalize(mu_proxy, dim=-1)       # [C,D]
        logits_e2p = torch.einsum('bld,cd->blc', z_norm, mu_norm)  # [B,L,C]
        probs_e2p  = F.softmax(logits_e2p / self.tau_e2p, dim=-1)

        # --------------------------------------------------
        # Added: proxy -> embedding soft assignment
        # --------------------------------------------------
        logits_p2e = torch.einsum('cd,bld->bcl', mu_norm, z_norm)   # [B,C,L]
        probs_p2e = F.softmax(logits_p2e / self.tau_p2e, dim=1)

        # Align dimensions
        probs_p2e_t = probs_p2e.permute(0, 2, 1)   # [B, L, C]

        # -----------------------------------
        # Prevent log(0)
        # -----------------------------------
        probs_e2p_safe = probs_e2p.clamp(min=1e-8)
        probs_p2e_t_safe = probs_p2e_t.clamp(min=1e-8)

        # -----------------------------------
        # Symmetric KL divergence
        # -----------------------------------
        loss_e2p_p2e_kl = F.kl_div(probs_e2p_safe.log(), probs_p2e_t_safe, reduction='batchmean') + \
                F.kl_div(probs_p2e_t_safe.log(), probs_e2p_safe, reduction='batchmean')

        # Symmetric consistency (per class)
        bi_sim = probs_e2p * probs_p2e_t            # [B, L, C]

        #  Optional: normalize into a distribution
        # (for each token over all classes)
        bi_sim = bi_sim / (bi_sim.sum(dim=-1, keepdim=True) + 1e-6)



        prior_p2e_tok = torch.einsum('bcl,cd->bld', probs_p2e, mu_norm)
        prior_p2e_tok = F.normalize(prior_p2e_tok, dim=-1)

        # --------------------------------------------------
        # Original proxy contrastive-style loss 
        # --------------------------------------------------
        C, S = z_proxy_norm.shape[:2]
        att_flat = att.permute(0, 2, 1, 3).reshape(B * L, C, S)  # [B*L, C, S]

        if y_dist_flat is not None:
            # With GT: use the real token semantic proportion
            w = y_dist_flat.reshape(B*L, C)
        else:
            # Without GT: use model prediction, but stop gradient
            w = bi_sim.reshape(B*L, C).detach()

        # positive/negative weighted similarity
        pos_mean = (att_flat * w.unsqueeze(-1)).mean(dim=2)
        neg_mean = (att_flat * (1.0 - w).unsqueeze(-1)).mean(dim=2)

        loss_per_class = torch.exp(-(pos_mean - neg_mean)).mean(dim=0)
        proxy_loss = loss_per_class.mean()

        # --------------------------------------------------
        # >>> NEW <<< Soft distribution consistency loss
        # --------------------------------------------------
        # Distance metric: 1 - cosine similarity
        dist = 1.0 - att_flat                     # [B*L, C, S]
        dist_mean = dist.mean(dim=2)              # E_s[dist], [B*L, C]

        # Only enforce where embedding has high "reference" to proxy
        # dist_loss = (probs_flat * dist_mean).sum(dim=1).mean()
        dist_loss = (w * dist_mean).sum(dim=1).mean()

        # --------------------------------------------------
        # >>> NEW <<< Automatic scale option
        # --------------------------------------------------
        if hasattr(self, 'auto_scale_dist') and self.auto_scale_dist:
            with torch.no_grad():
                scale = proxy_loss.detach() / (dist_loss.detach() + 1e-6)
            proxy_loss = proxy_loss + scale * dist_loss
        else:
            proxy_loss = proxy_loss + self.dist_loss_weight * dist_loss
        return proxy_loss, prior_p2e_tok, probs_e2p, z_sampled_norm, class_anchor_loss, loss_e2p_p2e_kl



def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5

def visualize_student_t_distributions(mu_pos, sigma_pos, v_pos, mu_neg, sigma_neg, v_neg, title, filename):
    num_distributions = len(mu_pos)
    num_cols = 4
    num_rows = (num_distributions + num_cols - 1) // num_cols  
    x = np.linspace(-0.1, 0.1, 1000)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12))
    axes = axes.flatten() 

    for i in range(num_distributions):
        y_pos = StudentT.pdf(x, df=v_pos[i], loc=mu_pos[i], scale=sigma_pos[i]) 
        y_neg = StudentT.pdf(x, df=v_neg[i], loc=mu_neg[i], scale=sigma_neg[i]) 
        axes[i].plot(x, y_pos, label=f'Positive (v={v_pos[i]:.8f}, loc={mu_pos[i]:.8f}, scale={sigma_pos[i]:.8f})',
                     color='blue')
        axes[i].plot(x, y_neg, label=f'Negative (v={v_neg[i]:.8f}, loc={mu_neg[i]:.8f}, scale={sigma_neg[i]:.8f})',
                     color='red')
        axes[i].set_title(f'Sample {i + 1}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('Probability Density')
        axes[i].legend()
        axes[i].grid(True)

    for i in range(num_distributions, num_rows * num_cols):
        fig.delaxes(axes[i])

    fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    plt.savefig(filename, format='pdf')

def gn_groups(c, max_groups=8):
    max_groups = min(max_groups, c)
    for g in [max_groups, 4, 2, 1]:
        if c % g == 0:
            return g
    return 1
 

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm="gn"):
        super().__init__()
        if norm == "bn":
            Norm = lambda c: nn.BatchNorm2d(c)
        else:
            Norm = lambda c: nn.GroupNorm(num_groups=gn_groups(c), num_channels=c)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UpFuseBlock(nn.Module):
    """
    up(x) + skip + up(prior) -> fuse
    """
    def __init__(self, x_in_ch, skip_ch, out_ch, prior_ch, norm="gn"):
        super().__init__()
        self.prior_proj = nn.Conv2d(prior_ch, out_ch, kernel_size=1, bias=False)
        self.fuse = DoubleConv(x_in_ch + skip_ch + out_ch, out_ch, norm=norm)

    def forward(self, x, skip, prior):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        prior = F.interpolate(prior, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        p = self.prior_proj(prior)                 # [B, out_ch, H, W]
        x = torch.cat([x, skip, p], dim=1)         # [B, x_in_ch+skip_ch+out_ch, H, W]
        x = self.fuse(x)                           # [B, out_ch, H, W]
        return x, prior

class UNetDecoderWithPrior(nn.Module):
    def __init__(self, base=64, token_dim=1024, prior_ch=512, num_classes=14, norm="gn"):
        super().__init__()
        b = base
        self.num_classes = num_classes
        self.stem = DoubleConv(token_dim + prior_ch, 16*b, norm=norm)  # -> [B,16b,12,12]

        # 12->24 (skip x4: 16b)  output 16b
        self.up4 = UpFuseBlock(x_in_ch=16*b, skip_ch=16*b, out_ch=16*b, prior_ch=prior_ch, norm=norm)
        # 24->48 (skip x3: 8b)   output 8b
        self.up3 = UpFuseBlock(x_in_ch=16*b, skip_ch=8*b,  out_ch=8*b,  prior_ch=prior_ch, norm=norm)
        # 48->96 (skip x2: 4b)   output 4b
        self.up2 = UpFuseBlock(x_in_ch=8*b,  skip_ch=4*b,  out_ch=4*b,  prior_ch=prior_ch, norm=norm)
        # 96->192 (skip x1: 2b)  output 2b
        self.up1 = UpFuseBlock(x_in_ch=4*b,  skip_ch=2*b,  out_ch=2*b,  prior_ch=prior_ch, norm=norm)
        # 192->384 (skip x0: b)  output b
        self.up0 = UpFuseBlock(x_in_ch=2*b,  skip_ch=b,    out_ch=b,    prior_ch=prior_ch, norm=norm)

        self.head = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, fundus_out, prior_12):
        x0 = fundus_out["x0"]
        x1 = fundus_out["x1"]
        x2 = fundus_out["x2"]
        x3 = fundus_out["x3"]
        x4 = fundus_out["x4"]
        feat_12 = fundus_out["feat"]  # [B,token_dim,12,12]

        # Starting point
        x = torch.cat([feat_12, prior_12], dim=1)   # [B, token_dim+prior_ch, 12,12]
        x = self.stem(x)                             # [B, 16b, 12,12]

        # UNet-style decode with prior at every stage
        x, prior = self.up4(x, x4, prior_12)   # 24
        x, prior = self.up3(x, x3, prior)      # 48
        x, prior = self.up2(x, x2, prior)      # 96
        x, prior = self.up1(x, x1, prior)      # 192
        x, prior = self.up0(x, x0, prior)      # 384

        logits = self.head(x)                  # [B,num_classes,384,384]
        return logits





        
