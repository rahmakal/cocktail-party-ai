import torch
import torch.nn as nn
from itertools import permutations


def si_snr(est, target, eps=1e-8):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR).

    Args:
        est    : (B, T)
        target : (B, T)
    Returns:
        si_snr : (B,)  — une valeur par item du batch
    """
    # Centrer
    est    = est    - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Projection de est sur target
    dot      = (est * target).sum(dim=-1, keepdim=True)        # (B, 1)
    norm_t   = (target ** 2).sum(dim=-1, keepdim=True) + eps   # (B, 1)
    s_target = dot / norm_t * target                           # (B, T)

    # Bruit résiduel
    e_noise = est - s_target                                   # (B, T)

    # Ratio signal/bruit  — clamp pour éviter log10(0) = -inf
    ratio = (s_target ** 2).sum(dim=-1) / \
            ((e_noise ** 2).sum(dim=-1) + eps)                 # (B,)
    ratio = torch.clamp(ratio, min=eps)                        # ← FIX bug 2

    return 10 * torch.log10(ratio)                             # (B,)


def pit_si_snr_loss(est_sources, target_sources, eps=1e-8):
    """
    Permutation Invariant Training (PIT) avec SI-SNR.

    Args:
        est_sources    : (B, n_src, T)
        target_sources : (B, n_src, T)
    Returns:
        loss scalaire (à minimiser)
    """
    B, n_src, T = est_sources.shape
    perms = list(permutations(range(n_src)))

    # (B, n_perms) — SI-SNR moyen par permutation et par item
    all_losses = []

    for perm in perms:
        perm_est = est_sources[:, perm, :]      # (B, n_src, T)

        # SI-SNR pour chaque source → (n_src, B) puis moyenne sur sources → (B,)
        snr_per_src = torch.stack([
            si_snr(perm_est[:, i], target_sources[:, i], eps)
            for i in range(n_src)
        ], dim=0)                               # (n_src, B)

        mean_snr = snr_per_src.mean(dim=0)     # (B,)  ← FIX bug 1 : par item
        all_losses.append(-mean_snr)            # (B,)  on minimise le négatif

    # Stack → (n_perms, B), prendre la meilleure permutation par item
    all_losses = torch.stack(all_losses, dim=0)         # (n_perms, B)
    best_loss, _ = all_losses.min(dim=0)                # (B,)

    return best_loss.mean()                             # scalaire


class SISNRLoss(nn.Module):
    """Wrapper nn.Module de la loss PIT-SI-SNR."""

    def __init__(self):
        super().__init__()

    def forward(self, est_sources, target_sources):
        return pit_si_snr_loss(est_sources, target_sources)