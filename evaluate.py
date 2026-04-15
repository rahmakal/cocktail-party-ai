"""
evaluate.py — Évaluation complète du modèle sur le test set
Usage :
    python evaluate.py
    python evaluate.py --ckpt checkpoints_3src/best.ckpt
    python evaluate.py --ckpt checkpoints/epoch_30.ckpt

Installer les métriques optionnelles :
    pip install pesq pystoi
"""

import os
import argparse
import yaml
import torch
import torchaudio
from itertools import permutations

from src.dataset import get_dataloaders
from src.model   import build_model, load_checkpoint
from src.loss    import si_snr


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=str, default="checkpoints/best.ckpt")
    p.add_argument("--train_cfg",  type=str, default="configs/train.yaml")
    p.add_argument("--data_cfg",   type=str, default="configs/data.yaml")
    p.add_argument("--save_audio", action="store_true",
                   help="Sauvegarder les sources séparées en .wav")
    p.add_argument("--out_dir",    type=str, default="outputs/eval_audio")
    return p.parse_args()


# ─────────────────────────────────────────────
#  SDR
# ─────────────────────────────────────────────
def sdr(est, target, eps=1e-8):
    """Signal-to-Distortion Ratio — (1, T) → scalar"""
    dot    = (est * target).sum(dim=-1, keepdim=True)
    norm_t = (target ** 2).sum(dim=-1, keepdim=True) + eps
    s_true = dot / norm_t * target
    noise  = est - s_true
    ratio  = (s_true ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(torch.clamp(ratio, min=eps))


# ─────────────────────────────────────────────
#  Meilleure permutation (PIT) + toutes métriques
# ─────────────────────────────────────────────
def best_permutation(est_sources, target_sources, mixture, sr, use_pesq, use_stoi):
    """
    Trouve la permutation qui maximise le SI-SNR moyen.
    Calcule ensuite SI-SNR, SI-SNRi, SDR, SDRi, PESQ, STOI.

    Args:
        est_sources    : (n_src, T) tensor
        target_sources : (n_src, T) tensor
        mixture        : (T,) tensor
        sr             : int
        use_pesq       : bool
        use_stoi       : bool

    Returns:
        (est_reordered, dict métriques)
    """
    n_src = est_sources.shape[0]

    # ── Trouver la meilleure permutation ─────
    best_snr  = -float("inf")
    best_perm = list(range(n_src))

    for perm in permutations(range(n_src)):
        perm_est = est_sources[list(perm)]
        snr_vals = torch.stack([
            si_snr(perm_est[i].unsqueeze(0), target_sources[i].unsqueeze(0))
            for i in range(n_src)
        ]).mean().item()
        if snr_vals > best_snr:
            best_snr  = snr_vals
            best_perm = list(perm)

    est_reordered = est_sources[best_perm]   # (n_src, T)

    # ── SI-SNR, SI-SNRi, SDR, SDRi ───────────
    si_snr_vals, si_snri_vals = [], []
    sdr_vals,    sdri_vals    = [], []

    for i in range(n_src):
        est_i = est_reordered[i].unsqueeze(0)   # (1, T)
        ref_i = target_sources[i].unsqueeze(0)  # (1, T)
        mix_i = mixture.unsqueeze(0)            # (1, T)

        sisnr_val = si_snr(est_i, ref_i).item()
        sisnr_mix = si_snr(mix_i, ref_i).item()
        sdr_val   = sdr(est_i, ref_i).item()
        sdr_mix   = sdr(mix_i, ref_i).item()

        si_snr_vals.append(sisnr_val)
        si_snri_vals.append(sisnr_val - sisnr_mix)
        sdr_vals.append(sdr_val)
        sdri_vals.append(sdr_val - sdr_mix)

    result = {
        "si_snr"  : sum(si_snr_vals)  / n_src,
        "si_snri" : sum(si_snri_vals) / n_src,
        "sdr"     : sum(sdr_vals)     / n_src,
        "sdri"    : sum(sdri_vals)    / n_src,
    }

    # ── PESQ ─────────────────────────────────
    if use_pesq:
        try:
            from pesq import pesq as pesq_fn
            mode = "nb" if sr == 8000 else "wb"
            scores = []
            for i in range(n_src):
                ref_np = target_sources[i].cpu().numpy()
                est_np = est_reordered[i].cpu().numpy()
                ref_np = ref_np / (abs(ref_np).max() + 1e-8)
                est_np = est_np / (abs(est_np).max() + 1e-8)
                scores.append(pesq_fn(sr, ref_np, est_np, mode))
            result["pesq"] = sum(scores) / len(scores)
        except Exception:
            result["pesq"] = None

    # ── STOI ─────────────────────────────────
    if use_stoi:
        try:
            from pystoi import stoi as stoi_fn
            scores = []
            for i in range(n_src):
                ref_np = target_sources[i].cpu().numpy()
                est_np = est_reordered[i].cpu().numpy()
                ref_np = ref_np / (abs(ref_np).max() + 1e-8)
                est_np = est_np / (abs(est_np).max() + 1e-8)
                scores.append(stoi_fn(ref_np, est_np, sr, extended=False))
            result["stoi"] = sum(scores) / len(scores)
        except Exception:
            result["stoi"] = None

    return est_reordered, result


# ─────────────────────────────────────────────
#  Vérification dépendances optionnelles
# ─────────────────────────────────────────────
def check_optional_deps():
    use_pesq = use_stoi = False
    try:
        import pesq
        use_pesq = True
        print("  ✓ PESQ disponible")
    except ImportError:
        print("  ✗ PESQ non installé  →  pip install pesq")
    try:
        import pystoi
        use_stoi = True
        print("  ✓ STOI disponible")
    except ImportError:
        print("  ✗ STOI non installé  →  pip install pystoi")
    return use_pesq, use_stoi


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    tcfg = load_config(args.train_cfg)
    dcfg = load_config(args.data_cfg)

    mod = tcfg["model"]
    ds  = dcfg["dataset"]
    sp  = dcfg["splits"]
    sr  = ds["sample_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Config] Device     : {device}")
    print(f"[Config] Checkpoint : {args.ckpt}")

    print("\n[Dépendances]")
    use_pesq, use_stoi = check_optional_deps()

    # ── Test loader ──────────────────────────
    _, _, test_loader = get_dataloaders(
        root_dir         = ds["root_dir"],
        n_src            = ds["n_src"],
        sample_rate      = sr,
        segment_duration = ds["segment_duration"],
        batch_size       = 1,
        num_workers      = 0,
        train_ratio      = sp["train_ratio"],
        val_ratio        = sp["val_ratio"],
    )

    # ── Modèle ───────────────────────────────
    model = build_model(
        n_src         = ds["n_src"],
        sample_rate   = sr,
        n_filters     = mod["n_filters"],
        filter_length = mod["filter_length"],
        stride        = mod["stride"],
        n_blocks      = mod["n_blocks"],
        n_repeats     = mod["n_repeats"],
        bn_chan        = mod["bn_chan"],
        hid_chan       = mod["hid_chan"],
        skip_chan      = mod["skip_chan"],
        norm_type      = mod["norm_type"],
        mask_act       = mod["mask_act"],
        use_gradient_checkpointing = False,
    ).to(device)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint introuvable : {args.ckpt}")

    load_checkpoint(model, args.ckpt, device)
    model.eval()

    ckpt     = torch.load(args.ckpt, map_location="cpu")
    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", None)
    print(f"\n[Model] Epoch {epoch}", end="")
    if val_loss is not None:
        print(f"  |  Val SI-SNR : {-val_loss:.2f} dB")
    else:
        print()

    # ── Évaluation ───────────────────────────
    all_metrics = {"si_snr": [], "si_snri": [], "sdr": [], "sdri": []}
    if use_pesq: all_metrics["pesq"] = []
    if use_stoi: all_metrics["stoi"] = []

    # Header tableau
    header = f"{'Mix':<8} {'SI-SNR':>10} {'SI-SNRi':>10} {'SDR':>10} {'SDRi':>10}"
    if use_pesq: header += f" {'PESQ':>8}"
    if use_stoi: header += f" {'STOI':>8}"
    header += f" {'Qualité':>10}"
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    with torch.no_grad():
        for idx, (mixture, sources) in enumerate(test_loader):
            mixture = mixture.to(device)    # (1, T)
            sources = sources.to(device)    # (1, n_src, T)

            est     = model(mixture).squeeze(0)  # (n_src, T)
            sources = sources.squeeze(0)         # (n_src, T)
            mix_1d  = mixture.squeeze(0)         # (T,)

            _, metrics = best_permutation(
                est, sources, mix_1d, sr, use_pesq, use_stoi
            )

            for k in all_metrics:
                if metrics.get(k) is not None:
                    all_metrics[k].append(metrics[k])

            # Qualité basée sur SI-SNRi (gain réel)
            si_snri_val = metrics["si_snri"]
            if si_snri_val > 10:   qualite = "Excellent"
            elif si_snri_val > 5:  qualite = "Bon"
            elif si_snri_val > 0:  qualite = "Moyen"
            else:                  qualite = "Faible"

            row = (f"Mix {idx:<4} "
                   f"{metrics['si_snr']:>10.2f} "
                   f"{metrics['si_snri']:>10.2f} "
                   f"{metrics['sdr']:>10.2f} "
                   f"{metrics['sdri']:>10.2f}")
            if use_pesq:
                v = metrics.get("pesq")
                row += f" {v:>8.3f}" if v is not None else f" {'N/A':>8}"
            if use_stoi:
                v = metrics.get("stoi")
                row += f" {v:>8.3f}" if v is not None else f" {'N/A':>8}"
            row += f" {qualite:>10}"
            print(row)

            # Audio
            if args.save_audio:
                os.makedirs(args.out_dir, exist_ok=True)
                for i in range(est.shape[0]):
                    src = est[i].unsqueeze(0).cpu()
                    src = src / (src.abs().max() + 1e-8) * 0.9
                    path = os.path.join(args.out_dir, f"mix{idx}_source{i+1}.wav")
                    torchaudio.save(path, src, sr)

    # ── Résumé final ─────────────────────────
    def avg(lst): return sum(lst) / len(lst) if lst else float("nan")

    print(sep)
    row = (f"{'MOYENNE':<8} "
           f"{avg(all_metrics['si_snr']):>10.2f} "
           f"{avg(all_metrics['si_snri']):>10.2f} "
           f"{avg(all_metrics['sdr']):>10.2f} "
           f"{avg(all_metrics['sdri']):>10.2f}")
    if use_pesq and all_metrics.get("pesq"):
        row += f" {avg(all_metrics['pesq']):>8.3f}"
    if use_stoi and all_metrics.get("stoi"):
        row += f" {avg(all_metrics['stoi']):>8.3f}"
    print(row)
    print(sep)

    print(f"\n{'='*55}")
    print(f"  SI-SNR  moyen : {avg(all_metrics['si_snr']):.2f} dB")
    print(f"  SI-SNRi moyen : {avg(all_metrics['si_snri']):.2f} dB  ← gain vs mixture")
    print(f"  SDR     moyen : {avg(all_metrics['sdr']):.2f} dB")
    print(f"  SDRi    moyen : {avg(all_metrics['sdri']):.2f} dB  ← gain vs mixture")
    if use_pesq and all_metrics.get("pesq"):
        print(f"  PESQ    moyen : {avg(all_metrics['pesq']):.3f}      (1–4.5)")
    if use_stoi and all_metrics.get("stoi"):
        print(f"  STOI    moyen : {avg(all_metrics['stoi']):.3f}      (0–1)")
    print(f"{'='*55}")

    print(f"\n  Interprétation (SI-SNRi) :")
    print(f"    > 10 dB  → Excellente séparation")
    print(f"    5-10 dB  → Bonne séparation")
    print(f"    0-5  dB  → Séparation partielle")
    print(f"    < 0  dB  → Entraînement insuffisant")

    if args.save_audio:
        print(f"\n  Audio sauvegardé dans : {args.out_dir}/")


if __name__ == "__main__":
    main()