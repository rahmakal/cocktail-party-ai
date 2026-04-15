"""
separate.py — Séparation de sources avec le modèle entraîné
Usage :
    python separate.py --mix data/mixture/mix_0/mixture.wav
    python separate.py --mix data/mixture/mix_0/mixture.wav --ckpt checkpoints/best.ckpt
    python separate.py --mix mon_audio.wav --out_dir outputs/separated_audio
"""

import os
import argparse
import torch
import torchaudio

from src.model import build_model, load_checkpoint
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Séparation de sources Conv-TasNet")
    p.add_argument("--mix",       type=str, required=True,
                   help="Chemin vers le fichier mixture.wav à séparer")
    p.add_argument("--ckpt",      type=str, default="checkpoints/best.ckpt",
                   help="Checkpoint du modèle entraîné")
    p.add_argument("--out_dir",   type=str, default="outputs/separated_audio",
                   help="Dossier de sortie pour les sources séparées")
    p.add_argument("--train_cfg", type=str, default="configs/train.yaml")
    p.add_argument("--data_cfg",  type=str, default="configs/data.yaml")
    return p.parse_args()


def separate(mix_path, model, sample_rate, device, out_dir):
    """Charge un mixture.wav, sépare les sources, sauvegarde les .wav."""

    # ── Charger le fichier audio ─────────────
    mixture, sr = torchaudio.load(mix_path)

    if sr != sample_rate:
        print(f"  Resample {sr} Hz → {sample_rate} Hz")
        mixture = torchaudio.functional.resample(mixture, sr, sample_rate)

    # Mono (1, T)
    if mixture.shape[0] > 1:
        mixture = mixture.mean(dim=0, keepdim=True)

    print(f"  Durée   : {mixture.shape[-1] / sample_rate:.2f}s  "
          f"({mixture.shape[-1]} samples)")

    # ── Inférence ────────────────────────────
    mixture = mixture.to(device)           # (1, T)
    with torch.no_grad():
        # Le modèle attend (B, T) → unsqueeze batch dim
        est_sources = model(mixture.unsqueeze(0))   # (1, n_src, T)
        est_sources = est_sources.squeeze(0)         # (n_src, T)

    # ── Sauvegarder les sources séparées ─────
    os.makedirs(out_dir, exist_ok=True)
    mix_name = os.path.splitext(os.path.basename(mix_path))[0]

    for i, src in enumerate(est_sources):
        src_cpu = src.unsqueeze(0).cpu()   # (1, T)

        # Normaliser pour éviter la saturation
        max_val = src_cpu.abs().max()
        if max_val > 0:
            src_cpu = src_cpu / max_val * 0.9

        out_path = os.path.join(out_dir, f"{mix_name}_source_{i+1}.wav")
        torchaudio.save(out_path, src_cpu, sample_rate)
        print(f"  ✓ Source {i+1} sauvegardée : {out_path}")

    return est_sources


def main():
    args  = parse_args()
    tcfg  = load_config(args.train_cfg)
    dcfg  = load_config(args.data_cfg)

    mod = tcfg["model"]
    ds  = dcfg["dataset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Config] Device      : {device}")
    print(f"[Config] Checkpoint  : {args.ckpt}")
    print(f"[Config] Fichier mix : {args.mix}\n")

    # ── Charger le modèle ────────────────────
    model = build_model(
        n_src         = ds["n_src"],
        sample_rate   = ds["sample_rate"],
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
        use_gradient_checkpointing = False,   # pas besoin en inférence
    )

    # ── Charger les poids entraînés ──────────
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f"Checkpoint introuvable : {args.ckpt}\n"
            f"Lancez d'abord : python train.py"
        )

    load_checkpoint(model, args.ckpt, device)
    model.to(device)
    model.eval()

    ckpt  = torch.load(args.ckpt, map_location="cpu")
    epoch = ckpt.get("epoch", "?")
    val   = ckpt.get("best_val_loss", None)
    if val is not None:
        print(f"[Model] Checkpoint chargé  (epoch {epoch}, val loss {val:.4f})\n")
    else:
        print(f"[Model] Checkpoint chargé  (epoch {epoch})\n")

    # ── Séparation ───────────────────────────
    separate(
        mix_path    = args.mix,
        model       = model,
        sample_rate = ds["sample_rate"],
        device      = device,
        out_dir     = args.out_dir,
    )

    print(f"\n[Done] Sources séparées dans : {args.out_dir}/")


if __name__ == "__main__":
    main()