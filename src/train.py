"""
train.py — Conv-TasNet
  - Mixed precision (fp16)
  - Gradient accumulation
  - Warmup + Cosine Annealing
  - Early stopping
  - Loss curve saved as PNG after training
  - Suivi du temps (epoch / total estimé / ETA)

Usage :
    python main.py train
    python main.py train --resume checkpoints_3src_2/best.ckpt
"""

import os
import time
import argparse
import math
import json
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from collections import deque

from src.dataset import get_dataloaders
from src.model   import build_model
from src.loss    import SISNRLoss


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_cfg", type=str, default="configs/train.yaml")
    p.add_argument("--data_cfg",  type=str, default="configs/data.yaml")
    p.add_argument("--resume",    type=str, default=None)
    p.add_argument("--override_lr", type=float, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────
#  Formatage du temps
# ─────────────────────────────────────────────
def fmt_duration(seconds):
    """Convertit des secondes en format lisible : 1j 4h 32m 10s"""
    seconds = int(seconds)
    days    = seconds // 86400
    hours   = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs    = seconds % 60

    if days > 0:
        return f"{days}j {hours:02d}h {minutes:02d}m {secs:02d}s"
    elif hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs}s"


def fmt_timestamp(ts):
    """Convertit un timestamp Unix en heure locale lisible"""
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%d/%m %H:%M:%S")


# ─────────────────────────────────────────────
#  LR scheduler : Warmup + Cosine Annealing
# ─────────────────────────────────────────────
def get_lr(epoch, total_epochs, lr_max, lr_min, warmup_epochs):
    if epoch <= warmup_epochs:
        return lr_max * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min + (lr_max - lr_min) * cosine


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ─────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion,
                device, grad_clip, accum_steps, scaler):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (mixtures, sources) in enumerate(loader):
        mixtures = mixtures.to(device)
        sources  = sources.to(device)

        with autocast(enabled=(device.type == "cuda")):
            est  = model(mixtures)
            loss = criterion(est, sources) / accum_steps

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [LOSS] ⚠ Batch {batch_idx} SKIPPED — loss NaN/Inf")
            optimizer.zero_grad()
            continue
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)

            # ── Logging gradients ────────────────
            total_norm = 0.0
            max_norm   = 0.0
            max_name   = ""
            nan_params = []

            for name, param in model.named_parameters():
                if param.grad is not None:
                    norm = param.grad.data.norm(2).item()
                    total_norm += norm ** 2
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        nan_params.append(name)
                    if norm > max_norm:
                        max_norm = norm
                        max_name = name

            total_norm = total_norm ** 0.5

            if total_norm > 10.0 or nan_params:
                print(f"\n  [GRAD] batch {batch_idx} — norm totale : {total_norm:.2f}")
                print(f"  [GRAD] grad max : {max_norm:.4f}  ({max_name})")
                if nan_params:
                    print(f"  [GRAD] ⚠ NaN/Inf dans : {nan_params}")

            # ── NOUVEAU : skip si NaN ────────────
            if nan_params or math.isnan(total_norm) or math.isinf(total_norm):
                print(f"  [GRAD] ⚠ Batch {batch_idx} SKIPPED — grads corrompus")
                optimizer.zero_grad()
                scaler.update()  # important : mettre à jour le scaler quand même
                continue
            # ────────────────────────────────────

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / len(loader)


# ─────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for mixtures, sources in loader:
        mixtures = mixtures.to(device)
        sources  = sources.to(device)
        with autocast(enabled=(device.type == "cuda")):
            total_loss += criterion(model(mixtures), sources).item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────
#  Checkpoint
# ─────────────────────────────────────────────
def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  ✓ Checkpoint : {path}")


# ─────────────────────────────────────────────
#  Loss curve plot
# ─────────────────────────────────────────────
def save_loss_plot(history, out_dir, stopped_epoch=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [Plot] matplotlib not installed — skipping plot.")
        return

    os.makedirs(out_dir, exist_ok=True)

    epochs     = history["epochs"]
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    lr_history = history["lr"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Conv-TasNet — Training curves", fontsize=14, fontweight="bold")

    ax1.plot(epochs, train_loss, label="Train loss",
             color="#1D9E75", linewidth=2)
    ax1.plot(epochs, val_loss,   label="Val loss",
             color="#D85A30", linewidth=2, linestyle="--")

    best_val  = min(val_loss)
    best_ep   = epochs[val_loss.index(best_val)]
    ax1.scatter([best_ep], [best_val], color="#D85A30", s=80, zorder=5,
                label=f"Best val: {best_val:.3f} @ epoch {best_ep}")

    if stopped_epoch:
        ax1.axvline(x=stopped_epoch, color="gray", linestyle=":",
                    linewidth=1.5, label=f"Early stop @ epoch {stopped_epoch}")

    for i, (e, tl, vl) in enumerate(zip(epochs, train_loss, val_loss)):
        if vl - tl > 0.5:
            ax1.axvspan(e - 0.5, e + 0.5, alpha=0.08, color="red")

    ax1.set_ylabel("Loss  (lower = better)", fontsize=11)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Train vs Val loss  —  overfitting = val rises while train falls",
                  fontsize=10, color="gray")

    ax1_r = ax1.twinx()
    ax1_r.set_ylabel("SI-SNR (dB)  ↑", fontsize=10, color="gray")
    ax1_r.set_ylim(-ax1.get_ylim()[1], -ax1.get_ylim()[0])
    ax1_r.tick_params(axis="y", colors="gray")

    ax2.plot(epochs, lr_history, color="#378ADD", linewidth=1.5)
    ax2.set_ylabel("Learning rate", fontsize=10)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("LR schedule  (warmup → cosine annealing)", fontsize=10, color="gray")

    plt.tight_layout()

    plot_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Loss curve  : {plot_path}")

    json_path = os.path.join(out_dir, "history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ History JSON: {json_path}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    tcfg = load_config(args.train_cfg)
    dcfg = load_config(args.data_cfg)

    tr  = tcfg["training"]
    mod = tcfg["model"]
    sch = tcfg["scheduler"]
    ds  = dcfg["dataset"]
    sp  = dcfg["splits"]

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accum_steps = tr.get("accumulation_steps", 1)
    eff_batch   = tr["batch_size"] * accum_steps
    lr_max      = tr["learning_rate"]
    lr_min      = sch.get("min_lr", 1e-6)
    warmup_ep   = sch.get("warmup_epochs", 10)
    total_ep    = tr["epochs"]
    patience    = tr.get("early_stopping", 0)
    log_dir     = tcfg["paths"]["log_dir"]
    ckpt_dir    = tcfg["paths"]["checkpoint_dir"]
    save_every  = tr["save_every"]

    print(f"\n[Config] Device         : {device}")
    if device.type == "cuda":
        print(f"[Config] GPU            : {torch.cuda.get_device_name(0)}")
        print(f"[Config] VRAM           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"[Config] Batch effectif : {tr['batch_size']} × {accum_steps} = {eff_batch}")
    print(f"[Config] LR             : 0 → {lr_max} (warmup {warmup_ep} ep) → {lr_min} (cosine)")
    print(f"[Config] Max epochs     : {total_ep}")
    print(f"[Config] Early stopping : {'disabled' if patience == 0 else f'patience = {patience} epochs'}\n")
    print(f"[Config] Grad clip      : {tr['grad_clip']}")

    # ── DataLoaders ──────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        root_dir         = ds["root_dir"],
        n_src            = ds["n_src"],
        sample_rate      = ds["sample_rate"],
        segment_duration = ds["segment_duration"],
        batch_size       = tr["batch_size"],
        num_workers      = tr["num_workers"],
        train_ratio      = sp["train_ratio"],
        val_ratio        = sp["val_ratio"],
    )

    # ── Model ────────────────────────────────
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
        use_gradient_checkpointing = mod.get("gradient_checkpointing", False),
    ).to(device)

    criterion = SISNRLoss()
    optimizer = Adam(model.parameters(), lr=lr_max,
                     weight_decay=tcfg["optimizer"]["weight_decay"])
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    start_epoch   = 1
    best_val_loss = float("inf")
    no_improve    = 0
    stopped_epoch = None

    # ── Resume ───────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        no_improve    = 0

        if args.override_lr is not None:
            lr_max = args.override_lr   # ← le schedule repart de cette LR
            set_lr(optimizer, lr_max)
            print(f"[Resume] LR overridée à {lr_max:.6f}")
        else:
            resumed_lr = get_lr(start_epoch, total_ep, lr_max, lr_min, warmup_ep)
            set_lr(optimizer, resumed_lr)

    # ── History ──────────────────────────────
    history = {"epochs": [], "train_loss": [], "val_loss": [], "lr": [],
               "epoch_duration": []}

    history_path = os.path.join(log_dir, "history.json")
    if args.resume and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        if "epoch_duration" not in history:
            history["epoch_duration"] = []
        print(f"[History] Loaded {len(history['epochs'])} previous epochs\n")

    # ── Suivi du temps ───────────────────────
    # Fenêtre glissante sur les 5 dernières epochs pour ETA précis
    epoch_times   = deque(maxlen=5)
    train_start   = time.time()
    epochs_done   = len(history["epochs"])   # epochs déjà faites (resume)
    epochs_remaining = total_ep - start_epoch + 1

    print(f"[Temps]  Début de l'entraînement : {fmt_timestamp(train_start)}")
    if epochs_done > 0 and history["epoch_duration"]:
        avg_prev = sum(history["epoch_duration"]) / len(history["epoch_duration"])
        eta_s    = avg_prev * epochs_remaining
        print(f"[Temps]  ETA estimée (historique) : {fmt_duration(eta_s)}")
    print()

    # ── Training loop ────────────────────────
    for epoch in range(start_epoch, total_ep + 1):

        if args.override_lr is not None and args.resume:
            # Schedule repart de lr_max (override) à partir de start_epoch
            relative_epoch = epoch - start_epoch + 1
            relative_total = total_ep - start_epoch + 1
            current_lr = get_lr(relative_epoch, relative_total, lr_max, lr_min, warmup_epochs=0)
        else:
            current_lr = get_lr(epoch, total_ep, lr_max, lr_min, warmup_ep)
        set_lr(optimizer, current_lr)

        if epoch <= warmup_ep:
            phase = f"warmup  ({epoch}/{warmup_ep})"
        else:
            pct   = (epoch - warmup_ep) / (total_ep - warmup_ep) * 100
            phase = f"cosine  {pct:.0f}%"

        print(f"\n{'='*58}")
        print(f"Epoch {epoch}/{total_ep}  |  lr = {current_lr:.6f}  [{phase}]")
        print(f"{'='*58}")

        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 device, tr["grad_clip"], accum_steps, scaler)
        val_loss   = validate(model, val_loader, criterion, device)

        epoch_end      = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)

        # ETA basée sur la moyenne glissante des 5 dernières epochs
        avg_epoch_time   = sum(epoch_times) / len(epoch_times)
        epochs_left      = total_ep - epoch
        eta_seconds      = avg_epoch_time * epochs_left
        eta_finish       = epoch_end + eta_seconds

        # Temps total écoulé depuis le début de cette session
        elapsed_session  = epoch_end - train_start

        # VRAM + résultats
        if device.type == "cuda":
            vram = torch.cuda.max_memory_allocated(device) / 1e9
            torch.cuda.reset_peak_memory_stats(device)
            print(f"\n  Train loss : {train_loss:.4f}  (SI-SNR = {-train_loss:.2f} dB)")
            print(f"  Val   loss : {val_loss:.4f}  (SI-SNR = {-val_loss:.2f} dB)"
                  f"  [best: {best_val_loss:.4f}]")
            print(f"  VRAM max   : {vram:.2f} GB")
        else:
            print(f"\n  Train loss : {train_loss:.4f}  (SI-SNR = {-train_loss:.2f} dB)")
            print(f"  Val   loss : {val_loss:.4f}  (SI-SNR = {-val_loss:.2f} dB)"
                  f"  [best: {best_val_loss:.4f}]")

        # ── Affichage du temps ───────────────
        print(f"\n  ⏱  Epoch        : {fmt_duration(epoch_duration)}")
        print(f"  ⏱  Session      : {fmt_duration(elapsed_session)}")
        print(f"  ⏱  Moy/epoch    : {fmt_duration(avg_epoch_time)}  "
              f"(sur {len(epoch_times)} dernière{'s' if len(epoch_times)>1 else ''})")
        if epochs_left > 0:
            print(f"  ⏱  ETA          : {fmt_duration(eta_seconds)}  "
                  f"(fin estimée : {fmt_timestamp(eta_finish)})")
        else:
            print(f"  ⏱  ETA          : dernière epoch !")

        # Record history
        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["lr"].append(round(current_lr, 7))
        history["epoch_duration"].append(round(epoch_duration, 2))
        os.makedirs(log_dir, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # ── Early stopping ───────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            ckpt_state = {
                "epoch"               : epoch,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict"   : scaler.state_dict(),
                "best_val_loss"       : best_val_loss,
                "no_improve"          : no_improve,
            }
            save_checkpoint(ckpt_state, os.path.join(ckpt_dir, "best.ckpt"))
        else:
            no_improve += 1
            if patience > 0:
                print(f"  No improve : {no_improve}/{patience}")
                if no_improve >= patience:
                    stopped_epoch = epoch
                    print(f"\n[Early Stop] No improvement for {patience} epochs.")
                    print(f"[Early Stop] Best val loss : {best_val_loss:.4f}"
                          f"  (SI-SNR = {-best_val_loss:.2f} dB)")
                    break

        # Periodic checkpoint
        if epoch % save_every == 0:
            save_checkpoint(
                {
                    "epoch"               : epoch,
                    "model_state_dict"    : model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict"   : scaler.state_dict(),
                    "best_val_loss"       : best_val_loss,
                    "no_improve"          : no_improve,
                },
                os.path.join(ckpt_dir, f"epoch_{epoch}.ckpt"),
            )

    # ── Résumé final ─────────────────────────
    total_duration = time.time() - train_start
    print(f"\n{'='*58}")
    print(f"  Training finished")
    print(f"  Best val loss  : {best_val_loss:.4f}  "
          f"(SI-SNR = {-best_val_loss:.2f} dB)")
    if stopped_epoch:
        print(f"  Stopped at epoch {stopped_epoch} (early stopping)")
    else:
        print(f"  Completed all {total_ep} epochs")
    print(f"  Durée totale   : {fmt_duration(total_duration)}")
    if history["epoch_duration"]:
        avg = sum(history["epoch_duration"]) / len(history["epoch_duration"])
        print(f"  Moy par epoch  : {fmt_duration(avg)}")
    print(f"{'='*58}")

    # ── Save loss curve ──────────────────────
    print(f"\n[Plot] Saving loss curve...")

    save_loss_plot(history, log_dir, stopped_epoch)


if __name__ == "__main__":
    main()
