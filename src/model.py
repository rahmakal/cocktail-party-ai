import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from asteroid.models import ConvTasNet


def build_model(n_src=5, sample_rate=8000,
                n_filters=512, filter_length=16,
                stride=8, n_blocks=8, n_repeats=3,
                bn_chan=128, hid_chan=512, skip_chan=128,
                norm_type="gLN", mask_act="relu",
                use_gradient_checkpointing=False):

    model = ConvTasNet(
        n_src=n_src, sample_rate=sample_rate,
        n_filters=n_filters, filter_length=filter_length,
        stride=stride, n_blocks=n_blocks, n_repeats=n_repeats,
        bn_chan=bn_chan, hid_chan=hid_chan, skip_chan=skip_chan,
        norm_type=norm_type, mask_act=mask_act,
    )

    if use_gradient_checkpointing:
        _apply_gradient_checkpointing(model)
        print("[Model] Gradient checkpointing : ACTIVÉ  (-50% VRAM, +30% temps)")
    else:
        print("[Model] Gradient checkpointing : désactivé")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Conv-TasNet  |  Paramètres entraînables : {n_params:,}")
    return model


def _apply_gradient_checkpointing(model):
    if not hasattr(model, "masker") or not hasattr(model.masker, "TCN"):
        print("[Warning] masker.TCN introuvable — gradient checkpointing non appliqué.")
        return

    original_blocks = list(model.masker.TCN.named_children())
    if not original_blocks:
        return

    for name, block in original_blocks:
        _wrap_block(model.masker.TCN, name, block)

    print(f"[Model] {len(original_blocks)} blocs TCN checkpointés.")


def _wrap_block(parent, name, block):
    class CheckpointedBlock(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            if not x.requires_grad:
                x = x.requires_grad_(True)
            return checkpoint(self.inner, x, use_reentrant=False)

    setattr(parent, name, CheckpointedBlock(block))


def load_checkpoint(model, path, device="cpu"):
    """
    Load checkpoint safely.
    Automatically handles the .inner. key mismatch caused by
    gradient checkpointing wrapper (CheckpointedBlock).
    """
    ckpt  = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    model_keys = set(model.state_dict().keys())

    # Try loading as-is first
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_set    = set(missing)
    unexpected_set = set(unexpected)

    # Case 1 : checkpoint has .inner. but model doesn't → strip .inner.
    if any(".inner." in k for k in unexpected_set) and \
       any(".inner." not in k for k in missing_set):
        state = {k.replace(".inner.", "."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        print("[Model] '.inner.' stripped from checkpoint keys (GC ON → OFF)")

    # Case 2 : model has .inner. but checkpoint doesn't → add .inner.
    elif any(".inner." in k for k in missing_set) and \
         any(".inner." not in k for k in unexpected_set):
        new_state = {}
        for k, v in state.items():
            if "masker.TCN." in k and ".inner." not in k:
                parts = k.split(".")
                parts.insert(3, "inner")
                k = ".".join(parts)
            new_state[k] = v
        model.load_state_dict(new_state, strict=True)
        print("[Model] '.inner.' added to checkpoint keys (GC OFF → ON)")

    # Case 3 : loaded fine on first try
    elif len(missing) == 0 and len(unexpected) == 0:
        print("[Model] Checkpoint chargé sans modification")

    else:
        raise RuntimeError(
            f"Cannot load checkpoint — unresolvable key mismatch:\n"
            f"  Missing   : {list(missing)[:3]}...\n"
            f"  Unexpected: {list(unexpected)[:3]}..."
        )

    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("best_val_loss", "?")
    print(f"[Model] Checkpoint chargé depuis {path}  "
          f"(epoch {epoch}, val loss {val_loss})")
    return model