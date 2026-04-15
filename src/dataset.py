import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset


class MixtureDataset(Dataset):
    def __init__(self, root_dir, split=None, n_src=5,
                 sample_rate=16000, segment_duration=5.0):
        self.root_dir    = root_dir
        self.n_src       = n_src
        self.split       = split
        self.sample_rate = sample_rate
        self.segment_len = int(segment_duration * sample_rate) \
                           if segment_duration > 0 else None

        split_path = os.path.join(root_dir, "split.json")
        if os.path.exists(split_path) and split is not None:
            with open(split_path) as f:
                splits = json.load(f)
            if split not in splits:
                raise ValueError(f"Split '{split}' not in split.json "
                                 f"(available: {list(splits.keys())})")
            self.mix_dirs = [
                os.path.join(root_dir, d) for d in splits[split]
            ]
            print(f"[Dataset] split.json found — '{split}': "
                  f"{len(self.mix_dirs)} mixtures")
        else:
            self.mix_dirs = sorted([
                os.path.join(root_dir, d)
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
                and d.startswith("mix_")
            ])
            if split is not None:
                print(f"[Dataset] No split.json found — using all "
                      f"{len(self.mix_dirs)} mixtures")

        if len(self.mix_dirs) == 0:
            raise RuntimeError(f"No mix_N folders found in {root_dir}")

    def __len__(self):
        return len(self.mix_dirs)

    def _safe_load(self, path):
        """Charge un wav → tensor (1, T) normalisé, sans NaN/Inf."""
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = wav.abs().max()
        if max_val > 0:
            wav = wav / max_val
        return wav  # (1, T)

    def __getitem__(self, idx, _depth=0):
        # Protection contre la récursion infinie
        if _depth >= 10:
            segment_len = self.segment_len or self.sample_rate * 4
            dummy = torch.zeros(segment_len)
            return dummy, torch.zeros(self.n_src, segment_len)

        mix_dir = self.mix_dirs[idx]
        try:
            mixture = self._safe_load(os.path.join(mix_dir, "mixture.wav"))

            sources = []
            for i in range(1, self.n_src + 1):
                src = self._safe_load(os.path.join(mix_dir, f"source_{i}.wav"))
                sources.append(src)

            sources = torch.cat(sources, dim=0)   # (n_src, T)

            if self.segment_len is not None:
                T = mixture.shape[-1]
                if T > self.segment_len:
                    if self.split in ("val", "test"):
                        start = (T - self.segment_len) // 2
                    else:
                        start = torch.randint(0, T - self.segment_len, (1,)).item()
                    mixture = mixture[:, start: start + self.segment_len]
                    sources = sources[:, start: start + self.segment_len]
                else:
                    pad     = self.segment_len - T
                    mixture = torch.nn.functional.pad(mixture, (0, pad))
                    sources = torch.nn.functional.pad(sources, (0, pad))

            return mixture.squeeze(0), sources   # (T,), (n_src, T)

        except Exception as e:
            print(f"[WARNING] Corrompu : {mix_dir} ({e})")
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx, _depth=_depth + 1)


def collate_fn(batch):
    mixtures, sources = zip(*batch)
    max_len = max(m.shape[-1] for m in mixtures)
    mixtures_padded = torch.stack([
        torch.nn.functional.pad(m, (0, max_len - m.shape[-1]))
        for m in mixtures
    ])
    sources_padded = torch.stack([
        torch.nn.functional.pad(s, (0, max_len - s.shape[-1]))
        for s in sources
    ])
    return mixtures_padded, sources_padded


def get_dataloaders(root_dir, n_src=5, sample_rate=16000,
                    segment_duration=5.0, batch_size=4,
                    num_workers=2, train_ratio=0.8, val_ratio=0.1):
    split_path = os.path.join(root_dir, "split.json")

    if os.path.exists(split_path):
        train_set = MixtureDataset(root_dir, "train", n_src,
                                   sample_rate, segment_duration)
        val_set   = MixtureDataset(root_dir, "val",   n_src,
                                   sample_rate, segment_duration)
        test_set  = MixtureDataset(root_dir, "test",  n_src,
                                   sample_rate, segment_duration)
    else:
        print("[Dataset] No split.json — using auto split")
        full_set = MixtureDataset(root_dir, None, n_src,
                                  sample_rate, segment_duration)
        train_set, val_set, test_set = full_set.get_auto_splits(
            train_ratio, val_ratio
        )

    print(f"[Splits] Train: {len(train_set)} | "
          f"Val: {len(val_set)} | Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=True,
                              worker_init_fn=lambda id: torch.manual_seed(42 + id))
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=1,
                              shuffle=False, num_workers=num_workers,
                              collate_fn=collate_fn)

    return train_loader, val_loader, test_loader