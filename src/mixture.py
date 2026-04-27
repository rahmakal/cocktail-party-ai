from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import soundfile as sf
import torch
import torchaudio

# ── Config ────────────────────────────────────────────────────────────────────
ARABIC_INPUT_DIR = Path("data/raw_sources_cv")   # contains spk_*/ folders
OUTPUT_ROOT      = Path("data/arabic_data")

TARGET_SR      = 8000
TARGET_SECONDS = 5
RANDOM_SEED    = 42

SPLITS = {
    "train": 10000,
    "val":   500,
    "test":  500,
}
# ──────────────────────────────────────────────────────────────────────────────


def load_audio(audio_path: Path) -> Tuple[torch.Tensor, int]:
    try:
        waveform, sr = torchaudio.load(str(audio_path))
        return waveform, sr
    except Exception:
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        return torch.from_numpy(audio).transpose(0, 1), sr


def save_audio(audio_path: Path, waveform: torch.Tensor, sr: int) -> None:
    try:
        torchaudio.save(str(audio_path), waveform.cpu(), sr)
    except Exception:
        sf.write(str(audio_path), waveform.detach().cpu().transpose(0, 1).numpy(), sr)


def to_mono_resampled(
    waveform: torch.Tensor,
    sr: int,
    target_sr: int,
    cache: Dict[int, torchaudio.transforms.Resample],
) -> torch.Tensor:
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        if sr not in cache:
            cache[sr] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = cache[sr](waveform)
    return waveform


def fit_to_length(waveform: torch.Tensor, target_len: int, rng: random.Random) -> torch.Tensor:
    n = waveform.size(1)
    if n == target_len:
        return waveform
    if n < target_len:
        repeats = math.ceil(target_len / n)
        return waveform.repeat(1, repeats)[:, :target_len]
    start = rng.randint(0, n - target_len)
    return waveform[:, start: start + target_len]


def safe_normalize(*tensors: torch.Tensor, peak: float = 0.99) -> List[torch.Tensor]:
    max_val = max(t.abs().max().item() for t in tensors)
    if max_val == 0:
        return list(tensors)
    scale = min(1.0, peak / max_val)
    return [t * scale for t in tensors]


def ensure_dirs(root: Path, split: str) -> Dict[str, Path]:
    dirs = {k: root / split / k for k in ("mix", "s1", "s2", "s3")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def collect_speaker_files(input_dir: Path) -> Dict[str, List[Path]]:
    """Returns {speaker_id: [list of .mp3 paths]}"""
    speakers: Dict[str, List[Path]] = {}
    for spk_dir in sorted(input_dir.iterdir()):
        if not spk_dir.is_dir() or not spk_dir.name.startswith("spk_"):
            continue
        files = sorted(spk_dir.glob("*.mp3"))
        if files:
            speakers[spk_dir.name] = files
    return speakers


def generate_split(
    split: str,
    n: int,
    speaker_files: Dict[str, List[Path]],
    output_root: Path,
    rng: random.Random,
    resamplers: Dict[int, torchaudio.transforms.Resample],
    target_len: int,
) -> None:
    dirs = ensure_dirs(output_root, split)
    speaker_ids = list(speaker_files.keys())

    if len(speaker_ids) < 3:
        raise RuntimeError(f"Need at least 3 speakers, found {len(speaker_ids)}")

    metadata_rows: List[List[str]] = []
    skipped = 0

    for i in range(n):
        sample_id = f"sample_{i:04d}"

        # Pick 3 different speakers, one file each
        chosen_speakers = rng.sample(speaker_ids, 3)
        source_paths = [rng.choice(speaker_files[spk]) for spk in chosen_speakers]

        try:
            sources: List[torch.Tensor] = []
            for path in source_paths:
                waveform, sr = load_audio(path)
                waveform = to_mono_resampled(waveform, sr, TARGET_SR, resamplers)
                waveform = fit_to_length(waveform, target_len, rng)

                # Skip NaN/Inf files
                if not torch.isfinite(waveform).all():
                    raise ValueError(f"Non-finite audio in {path}")

                sources.append(waveform)

            s1, s2, s3 = sources
            mix = s1 + s2 + s3
            mix, s1, s2, s3 = safe_normalize(mix, s1, s2, s3)

            mix_path = dirs["mix"] / f"{sample_id}.wav"
            s1_path  = dirs["s1"]  / f"{sample_id}.wav"
            s2_path  = dirs["s2"]  / f"{sample_id}.wav"
            s3_path  = dirs["s3"]  / f"{sample_id}.wav"

            save_audio(mix_path, mix, TARGET_SR)
            save_audio(s1_path,  s1,  TARGET_SR)
            save_audio(s2_path,  s2,  TARGET_SR)
            save_audio(s3_path,  s3,  TARGET_SR)

            metadata_rows.append([
                sample_id,
                f"mix/{sample_id}.wav",
                f"s1/{sample_id}.wav",
                f"s2/{sample_id}.wav",
                f"s3/{sample_id}.wav",
                chosen_speakers[0], source_paths[0].name,
                chosen_speakers[1], source_paths[1].name,
                chosen_speakers[2], source_paths[2].name,
            ])

        except Exception as e:
            print(f"  [SKIP] {split}/{sample_id}: {e}")
            skipped += 1
            continue

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  [{split}] {i + 1}/{n} done  (skipped so far: {skipped})")

    # Write metadata CSV
    csv_path = output_root / split / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "mix_path", "s1_path", "s2_path", "s3_path",
            "spk1", "file1",
            "spk2", "file2",
            "spk3", "file3",
        ])
        writer.writerows(metadata_rows)

    print(f"  [{split}] saved {len(metadata_rows)} samples  (skipped {skipped})  → {csv_path}\n")


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    target_len = TARGET_SR * TARGET_SECONDS
    resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    print(f"Scanning speakers in {ARABIC_INPUT_DIR} …")
    speaker_files = collect_speaker_files(ARABIC_INPUT_DIR)
    print(f"Found {len(speaker_files)} speakers\n")

    for split, n in SPLITS.items():
        print(f"Generating {n} mixtures for [{split}] …")
        generate_split(split, n, speaker_files, OUTPUT_ROOT, rng, resamplers, target_len)

    print("All done!")


if __name__ == "__main__":
    main()