from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict, List

import soundfile as sf
import torch
import torchaudio


# Folder containing your 3 Arabic mp3 files
ARABIC_INPUT_DIR = Path("arabic_raw")

# Output folder with the same structure as before
OUTPUT_ROOT = Path("arabic_data")

TARGET_SR = 8000
TARGET_SECONDS = 5
RANDOM_SEED = 42

# Put your files here in the order you want them mapped to s1, s2, s3
ARABIC_FILES = [
    "common_voice_ar_19058307.mp3",
    "common_voice_ar_24079022.mp3",
    "common_voice_ar_24175744.mp3",
]


def build_resampler_cache() -> Dict[int, torchaudio.transforms.Resample]:
    return {}


def load_audio(audio_path: Path) -> tuple[torch.Tensor, int]:
    """
    Prefer torchaudio.load, fall back to soundfile if codec support is limited.
    """
    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        return waveform, sample_rate
    except Exception:
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio).transpose(0, 1)
        return waveform, sample_rate


def save_audio(audio_path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """
    Prefer torchaudio.save, fall back to soundfile if needed.
    """
    try:
        torchaudio.save(str(audio_path), waveform.cpu(), sample_rate)
    except Exception:
        audio = waveform.detach().cpu().transpose(0, 1).numpy()
        sf.write(str(audio_path), audio, sample_rate)


def to_mono_resampled(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sr: int,
    resamplers: Dict[int, torchaudio.transforms.Resample],
) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, time], got {tuple(waveform.shape)}")

    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sample_rate != target_sr:
        if sample_rate not in resamplers:
            resamplers[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sr,
            )
        waveform = resamplers[sample_rate](waveform)

    return waveform


def fit_to_length(waveform: torch.Tensor, target_num_samples: int, rng: random.Random) -> torch.Tensor:
    current_num_samples = waveform.size(1)

    if current_num_samples == target_num_samples:
        return waveform

    if current_num_samples < target_num_samples:
        repeats = math.ceil(target_num_samples / current_num_samples)
        return waveform.repeat(1, repeats)[:, :target_num_samples]

    max_start = current_num_samples - target_num_samples
    start = rng.randint(0, max_start)
    end = start + target_num_samples
    return waveform[:, start:end]


def safe_normalize(*tensors: torch.Tensor, peak_limit: float = 0.99) -> List[torch.Tensor]:
    peak = max(tensor.abs().max().item() for tensor in tensors)
    if peak == 0:
        return list(tensors)

    scale = min(1.0, peak_limit / peak)
    return [tensor * scale for tensor in tensors]


def ensure_output_dirs(output_root: Path) -> Dict[str, Path]:
    paths = {
        "root": output_root,
        "mix": output_root / "mix",
        "s1": output_root / "s1",
        "s2": output_root / "s2",
        "s3": output_root / "s3",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    target_num_samples = TARGET_SR * TARGET_SECONDS
    output_dirs = ensure_output_dirs(OUTPUT_ROOT)
    resamplers = build_resampler_cache()

    input_paths = [ARABIC_INPUT_DIR / filename for filename in ARABIC_FILES]

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    processed_sources: List[torch.Tensor] = []
    for source_path in input_paths:
        waveform, sample_rate = load_audio(source_path)
        waveform = to_mono_resampled(waveform, sample_rate, TARGET_SR, resamplers)
        waveform = fit_to_length(waveform, target_num_samples, rng)
        processed_sources.append(waveform)

    s1, s2, s3 = processed_sources
    mix = s1 + s2 + s3
    mix, s1, s2, s3 = safe_normalize(mix, s1, s2, s3)

    sample_id = "sample_0000"
    mix_path = output_dirs["mix"] / f"{sample_id}.wav"
    s1_path = output_dirs["s1"] / f"{sample_id}.wav"
    s2_path = output_dirs["s2"] / f"{sample_id}.wav"
    s3_path = output_dirs["s3"] / f"{sample_id}.wav"

    save_audio(mix_path, mix, TARGET_SR)
    save_audio(s1_path, s1, TARGET_SR)
    save_audio(s2_path, s2, TARGET_SR)
    save_audio(s3_path, s3, TARGET_SR)

    metadata_path = OUTPUT_ROOT / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "sample_id",
                "mix_path",
                "s1_path",
                "s2_path",
                "s3_path",
                "source1_file",
                "source2_file",
                "source3_file",
            ]
        )
        writer.writerow(
            [
                sample_id,
                mix_path.relative_to(OUTPUT_ROOT).as_posix(),
                s1_path.relative_to(OUTPUT_ROOT).as_posix(),
                s2_path.relative_to(OUTPUT_ROOT).as_posix(),
                s3_path.relative_to(OUTPUT_ROOT).as_posix(),
                input_paths[0].name,
                input_paths[1].name,
                input_paths[2].name,
            ]
        )

    print(f"Finished generating 1 Arabic mixture in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()