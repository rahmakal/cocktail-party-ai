from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict, List

import soundfile as sf
import torch
import torchaudio


LIBRISPEECH_ROOT = Path("data")
OUTPUT_ROOT = Path("custom_libri3mix")
NUM_MIXTURES = 500
TARGET_SR = 8000
TARGET_SECONDS = 5
RANDOM_SEED = 42


def collect_speaker_files(root: Path) -> Dict[str, List[Path]]:
    """Collect .flac files grouped by speaker ID inferred from speaker/chapter/file.flac."""
    speaker_to_files: Dict[str, List[Path]] = {}

    for flac_path in root.rglob("*.flac"):
        rel_parts = flac_path.relative_to(root).parts
        if len(rel_parts) < 3:
            continue

        speaker_id = rel_parts[0]
        speaker_to_files.setdefault(speaker_id, []).append(flac_path)

    return {
        speaker_id: sorted(files)
        for speaker_id, files in speaker_to_files.items()
        if files
    }


def build_resampler_cache() -> Dict[int, torchaudio.transforms.Resample]:
    return {}


def load_audio(audio_path: Path) -> tuple[torch.Tensor, int]:
    """
    Prefer torchaudio.load as requested, but fall back to soundfile when a
    torchaudio build requires TorchCodec for decoding.
    """
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform, sample_rate
    except ImportError as exc:
        if "TorchCodec" not in str(exc):
            raise

        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio).transpose(0, 1)
        return waveform, sample_rate


def save_audio(audio_path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """
    Prefer torchaudio.save, but fall back to soundfile if this torchaudio build
    cannot write without extra codec support.
    """
    try:
        torchaudio.save(str(audio_path), waveform.cpu(), sample_rate)
    except (ImportError, RuntimeError, OSError) as exc:
        message = str(exc)
        if "TorchCodec" not in message and "soundfile" not in message.lower():
            raise

        audio = waveform.detach().cpu().transpose(0, 1).numpy()
        sf.write(str(audio_path), audio, sample_rate)


def to_mono_resampled(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sr: int,
    resamplers: Dict[int, torchaudio.transforms.Resample],
) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with shape [channels, time], got {tuple(waveform.shape)}")

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sr:
        if sample_rate not in resamplers:
            resamplers[sample_rate] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
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

    speaker_to_files = collect_speaker_files(LIBRISPEECH_ROOT)
    speakers = sorted(speaker_to_files)
    if len(speakers) < 3:
        raise RuntimeError("Need at least 3 speakers with .flac files to create Libri3Mix samples.")

    output_dirs = ensure_output_dirs(OUTPUT_ROOT)
    metadata_path = OUTPUT_ROOT / "metadata.csv"
    resamplers = build_resampler_cache()

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "sample_id",
                "mix_path",
                "s1_path",
                "s2_path",
                "s3_path",
                "speaker1",
                "speaker2",
                "speaker3",
                "source1_file",
                "source2_file",
                "source3_file",
            ]
        )

        for index in range(NUM_MIXTURES):
            speaker_triplet = rng.sample(speakers, 3)
            source_paths = [rng.choice(speaker_to_files[speaker_id]) for speaker_id in speaker_triplet]

            processed_sources: List[torch.Tensor] = []
            for source_path in source_paths:
                waveform, sample_rate = load_audio(source_path)
                waveform = to_mono_resampled(waveform, sample_rate, TARGET_SR, resamplers)
                waveform = fit_to_length(waveform, target_num_samples, rng)
                processed_sources.append(waveform)

            s1, s2, s3 = processed_sources
            mix = s1 + s2 + s3
            mix, s1, s2, s3 = safe_normalize(mix, s1, s2, s3)

            sample_id = f"sample_{index:04d}"
            mix_path = output_dirs["mix"] / f"{sample_id}.wav"
            s1_path = output_dirs["s1"] / f"{sample_id}.wav"
            s2_path = output_dirs["s2"] / f"{sample_id}.wav"
            s3_path = output_dirs["s3"] / f"{sample_id}.wav"

            save_audio(mix_path, mix, TARGET_SR)
            save_audio(s1_path, s1, TARGET_SR)
            save_audio(s2_path, s2, TARGET_SR)
            save_audio(s3_path, s3, TARGET_SR)

            writer.writerow(
                [
                    sample_id,
                    mix_path.relative_to(OUTPUT_ROOT).as_posix(),
                    s1_path.relative_to(OUTPUT_ROOT).as_posix(),
                    s2_path.relative_to(OUTPUT_ROOT).as_posix(),
                    s3_path.relative_to(OUTPUT_ROOT).as_posix(),
                    speaker_triplet[0],
                    speaker_triplet[1],
                    speaker_triplet[2],
                    source_paths[0].relative_to(LIBRISPEECH_ROOT).as_posix(),
                    source_paths[1].relative_to(LIBRISPEECH_ROOT).as_posix(),
                    source_paths[2].relative_to(LIBRISPEECH_ROOT).as_posix(),
                ]
            )

            if (index + 1) % 50 == 0:
                print(f"Generated {index + 1}/{NUM_MIXTURES} mixtures")

    print(f"Finished generating {NUM_MIXTURES} mixtures in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
