"""
generate_mixtures.py — Conv-TasNet
  Supporte Mozilla Common Voice (TSV) + dossiers par locuteur

Features :
  - Lecture du client_id depuis validated.tsv (Mozilla Common Voice)
  - Fallback : structure par dossiers (speaker_01/, speaker_02/, ...)
  - Strict different-speaker-per-mixture enforcement
  - VAD (Voice Activity Detection)
  - Random SNR entre sources (-5 à +5 dB)
  - Split train / val / test automatique
  - Reproductible avec --seed

Structures supportées :
  ① Mozilla Common Voice (recommandé) :
      data/raw_sources/
          validated.tsv
          clips/
              clip_001.mp3
              clip_002.mp3
              ...

  ② Dossiers par locuteur :
      data/raw_sources/
          speaker_01/ clip_001.wav ...
          speaker_02/ ...

Usage :
    # Mozilla Common Voice
    python main.py generate \\
        --src_dir data/raw_sources \\
        --tsv     data/raw_sources/validated.tsv \\
        --n_mix   8000 --n_src 2

    # Dossiers classiques
    python main.py generate \\
        --src_dir data/raw_sources \\
        --n_mix   8000 --n_src 2
"""

import os
import json
import math
import random
import argparse
import torch
import torchaudio


# ─────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir",      type=str,   default="data/raw_sources")
    p.add_argument("--tsv",          type=str,   default=None,
                   help="Chemin vers validated.tsv (Mozilla Common Voice). "
                        "Si omis, utilise la structure par dossiers.")
    p.add_argument("--out_dir",      type=str,   default="data/mixtures")
    p.add_argument("--n_mix",        type=int,   default=8000)
    p.add_argument("--n_src",        type=int,   default=2,
                   help="Nb de sources par mixture (2 recommandé pour débuter)")
    p.add_argument("--sample_rate",  type=int,   default=16000)
    p.add_argument("--duration",     type=float, default=4.0)
    p.add_argument("--snr_min",      type=float, default=-5.0)
    p.add_argument("--snr_max",      type=float, default=5.0)
    p.add_argument("--min_speech",   type=float, default=0.3)
    p.add_argument("--min_clip_dur", type=float, default=1.5)
    p.add_argument("--train_ratio",  type=float, default=0.8)
    p.add_argument("--val_ratio",    type=float, default=0.1)
    p.add_argument("--min_spk_clips",type=int,   default=3,
                   help="Nb minimum de clips valides pour garder un locuteur")
    p.add_argument("--max_spk",      type=int,   default=None,
                   help="Limiter le nb de locuteurs (None = tous)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--no_vad",       action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────
#  Audio utilities
# ─────────────────────────────────────────────
def load_wav(path, target_sr):
    """Charge un fichier audio → tensor mono float32 (T,)"""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


def speech_ratio(wav, threshold=0.01):
    frame_len = 400
    if wav.shape[0] < frame_len:
        return 0.0
    frames = wav.unfold(0, frame_len, frame_len // 2)
    rms    = frames.pow(2).mean(dim=-1).sqrt()
    return (rms > threshold).float().mean().item()


def is_valid_clip(wav, sr, min_duration_s, min_speech_ratio, use_vad):
    dur = wav.shape[-1] / sr
    if dur < min_duration_s:
        return False, f"trop court ({dur:.1f}s < {min_duration_s}s)"
    rms = wav.pow(2).mean().sqrt().item()
    if rms < 1e-4:
        return False, "quasi-silencieux (RMS < 1e-4)"
    if use_vad:
        ratio = speech_ratio(wav)
        if ratio < min_speech_ratio:
            return False, f"trop de silence (ratio={ratio:.2f} < {min_speech_ratio})"
    return True, "ok"


def cut_or_pad(wav, target_len, sr):
    L = wav.shape[-1]
    if L >= target_len:
        start = random.randint(0, L - target_len)
        return wav[start: start + target_len]
    repeats = (target_len // L) + 2
    tiled   = wav.repeat(repeats)
    start   = random.randint(0, tiled.shape[-1] - target_len)
    return tiled[start: start + target_len]


def normalize_rms(wav, target_db=-25.0):
    rms = wav.pow(2).mean().sqrt()
    if rms < 1e-8:
        return wav
    target_rms = 10 ** (target_db / 20)
    return wav * (target_rms / rms)


def scale_to_snr(src, reference, snr_db):
    rms_ref    = reference.pow(2).mean().sqrt() + 1e-8
    rms_src    = src.pow(2).mean().sqrt()        + 1e-8
    target_rms = rms_ref * (10 ** (snr_db / 20))
    return src * (target_rms / rms_src)


def safe_mix(sources):
    mixture = sum(sources)
    peak    = mixture.abs().max()
    if peak > 0.98:
        scale   = 0.98 / peak
        mixture = mixture * scale
        sources = [s * scale for s in sources]
    return mixture, sources


# ─────────────────────────────────────────────
#  Index des locuteurs — MODE TSV (Mozilla CV)
# ─────────────────────────────────────────────
def build_index_from_tsv(tsv_path, src_dir, target_sr,
                          min_dur, min_speech, use_vad,
                          min_spk_clips, max_spk):
    """
    Lit validated.tsv et construit {client_id: [chemins valides]}.
    Les fichiers .mp3 sont cherchés dans src_dir/clips/.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas est requis pour le mode TSV : pip install pandas")

    print(f"[TSV] Lecture de {tsv_path} ...")
    df = pd.read_csv(tsv_path, sep="\t", usecols=["client_id", "path"])
    print(f"[TSV] {len(df)} entrées, {df['client_id'].nunique()} locuteurs uniques")

    clips_dir = os.path.join(src_dir, "clips")
    if not os.path.isdir(clips_dir):
        # Certaines versions CV mettent les clips directement dans src_dir
        clips_dir = src_dir
        print(f"[TSV] Dossier 'clips/' absent — cherche dans {clips_dir}")

    by_speaker     = {}
    n_rejected     = 0
    reject_reasons = {}
    n_missing      = 0

    rows = df.to_dict("records")
    total = len(rows)

    for idx, row in enumerate(rows):
        client_id = str(row["client_id"])
        fname     = row["path"]
        path      = os.path.join(clips_dir, fname)

        if not os.path.exists(path):
            n_missing += 1
            continue

        try:
            wav = load_wav(path, target_sr)
            ok, reason = is_valid_clip(wav, target_sr, min_dur, min_speech, use_vad)
        except Exception as e:
            ok, reason = False, f"erreur lecture: {e}"

        if not ok:
            n_rejected += 1
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            continue

        by_speaker.setdefault(client_id, []).append(path)

        if (idx + 1) % 1000 == 0:
            n_valid = sum(len(v) for v in by_speaker.values())
            print(f"  [{idx+1}/{total}] {n_valid} clips valides, "
                  f"{len(by_speaker)} locuteurs, {n_rejected} rejetés")

    # Filtrer les locuteurs avec trop peu de clips
    before = len(by_speaker)
    by_speaker = {k: v for k, v in by_speaker.items() if len(v) >= min_spk_clips}
    after = len(by_speaker)
    print(f"\n[Filter] {before - after} locuteurs retirés (< {min_spk_clips} clips valides)")
    print(f"[Filter] {after} locuteurs retenus")

    # Limiter le nb de locuteurs si demandé
    if max_spk and len(by_speaker) > max_spk:
        # Garder les locuteurs avec le plus de clips (plus de diversité)
        sorted_spk = sorted(by_speaker.items(), key=lambda x: len(x[1]), reverse=True)
        by_speaker = dict(sorted_spk[:max_spk])
        print(f"[Filter] Limité à {max_spk} locuteurs (--max_spk)")

    n_valid = sum(len(v) for v in by_speaker.values())
    stats = {
        "mode"          : "tsv",
        "total_rows"    : total,
        "missing_files" : n_missing,
        "valid"         : n_valid,
        "rejected"      : n_rejected,
        "reject_reasons": reject_reasons,
        "n_speakers"    : len(by_speaker),
    }
    return by_speaker, stats


# ─────────────────────────────────────────────
#  Index des locuteurs — MODE DOSSIERS
# ─────────────────────────────────────────────
def build_index_from_dirs(src_dir, target_sr, min_dur,
                           min_speech, use_vad, min_spk_clips):
    """
    Structure : src_dir/speaker_id/clip.wav
    Si flat (pas de sous-dossiers), tous les clips → 'unknown'.
    ATTENTION : si tout est 'unknown', les mélanges ne seront pas
    multi-locuteurs. Utilise plutôt le mode TSV avec Mozilla CV.
    """
    all_files = []
    for root, _, fnames in os.walk(src_dir):
        for fname in sorted(fnames):
            if fname.lower().endswith((".wav", ".mp3", ".flac")):
                all_files.append(os.path.join(root, fname))

    print(f"[Scan] {len(all_files)} fichiers audio trouvés — validation...")

    by_speaker     = {}
    n_rejected     = 0
    reject_reasons = {}

    for idx, path in enumerate(all_files):
        try:
            wav = load_wav(path, target_sr)
            ok, reason = is_valid_clip(wav, target_sr, min_dur, min_speech, use_vad)
        except Exception as e:
            ok, reason = False, f"erreur lecture: {e}"

        if not ok:
            n_rejected += 1
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            continue

        spk = os.path.basename(os.path.dirname(path))
        if spk == os.path.basename(src_dir) or spk == "":
            spk = "unknown"

        by_speaker.setdefault(spk, []).append(path)

        if (idx + 1) % 500 == 0:
            n_valid = sum(len(v) for v in by_speaker.values())
            print(f"  [{idx+1}/{len(all_files)}] {n_valid} valides, "
                  f"{n_rejected} rejetés")

    # Avertissement si flat dataset
    if list(by_speaker.keys()) == ["unknown"]:
        print("\n" + "!"*58)
        print("  AVERTISSEMENT : Tous les clips ont le speaker 'unknown'.")
        print("  Les mélanges ne seront PAS multi-locuteurs.")
        print("  → Utilise --tsv validated.tsv pour Mozilla Common Voice.")
        print("!"*58 + "\n")
    else:
        by_speaker = {k: v for k, v in by_speaker.items()
                      if len(v) >= min_spk_clips}

    n_valid = sum(len(v) for v in by_speaker.values())
    stats = {
        "mode"          : "dirs",
        "valid"         : n_valid,
        "rejected"      : n_rejected,
        "reject_reasons": reject_reasons,
        "n_speakers"    : len(by_speaker),
    }
    return by_speaker, stats


# ─────────────────────────────────────────────
#  Génération d'un mélange
# ─────────────────────────────────────────────
def generate_one_mixture(by_speaker, speakers, n_src,
                          target_len, sr, snr_min, snr_max):
    """
    Sélectionne n_src clips de n_src locuteurs DIFFÉRENTS.
    Garantit la diversité inter-locuteurs dans chaque mixture.
    """
    if len(speakers) >= n_src:
        chosen_spks  = random.sample(speakers, n_src)
        chosen_files = [random.choice(by_speaker[s]) for s in chosen_spks]
    else:
        # Fallback : pas assez de locuteurs distincts
        all_files = [f for files in by_speaker.values() for f in files]
        if len(all_files) < n_src:
            raise RuntimeError("Pas assez de clips valides")
        chosen_files = random.sample(all_files, n_src)

    sources = []
    for path in chosen_files:
        wav = load_wav(path, sr)
        wav = cut_or_pad(wav, target_len, sr)
        wav = normalize_rms(wav, target_db=-25.0)
        sources.append(wav)

    # SNR aléatoire entre les sources
    reference = sources[0]
    scaled    = [reference]
    for src in sources[1:]:
        snr_db = random.uniform(snr_min, snr_max)
        scaled.append(scale_to_snr(src, reference, snr_db))

    mixture, scaled = safe_mix(scaled)
    return mixture, scaled


def save_mixture(mix_dir, mixture, sources, sr):
    os.makedirs(mix_dir, exist_ok=True)
    torchaudio.save(os.path.join(mix_dir, "mixture.wav"),
                    mixture.unsqueeze(0), sr)
    for j, src in enumerate(sources):
        torchaudio.save(os.path.join(mix_dir, f"source_{j+1}.wav"),
                        src.unsqueeze(0), sr)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_len = int(args.duration * args.sample_rate)

    print(f"\n{'='*58}")
    print(f"  Conv-TasNet — Mixture Generator")
    print(f"{'='*58}")
    print(f"  Mode         : {'TSV (Mozilla CV)' if args.tsv else 'Dossiers'}")
    print(f"  Source dir   : {args.src_dir}")
    if args.tsv:
        print(f"  TSV          : {args.tsv}")
    print(f"  Output dir   : {args.out_dir}")
    print(f"  Mixtures     : {args.n_mix}")
    print(f"  Sources/mix  : {args.n_src}  "
          f"(PIT: {args.n_src}! = {math.factorial(args.n_src)} permutations)")
    print(f"  Durée        : {args.duration}s @ {args.sample_rate}Hz")
    print(f"  SNR range    : [{args.snr_min}, {args.snr_max}] dB")
    print(f"  VAD          : {'désactivé' if args.no_vad else 'activé'}")
    print(f"  Min clips/spk: {args.min_spk_clips}")
    print(f"{'='*58}\n")

    # ── Construire l'index des locuteurs ─────
    if args.tsv:
        by_speaker, scan_stats = build_index_from_tsv(
            tsv_path   = args.tsv,
            src_dir    = args.src_dir,
            target_sr  = args.sample_rate,
            min_dur    = args.min_clip_dur,
            min_speech = args.min_speech,
            use_vad    = not args.no_vad,
            min_spk_clips = args.min_spk_clips,
            max_spk    = args.max_spk,
        )
    else:
        by_speaker, scan_stats = build_index_from_dirs(
            src_dir    = args.src_dir,
            target_sr  = args.sample_rate,
            min_dur    = args.min_clip_dur,
            min_speech = args.min_speech,
            use_vad    = not args.no_vad,
            min_spk_clips = args.min_spk_clips,
        )

    print(f"\n[Scan] Résultats :")
    print(f"  Clips valides  : {scan_stats['valid']}")
    print(f"  Rejetés        : {scan_stats['rejected']}")
    for reason, count in scan_stats["reject_reasons"].items():
        print(f"    - {reason}: {count}")
    print(f"  Locuteurs      : {scan_stats['n_speakers']}")

    if scan_stats["n_speakers"] < args.n_src:
        raise RuntimeError(
            f"Seulement {scan_stats['n_speakers']} locuteurs valides — "
            f"besoin d'au moins {args.n_src} pour n_src={args.n_src}.\n"
            f"→ Vérifie ton TSV ou augmente min_spk_clips."
        )

    # Ratio locuteurs/sources (santé du dataset)
    ratio = scan_stats["n_speakers"] / args.n_src
    print(f"\n  Ratio locuteurs/n_src : {ratio:.1f}x  "
          f"({'✓ bon' if ratio >= 10 else '⚠ faible — diversité limitée'})")

    speakers = list(by_speaker.keys())

    # ── Génération ───────────────────────────
    print(f"\n[Gen] Génération de {args.n_mix} mélanges...")
    os.makedirs(args.out_dir, exist_ok=True)

    generated    = 0
    errors       = 0
    attempts     = 0
    max_attempts = args.n_mix * 10

    while generated < args.n_mix and attempts < max_attempts:
        attempts += 1
        try:
            mixture, sources = generate_one_mixture(
                by_speaker = by_speaker,
                speakers   = speakers,
                n_src      = args.n_src,
                target_len = target_len,
                sr         = args.sample_rate,
                snr_min    = args.snr_min,
                snr_max    = args.snr_max,
            )
            mix_dir = os.path.join(args.out_dir, f"mix_{generated:05d}")
            save_mixture(mix_dir, mixture, sources, args.sample_rate)
            generated += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [Erreur] {e}")
            continue

        if generated % 500 == 0 or generated == args.n_mix:
            print(f"  {generated}/{args.n_mix} générés  ({errors} erreurs ignorées)")

    # ── Split train / val / test ─────────────
    print(f"\n[Split] Création du split train/val/test...")
    all_mix = sorted([
        d for d in os.listdir(args.out_dir)
        if os.path.isdir(os.path.join(args.out_dir, d)) and d.startswith("mix_")
    ])

    # Shuffle reproductible avec le seed
    rng = random.Random(args.seed)
    rng.shuffle(all_mix)

    n_total = len(all_mix)
    n_train = int(n_total * args.train_ratio)
    n_val   = int(n_total * args.val_ratio)
    n_test  = n_total - n_train - n_val

    split = {
        "train" : all_mix[:n_train],
        "val"   : all_mix[n_train: n_train + n_val],
        "test"  : all_mix[n_train + n_val:],
    }
    split_path = os.path.join(args.out_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    # Sauvegarde des stats
    stats_path = os.path.join(args.out_dir, "generation_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "args"       : vars(args),
            "scan_stats" : scan_stats,
            "generated"  : generated,
            "errors"     : errors,
            "split"      : {"train": n_train, "val": n_val, "test": n_test},
        }, f, indent=2)

    print(f"\n{'='*58}")
    print(f"  TERMINÉ")
    print(f"{'='*58}")
    print(f"  Générés      : {generated} mélanges")
    print(f"  Erreurs      : {errors}")
    print(f"  Train        : {n_train}")
    print(f"  Val          : {n_val}")
    print(f"  Test         : {n_test}")
    print(f"  Split file   : {split_path}")
    print(f"\n  Étape suivante : python main.py train")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
