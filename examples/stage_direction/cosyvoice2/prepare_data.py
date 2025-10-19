#!/usr/bin/env python3
import argparse, os, logging, random
from collections import defaultdict

logger = logging.getLogger(__name__)

import re
from pathlib import Path

def clean_filename(name: str) -> str:
    """Strip invisible characters and remove double extensions."""
    name = name.strip().strip('"').strip("'").replace("\r", "").replace("\n", "").replace("\u200b", "")
    # Remove extension if it already ends with .wav or .WAV
    if name.lower().endswith(".wav"):
        name = name[: -4]
    return name


def guess_spk(utt_id: str) -> str:
    """
    Extract speaker ID as the substring between the first and second underscores.
    Example:
        'VO_hin123_Scene001' -> 'hin123'
        'VO_RitaGearspark_SceneR0010' -> 'RitaGearspark'
    """
    m = re.match(r'^[^_]+_([^_]+)_', utt_id)
    if m:
        return m.group(1)
    return utt_id


    """Extract speaker ID from utterance ID."""
    parts = utt_id.split('_')
    if len(parts) >= 2 and parts[0] == 'VO':
        return parts[1]
    return parts[0]

def write_kaldi_files(out_dir, utts, utt2wav, utt2text, utt2spk):
    """Write wav.scp, text, utt2spk, and spk2utt files."""
    os.makedirs(out_dir, exist_ok=True)
    spk2utt = defaultdict(list)
    for u in utts:
        spk2utt[utt2spk[u]].append(u)

    utts_sorted = sorted(utts)
    for s in spk2utt:
        spk2utt[s] = sorted(spk2utt[s])

    with open(os.path.join(out_dir, "wav.scp"), "w", encoding="utf-8") as f:
        for u in utts_sorted:
            f.write(f"{u} {utt2wav[u]}\n")

    with open(os.path.join(out_dir, "text"), "w", encoding="utf-8") as f:
        for u in utts_sorted:
            f.write(f"{u} {utt2text[u]}\n")

    with open(os.path.join(out_dir, "utt2spk"), "w", encoding="utf-8") as f:
        for u in utts_sorted:
            f.write(f"{u} {utt2spk[u]}\n")

    with open(os.path.join(out_dir, "spk2utt"), "w", encoding="utf-8") as f:
        for s in sorted(spk2utt):
            f.write(f"{s} {' '.join(spk2utt[s])}\n")

def split_train_dev(utts, seed, ratio_train=0.9):
    """Split into train/dev deterministically and reproducibly."""
    rng = random.Random(seed)
    order = sorted(utts)
    rng.shuffle(order)

    n = len(order)
    n_train = int(ratio_train * n)
    train = order[:n_train]
    dev = order[n_train:]
    return train, dev

def main(args):
    os.makedirs(args.des_dir, exist_ok=True)

    utt2wav, utt2text, utt2spk = {}, {}, {}

    # Read the input list
    with open(args.list_file, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('|')
            if len(parts) < 2:
                logger.warning("Skipping line %d: not enough fields -> %r", ln, line)
                continue
 
            file_name = parts[0].strip()
            transcript = parts[1].strip()
            utt = file_name
            file_name = clean_filename(parts[0])
            wav_path = os.path.join(args.audio_folder, file_name + ".wav")

            if not os.path.exists(wav_path):
                logger.warning("Audio not found: %s (line %d)", wav_path, ln)
                continue

            spk = guess_spk(utt)
            utt2wav[utt] = wav_path
            utt2text[utt] = transcript
            utt2spk[utt] = spk

    all_utts = list(utt2wav.keys())
    train, dev = split_train_dev(all_utts, args.seed, ratio_train=0.9)

    write_kaldi_files(os.path.join(args.des_dir, "train"), train, utt2wav, utt2text, utt2spk)
    write_kaldi_files(os.path.join(args.des_dir, "dev"), dev, utt2wav, utt2text, utt2spk)

    logging.info(f"Total: {len(all_utts)} | Train: {len(train)} ({len(train)/len(all_utts):.1%}) | "
                 f"Dev: {len(dev)} ({len(dev)/len(all_utts):.1%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_file", type=str, required=True,
                        help="Path to file.txt (file_name|transcript|style...)")
    parser.add_argument("--audio_folder", type=str, required=True,
                        help="Folder containing the .wav files")
    parser.add_argument("--des_dir", type=str, required=True,
                        help="Output directory (creates train/dev subfolders)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for reproducible splits")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main(args)
