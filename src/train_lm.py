import json
import os
from pathlib import Path
import pandas as pd
from utils import space_separate
import subprocess

def load_train_folders(data_dir: str, split_file: str):
    data_dir = Path(data_dir)
    split = json.loads(Path(split_file).read_text())
    train_dirs = [data_dir / p for p in split["train"]]
    return [d for d in train_dirs if d.exists()]


def read_transcripts(folders):
    lines = []
    for folder in folders:
        tsv = folder / "dataset.csv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, dtype=str)
        if "transcript" not in df:
            continue
        for t in df["transcript"].astype(str):
            t = " ".join(t.split())  # normalize whitespace
            lines.append(t)
    return lines


def write_corpus(lines, out_path):
    """
    folders: list of Path objects
    output_path: Path to write corpus.txt
    tokenize_phonemes: function(str) -> List[str]
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            phonemes = space_separate(line)  # returns list of tokens
            if not phonemes:
                continue
            # join with spaces to make KenLM-friendly corpus
            f.write("".join(phonemes) + "\n")



def train_kn_lm(corpus_txt, output_dir, order=2):
    """
    Train a Kneser-Ney n-gram language model using KenLM's lmplz.

    Args:
        corpus_txt (str or Path): Path to training corpus 
                                  (one sentence per line, tokens space-separated)
        output_dir (str or Path): Directory to store LM outputs
        order (int): n-gram order (default 3)
        memory (str): memory limit for lmplz (e.g. '50%', '10G')
    """

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arpa_path = output_dir / "lm.arpa"
    binary_path = output_dir / "lm.klm"

    print(f"Training KenLM {order}-gram Kneser-Ney LM…")
    print(f"Input corpus: {corpus_txt}")
    print(f"Output ARPA:  {arpa_path}")
    print(f"Output binary: {binary_path}")

    # Run lmplz to build ARPA
    lmplz_cmd = [
    "lmplz",
    f"--order={order}",
    "--discount_fallback",
    "--text", str(corpus_txt),
    "--arpa", str(arpa_path),
    "--temp_prefix", str(output_dir / "lm_temp"),
    "--memory", "200G"
    ] 

    subprocess.run(lmplz_cmd, check=True)

    # Convert ARPA → binary KenLM format (faster loading)
    build_binary_cmd = [
        "build_binary",
        str(arpa_path),
        str(binary_path)
    ]

    subprocess.run(build_binary_cmd, check=True)

    print("Done.")
    print(f"ARPAs saved to: {arpa_path}")
    print(f"Binary LM saved to: {binary_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split-file", required=True)

    p.add_argument("--order", type=int, default=2)
    args = p.parse_args()
    ## output dir is models/data_dir.parts[1:]/split_file.name without .json
    output_dir = Path("models") / Path(args.data_dir).name / 'lm' / Path(args.split_file).stem
    os.makedirs(output_dir, exist_ok=True)

    folders = load_train_folders(args.data_dir, args.split_file)
    transcripts = read_transcripts(folders)

    corpus = output_dir / "corpus.txt"
    write_corpus(transcripts, corpus)

    # Train Kneser-Ney LM
    train_kn_lm(str(corpus), output_dir, order=args.order)
