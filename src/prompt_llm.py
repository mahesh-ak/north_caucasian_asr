import argparse
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from utils import compute_char_stats, plot_confusion_matrix, wer_metric, cer_metric, cyrillic_to_ipa, invert_mapping
from openai import OpenAI
import base64
import io
import soundfile as sf



load_dotenv()  # <-- loads OPENAI_API_KEY from .env
client = OpenAI()  # key is picked automatically


def collect_split_samples(data_dir, split_json, split_name):
    """
    Returns a list of tuples (audio_path, transcript) without modifying paths.
    audio_path values are assumed to be absolute paths already.
    """
    with open(split_json, "r", encoding="utf-8") as f:
        split_map = json.load(f)

    if split_name not in split_map:
        raise ValueError(f"Split '{split_name}' not found in {split_json}")

    samples = []
    for folder in split_map[split_name]:
        csv_path = Path(data_dir) / folder / "dataset.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing: {csv_path}, skipping.")
            continue

        df = pd.read_csv(csv_path)
        if "audio_path" not in df or "transcript" not in df:
            raise ValueError(f"CSV missing required columns: {csv_path}")

        # NO path modification here:
        samples.extend(list(zip(df["audio_path"], df["transcript"])))

    return samples

def compute_metrics_openai(pred_str, ref_str, save_results=False, results_folder="results/openai", cyrl2ipa=None):
    """
    Compute WER, CER, PER and confusion stats using your existing utilities and metric objects
    (wer_metric, cer_metric) imported from utils.py. pred_str and ref_str must be lists of strings.
    """
    
    if cyrl2ipa: 
        pred_str = [cyrillic_to_ipa(s, cyrl2ipa) for s in pred_str]
    # compute standard metrics via your imported metric objects
    wer_val = wer_metric.compute(predictions=pred_str, references=ref_str)
    cer_val = cer_metric.compute(predictions=pred_str, references=ref_str)

    # character-level stats / PER & confusion
    char_stats = compute_char_stats(pred_str, ref_str)

    if save_results:
        os.makedirs(results_folder, exist_ok=True)

        # store references and predictions in a single tsv (same format as training pipeline)
        with open(os.path.join(results_folder, "predictions.tsv"), "w", encoding="utf-8") as f:
            f.write("Reference\tPrediction\n")
            for r, p in zip(ref_str, pred_str):
                f.write(f"{r}\t{p}\n")

        # plot confusion matrix, with labels sorted by unicode value, <eps> first
        plot_confusion_matrix(
            char_stats["confusion_matrix"],
            char_stats["confusion_labels"],
            title=f"Normalized Phoneme Confusion Matrix - {results_folder.replace('results/','').replace('/','_')}",
            savepath=os.path.join(results_folder, "confusion_matrix.png"),
        )

        stats_dict = {
            "wer": wer_val,
            "cer": cer_val,
            "char_stats": char_stats
        }

        # convert numpy arrays to lists for JSON serialisation
        stats_dict["char_stats"]["confusion_matrix"] = stats_dict["char_stats"]["confusion_matrix"].tolist()

        # write stats json
        with open(os.path.join(results_folder, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats_dict, f, indent=4, ensure_ascii=False)

    # Return numbers in same shape as your original compute_metrics
    return {"wer": round(wer_val, 3), "cer": round(cer_val, 3), "per": round(char_stats["per"], 3)}


def transcribe_openai(audio_path, model="gpt-4o-transcribe", prompt=None):
    """
    Transcribe audio using OpenAI models.
    - Audio-capable models: use client.audio.transcriptions.create()
    - Chat-only models (gpt-5): send as chat message with instruction
    """
    audio_models = [
        "gpt-4o-transcribe",
        "gpt-4o-audio-preview",
        "gpt-audio",
        "gpt-audio-mini",
    ]

    if any(audio_model in model for audio_model in audio_models):
        # Use audio transcription endpoint
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model=model,
                file=f,
                prompt=prompt
            )
        return resp.text
    else:
        # Thinking models (gpt-5, gpt-4o) -> chat approach
        # Convert audio to PCM16 WAV bytes
        audio_data, sr = sf.read(audio_path, dtype="int16")
        buf = io.BytesIO()
        sf.write(buf, audio_data, sr, format="WAV")
        wav_bytes = buf.getvalue()
        # Encode as base64
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        system_prompt = (
            prompt
            if prompt
            else "Transcribe the audio (base64 WAV) into IPA, output only the phonetic transcription."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<audio:{audio_b64}>"},
        ]

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()





def run_inference(data_dir, split_json, split_name, model_name, results_dir, prompt):
    samples = collect_split_samples(data_dir, split_json, split_name)

    preds, refs = [], []
    for audio, ref in tqdm(samples, desc=f"Running inference on {split_name}"):
        suceeded = False
        count = 0
        while not suceeded:
            try:
                pred = transcribe_openai(audio, model=model_name, prompt=prompt)
            except Exception as e:
                count += 1
                print(f"[ERROR] Failed on {audio}: {e}, {count} time")
                pred = ""
                continue
            suceeded = True
        preds.append(pred)
        refs.append(ref)

    with open(Path(data_dir) / "ipa2cyrl.json", "r", encoding="utf-8") as f:
        ipa2cyrl = json.load(f)
    cyrl2ipa = invert_mapping(ipa2cyrl)
    eval_summary = compute_metrics_openai(preds, refs, save_results=True, results_folder=results_dir, cyrl2ipa=cyrl2ipa)
    print(eval_summary)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True,
                   help="Root folder with subfolders containing dataset.csv")
    p.add_argument("--split-json", required=True,
                   help="Path to split.json file mapping split -> [folders]")
    p.add_argument("--model", default="gpt-4o-transcribe",
                   help="OpenAI ASR model name")
    p.add_argument("--results-dir", required=False,
                     help="Folder to save inference results. If not provided, defaults to results/<data_dir.parts[1:]>/<model>/<split name>")
    p.add_argument(
                    "--prompt",
                    default="Transcribe the audio in {lang} (a North Caucasian language) into Cyrillic. Do not translate, interpret, or add punctuation. Output only the transcription.",
                    help="Optional transcription prompt for IPA output"
                )
    p.add_argument("--vocab", required=False,
                   help="Path to vocab.json whose keys define allowed phonemes (exclude <s>, </s>, <unk>, <pad>, '|', ' ' )")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    split_json = Path(args.split_json)
    model_name = args.model

    if args.results_dir:
        inferred_results = Path(args.results_dir)
    else:
        inferred_results = (
            Path("results")
            / Path(*data_dir.parts[1:])
            / model_name
            / split_json.stem
        )

    # ---------- PROMPT HANDLING ----------
    prompt = args.prompt
    if "{lang}" in prompt:
        lang = data_dir.parts[1]
        prompt = prompt.replace("{lang}", lang)
 
    # Inject vocab if present
    if args.vocab:
        with open(args.vocab) as f:
            vocab = json.load(f)
        # Keep only phoneme keys
        phonemes = sorted([k for k in vocab.keys()
                           if k not in ["<s>", "</s>", "<pad>", "<unk>", "|", " "]])
        ipa_inventory = ", ".join(phonemes)

        prompt += (
            f"\nOnly use the following IPA symbols known to occur in this language:\n"
            f"{ipa_inventory}\n"
            f"Do not invent phonemes outside this inventory."
        )

    # ---------- CALL ----------
    run_inference(
        data_dir=str(data_dir),
        split_json=str(split_json),
        split_name="test",
        model_name=model_name,
        results_dir=str(inferred_results),
        prompt=prompt,  
    )



if __name__ == "__main__":
    main()
