import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from transformers import Wav2Vec2Processor
from collections import defaultdict
import evaluate
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import json

def space_separate(sent):
    # Separate phonemes in a given sentence
    phonemes = []
    modifiers = ['ʲ', 'ʷ', 'ʼ', 'ː', 'ˤ', "'"]
    tie_bar = ['͡']

    i = 0
    L = len(sent)

    while i < L:
        # First-level lookahead
        if i + 1 < L and sent[i+1] in modifiers:
            # second-level lookahead
            if i + 2 < L and sent[i+2] in modifiers:
                # third-level lookahead
                if i + 3 < L and sent[i+3] in modifiers:
                    phonemes.append(sent[i:i+4])
                    i += 4
                else:
                    phonemes.append(sent[i:i+3])
                    i += 3
            else:
                phonemes.append(sent[i:i+2])
                i += 2
            continue

        # Tie-bar handling
        if i + 1 < L and sent[i+1] in tie_bar:
            if i + 3 < L and sent[i+3] in modifiers:
                if i + 4 < L and sent[i+4] in modifiers:
                    phonemes.append(sent[i:i+5])
                    i += 5
                else:
                    phonemes.append(sent[i:i+4])
                    i += 4
            elif i + 2 < L:
                phonemes.append(sent[i:i+3])
                i += 3
            else:
                # unexpected end, just take remaining characters
                phonemes.append(sent[i:])
                break
            continue

        # Default: single character
        phonemes.append(sent[i])
        i += 1

    return phonemes

def invert_mapping(ipa2cyrl: dict) -> dict:
    """Safe inverse: ensure no collisions."""
    cyrl2ipa = {}
    for ipa, cyrl in ipa2cyrl.items():
        if cyrl in cyrl2ipa:
            print(f"Warning: Collision for '{cyrl}' ({cyrl2ipa[cyrl]} vs {ipa})")
            continue
        cyrl2ipa[cyrl] = ipa
    return cyrl2ipa

def cyrillic_to_ipa(text: str, cyrl2ipa: dict) -> str:
    text = text.strip().lower()
    keys = sorted(cyrl2ipa.keys(), key=len, reverse=True)
    i, n = 0, len(text)
    out = []

    while i < n:
        matched = False
        for k in keys:
            if text.startswith(k, i):
                out.append(cyrl2ipa[k])
                i += len(k)
                matched = True
                break
        if not matched:
            out.append(text[i])
            i += 1

    return "".join(out)

## Data collator that will dynamically pad the inputs received, as well as the labels.
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # handle additional id field
        if "id" in features[0]:
            id_features = [{"id": feature["id"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # id shouldn't be padded as it is just an index
        if "id" in features[0]:
            batch["id"] = torch.tensor([f["id"] for f in id_features], dtype=torch.long)
            
        batch["labels"] = labels

        return batch


@dataclass
class DataCollatorWhisperWithPadding:
    processor: Any        # WhisperProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # ------------------------------
        # 1. Split input_values + labels
        # ------------------------------
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Preserve ID field if present
        id_present = "id" in features[0]
        if id_present:
            id_list = [f["id"] for f in features]

        # ---------------------------------------
        # 2. Pad encoder inputs (audio features)
        # ---------------------------------------
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
            padding=self.padding,
        )

        # ------------------------------------------------
        # 3. Pad decoder input_ids (text labels)
        #    Important: Whisper MUST KEEP pad_token_id,
        #    not replace with -100 like CTC.
        # ------------------------------------------------
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
            padding=self.padding,
        )

        # HF Whisper expects raw padded labels
        labels = labels_batch["input_ids"]

        # Add id field
        if id_present:
            batch["id"] = torch.tensor(id_list, dtype=torch.long)

        batch["labels"] = labels

        return batch

@dataclass
class DataCollatorQwenAudio:
    """
    Collator for Qwen2-Audio-Instruct and Qwen2.5-Omni.
    Pads encoder audio features and labels for batching.
    Keeps input_ids (prompt) as-is.
    """
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: list) -> dict:
        # ---------------------------------------
        # 1. Pad encoder audio features
        # ---------------------------------------
        prompts = [f["prompts"] for f in features]
        audios = [f["audio_array"] for f in features]
 
        batch = self.processor(
            text=prompts,
            audio=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        if "labels" in features[0]:
            # copy labels from batch["input_ids"], tokenize f["labels"] and mark the left part as -100
            batch["labels"] = batch["input_ids"].clone()
            for i, f in enumerate(features):
                prompt_len = batch["attention_mask"][i].tolist().index(0) if 0 in batch["attention_mask"][i].tolist() else len(batch["attention_mask"][i])
                label_ids = self.processor.tokenizer(f["labels"], return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                label_len = label_ids.size(0)

                # Replace prompt part with -100
                batch["labels"][i, :prompt_len-label_len-2] = -100
        else:
            batch["labels"] = self.processor.tokenizer([f["transcript"] for f in features], add_special_tokens=False, return_tensors="pt", padding=True, padding_side='right').input_ids  # dummy

        # Optional: preserve ID
        if "id" in features[0]:
            batch["id"] = torch.tensor([f["id"] for f in features], dtype=torch.long)

        return batch
         
# Define metrics for evaluation
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def levenshtein_alignment(ref, hyp):
    """
    Align ref and hyp using Levenshtein distance (dynamic programming).
    Returns list of (r_char, h_char) alignments where '-' indicates gap.
    """
    n, m = len(ref), len(hyp)
    dp = np.zeros((n+1, m+1), dtype=int)
    back = np.zeros((n+1, m+1), dtype=object)

    for i in range(1, n+1):
        dp[i, 0] = i
        back[i, 0] = "del"
    for j in range(1, m+1):
        dp[0, j] = j
        back[0, j] = "ins"

    for i in range(1, n+1):
        for j in range(1, m+1):
            if ref[i-1] == hyp[j-1]:
                dp[i, j] = dp[i-1, j-1]
                back[i, j] = "ok"
            else:
                choices = {
                    "sub": dp[i-1, j-1] + 1,
                    "del": dp[i-1, j] + 1,
                    "ins": dp[i, j-1] + 1,
                }
                back[i, j], dp[i, j] = min(choices.items(), key=lambda x: x[1])

    # traceback
    i, j = n, m
    alignment = []
    while i > 0 or j > 0:
        op = back[i, j]
        if op == "ok":
            alignment.append((ref[i-1], hyp[j-1]))
            i, j = i-1, j-1
        elif op == "sub":
            alignment.append((ref[i-1], hyp[j-1]))
            i, j = i-1, j-1
        elif op == "del":
            alignment.append((ref[i-1], "-"))
            i -= 1
        elif op == "ins":
            alignment.append(("-", hyp[j-1]))
            j -= 1
    return alignment[::-1]

def compute_char_stats(pred_str, ref_str):
    """
    Compute S, D, I, N, precision, recall, F1, confusion matrix.
    """
    ref_phonemes = [space_separate(s) for s in ref_str]
    pred_phonemes = [space_separate(s) for s in pred_str]
    
    S = D = I = N = 0
    y_true, y_pred = [], []
    
    for ref, hyp in zip(ref_phonemes, pred_phonemes):
        alignment = levenshtein_alignment(ref, hyp)
        for r, h in alignment:
            if r == h and r != "-":
                N += 1
                y_true.append(r)
                y_pred.append(h)
            elif r == "-" and h != "-":  # insertion
                I += 1
                y_true.append("<eps>")
                y_pred.append(h)
            elif r != "-" and h == "-":  # deletion
                D += 1
                y_true.append(r)
                y_pred.append("<eps>")
            else:  # substitution
                S += 1
                y_true.append(r)
                y_pred.append(h)

    # precision, recall, f1 at aggregate char level
    precision = N / (N + S + I) if (N + S + I) > 0 else 0
    recall = N / (N + S + D) if (N + S + D) > 0 else 0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0

    # compute error rate
    per = (S + D + I) / (N + S + D + I) if (N + S + D + I) > 0 else 0

    labels = list(set(y_true + y_pred))
    # sort labels by unicode value, with <eps> at first
    labels = sorted([l for l in labels if l != "<eps>"])
    labels = ["<eps>"] + labels

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "S": S, "D": D, "I": I, "N": N,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per": per,
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
        "confusion_labels": labels,
        "confusion_matrix": cm
    }


def plot_confusion_matrix(cm, labels, title="Normalized Phoneme Confusion Matrix", cmap="Blues", savepath=None):
    # replace label <eps> with - (hyphen)
    labels = ["-" if l == "<eps>" else l for l in labels]
    
    # Normalize confusion matrix by row
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap)

    # reduce font size of ticks if too many labels
    if len(labels) > 120:
        plt.xticks(fontsize=2)
        plt.yticks(fontsize=2)
    elif len(labels) > 80:
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)
    elif len(labels) > 40:
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
    
    
    plt.colorbar(im, ax=ax, label="Proportion")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # set minor ticks and grid
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.title(title)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


def compute_metrics(pred, processor, tokenized_dataset, model_type='ctc', save_results=False, results_folder="results/default", cyrl2ipa=None):
    if model_type != "ctc":
        # pred_logits is a list/array of variable-length token sequences
        # Whisper outputs token sequences (possibly ragged)
        pred_ids = pred.predictions["generated_tokens"]
        ## replace -100 in pred_ids with pad_token_id
        pad_token_id = processor.tokenizer.pad_token_id
        pred_ids = [np.where(ids == -100, pad_token_id, ids) for ids in pred_ids]
        
    else:
        # CTC branch
        pred_logits = pred.predictions["logits"]
        pred_ids = np.argmax(pred_logits, axis=-1)
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    if model_type == "encoder-decoder-llm":
        # remove prompt from the beginning of each prediction
        pred_str = [s.split('assistant\n')[-1].strip() if 'assistant' in s else s for s in pred_str]  # for Qwen chat format
    else:
        pred_str = [s.replace('\n',' ').replace('#', ' ') for s in pred_str]

    if cyrl2ipa: 
        pred_str = [cyrillic_to_ipa(s, cyrl2ipa) for s in pred_str]
        
    # obtain ref_str from tokenized_dataset and align with pred_str using ids
    id_to_ref = {i: t for i, t in zip(tokenized_dataset["id"], tokenized_dataset["transcript"])}
    ref_str = [id_to_ref[i] for i in pred.predictions["id"]]
     
    wer_val = wer_metric.compute(predictions=pred_str, references=ref_str)
    cer_val = cer_metric.compute(predictions=pred_str, references=ref_str)


    char_stats = compute_char_stats(pred_str, ref_str)

    if save_results:
        os.makedirs(results_folder, exist_ok=True)
        # store references and predictions in a single tsv
        with open(os.path.join(results_folder, "predictions.tsv"), "w", encoding="utf-8") as f:
            f.write("Reference\tPrediction\n")
            for r, p in zip(ref_str, pred_str):
                f.write(f"{r}\t{p}\n")
    
        # plot confusion matrix, with labels sorted by unicode value, <eps> at first
        plot_confusion_matrix(char_stats["confusion_matrix"], char_stats["confusion_labels"], 
                              title=f"Normalized Phoneme Confusion Matrix - {results_folder.replace('results/','').replace('/','_')}", 
                              savepath=os.path.join(results_folder, "confusion_matrix.png"))
        
        stats_dict = {
            "wer": wer_val,
            "cer": cer_val,
            "char_stats": char_stats
        }


        stats_dict["char_stats"]["confusion_matrix"] = stats_dict["char_stats"]["confusion_matrix"].tolist()		
        # save out_dict as json
        with open(os.path.join(results_folder, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats_dict, f, indent=4, ensure_ascii=False)
    
    return {"wer": round(wer_val,3), "cer": round(cer_val,3), "per": round(char_stats["per"],3)}
