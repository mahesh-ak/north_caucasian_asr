import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import Wav2Vec2Processor
from collections import defaultdict
import evaluate
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def space_separate(sent):
    # Separate phonemes in a given sentence
    phonemes = []

    i = 0
    while i < len(sent):
        if i < len(sent)-1:
            if sent[i+1] in ['ʲ', 'ʷ', 'ʼ', 'ː','ˤ', "'"]:
                if i < len(sent) - 2 and sent[i+2] in ['ʲ', 'ʷ', 'ʼ', 'ː','ˤ', "'"]:
                    phonemes.append(sent[i]+sent[i+1]+sent[i+2])
                    i += 3
                else:
                    phonemes.append(sent[i]+sent[i+1])
                    i += 2
                continue
            elif sent[i+1] in ['͡']:
                if i < len(sent) - 3 and sent[i+3] in ['ʲ', 'ʷ', 'ʼ', 'ː','ˤ', "'"]:
                    phonemes.append(sent[i]+sent[i+1]+sent[i+2]+sent[i+3])
                    i += 4
                else:
                    phonemes.append(sent[i]+sent[i+1]+sent[i+2])
                    i += 3
                continue
        phonemes.append(sent[i])
        if sent[i] in ['ʲ', 'ʷ', 'ʼ', 'ː','ˤ', "'"]:
            print('\t',sent)
        i += 1
    return phonemes

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

        batch["labels"] = labels

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

    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "S": S, "D": D, "I": I, "N": N,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_labels": sorted(labels),
        "confusion_matrix": cm,
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    }


def plot_confusion_matrix(cm, labels, title="Normalized Phoneme Confusion Matrix", cmap="Blues", savepath=None):
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap)

    plt.colorbar(im, ax=ax, label="Proportion")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    plt.title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


def compute_metrics(pred, processor, tokenized_dataset, save_results=False, results_folder="results/default"):
    pred_logits = pred.predictions
    pred_ids    = np.argmax(pred_logits, axis=-1)

    pred_str  = processor.batch_decode(pred_ids)
    pred_str = [s.replace('#', ' ') for s in pred_str]

    ref_str = list(tokenized_dataset["transcript"])
    
    wer_val = wer_metric.compute(predictions=pred_str, references=ref_str)
    cer_val = cer_metric.compute(predictions=pred_str, references=ref_str)


    char_stats = compute_char_stats(pred_str, ref_str)

    if save_results:
        os.makedirs(results_folder, exist_ok=True)
        with open(os.path.join(results_folder, "predictions.txt"), "w") as f:
            for p in pred_str:
                f.write(p + "\n")
        with open(os.path.join(results_folder, "references.txt"), "w") as f:
            for r in ref_str:
                f.write(r + "\n")
        # optionally save confusion matrix
        np.savetxt(os.path.join(results_folder, "confusion_matrix.csv"), 
                   char_stats["confusion_matrix"], delimiter=",", fmt="%d")
    
    return {
        "wer": wer_val,
        "cer": cer_val,
        "char_stats": char_stats
    }
