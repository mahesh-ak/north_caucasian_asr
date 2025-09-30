import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from transformers import Wav2Vec2Processor
from collections import defaultdict
import evaluate
import numpy as np


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


## Function to compute metrics, include tokenized_dataset to get exact transcripts as labels may have been altered during tokenization
def compute_metrics(pred, processor, tokenized_dataset):
    # 1) decode predictions and labels as you had them
    pred_logits = pred.predictions
    pred_ids    = np.argmax(pred_logits, axis=-1)

    # replace -100 in labels as pad_token, so batch_decode keeps alignment
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # 2) compute overall WER/CER
    pred_str = [s.replace('#', ' ') for s in pred_str]
    label_str = [s.replace('#', ' ') for s in label_str]
    
    wer = wer_metric.compute( predictions=pred_str, references=label_str )
    cer = cer_metric.compute( predictions=pred_str, references=label_str )

    # 3) helper: align one ref/hyp pair at CHAR level
    def _align_chars(ref, hyp):
        # build DP tables
        n, m = len(ref), len(hyp)
        dp = [[0]*(m+1) for _ in range(n+1)]
        op = [[None]*(m+1) for _ in range(n+1)]
        for i in range(n+1):
            dp[i][0] = i; op[i][0] = 'D'
        for j in range(m+1):
            dp[0][j] = j; op[0][j] = 'I'
        op[0][0] = None

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if ref[i-1] == hyp[j-1] else 1
                # match/sub
                best, best_op = dp[i-1][j-1] + cost, ('M' if cost==0 else 'S')
                # delete
                if dp[i-1][j] + 1 < best:
                    best, best_op = dp[i-1][j] + 1, 'D'
                # insert
                if dp[i][j-1] + 1 < best:
                    best, best_op = dp[i][j-1] + 1, 'I'
                dp[i][j], op[i][j] = best, best_op

        # backtrace
        i, j = n, m
        ops = []
        while i>0 or j>0:
            o = op[i][j]
            if o in ('M','S'):
                ops.append((ref[i-1], hyp[j-1], o))
                i, j = i-1, j-1
            elif o == 'D':
                ops.append((ref[i-1], '', 'D'))
                i -= 1
            else:  # 'I'
                ops.append(('', hyp[j-1], 'I'))
                j -= 1
        return reversed(ops)

    # 4) accumulate per-character statistics
    char_stats = defaultdict(lambda: {'S':0, 'D':0, 'I':0, 'N':0})
    for r_str, p_str in zip(label_str, pred_str):
        for r_ch, p_ch, o in _align_chars(r_str, p_str):
            if r_ch:
                char_stats[r_ch]['N'] += 1
                if o == 'S':
                    char_stats[r_ch]['S'] += 1
                elif o == 'D':
                    char_stats[r_ch]['D'] += 1
            if o == 'I' and p_ch:
                # attribute insertion to the inserted character itself
                char_stats[p_ch]['I'] += 1

    # 5) return everything
    return {
        "wer": wer,
        "cer": cer,
        "pred": pred_str,
        "labels": label_str,
        "char_stats": dict(char_stats)
    }

