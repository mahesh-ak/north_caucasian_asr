import argparse
import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_from_disk
from tqdm import tqdm
from prompt_llm import compute_metrics_openai
import kenlm
import numpy as np
import math

def load_kenlm_model(lm_path):
    """
    Load a KenLM n-gram LM.
    lm_path: path to ARPA or binary LM (.arpa or .klm)
    """
    model = kenlm.Model(str(lm_path))
    return model


def lm_score(model, tokens, alpha=1.0):
    """
    Compute log-score of a sequence of tokens using KenLM.

    Args:
        model: kenlm.Model object
        tokens: list of words (strings)
        alpha: optional scaling factor

    Returns:
        total log-probability score
    """
    # KenLM expects a string with space-separated words
    sentence = " ".join(tokens)
    # .score returns log10 probability by default
    score = model.score(sentence, bos=True, eos=True)
    return alpha * score * (1 / math.log(10))

def ctc_beam_search_with_lm(
    logits,
    processor,
    lm,
    beam_size=3,
    alpha=0.5,
    beta=0.1,
):
    """
    Beam search in model token space, LM scoring in word space.
    """

    vocab_size = logits.size(-1)

    ## beams: list of (token_ids, ctc_log_prob)
    beams = [([], 0.0)]

    T = logits.size(0)

    for t in range(T):
        log_probs = torch.log_softmax(logits[t], dim=-1)

        new_beams = []

        for tokens, ctc_score in beams:
            for v in range(vocab_size):
                new_tokens = tokens + [v]

                # CTC score so far
                new_ctc_score = ctc_score + log_probs[v].item()

                # 🔥 decode to text **only for scoring**, not for generating tokens
                decoded = processor.batch_decode([new_tokens])[0]

                # convert to word tokens for LM
                words = decoded.split()

                # LM score
                lm_s = lm_score(lm, words, alpha) if words else 0.0

                # word insertion bonus
                total_score = new_ctc_score + lm_s + beta * len(words)

                new_beams.append((new_tokens, total_score))

        # prune
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # choose best sequence
    best_tokens, best_score = beams[0]
    best_text = processor.batch_decode([best_tokens], skip_special_tokens=True)[0]

    return best_text, best_score



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True,
                        help="tokenized_data/<lang>/wav2vec2/<split>/<test or val>")
    parser.add_argument("--model", required=True, help="path to wav2vec2 model")
    parser.add_argument("--lm-path", required=True, help="path to nltk LM pickle")
    parser.add_argument("--results-dir", required=False)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.1)

    args = parser.parse_args()

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)

    print("Loading wav2vec2 model...")
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).cuda()
    model.eval()

    print("Loading NLTK LM...")
    lm = load_kenlm_model(args.lm_path)

    print("Loading tokenized dataset...")
    dataset = load_from_disk(args.data_dir)

    preds = []
    golds = []

    for sample in tqdm(dataset):
        input_ids = torch.Tensor(sample["input_values"]).cuda()            # Tensor
        transcript = sample["transcript"]          # Gold text

        golds.append(transcript)

        with torch.no_grad():
            logits = model(input_ids.unsqueeze(0)).logits[0]

        decoded, _ = ctc_beam_search_with_lm(
            logits, processor, lm,
            beam_size=args.beam_size,
            alpha=args.alpha,
            beta=args.beta
        )

        preds.append(decoded)

    print("Computing metrics...")
    if not args.results_dir:
        metrics = compute_metrics_openai(preds, golds)
    else:
        metrics = compute_metrics_openai(preds, golds, save_results=True, results_folder=args.results_dir)

    print(metrics)

if __name__ == "__main__":
    main()
