# with the results folder structure results/<lang>/<model_name>/<split_name> tabulate wer, cer and char_stats['per'] from stats.json files in each split folder into a csv file
# The columns are: lang, model_name, split_name, wer, cer, per
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import evaluate
from lingpy import rc
import numpy as np
from utils import space_separate
from scipy.stats import wilcoxon


def pvalue_matrix(methods, metric="wer"):
    """
    Compute pairwise Wilcoxon signed-rank test p-value matrix.

    Args:
        methods: list of dicts with keys ["model", "wer", "cer"]
        metric: "wer" or "cer"

    Returns:
        pandas.DataFrame (models x models)
    """
    names = [m["model"] for m in methods]
    n = len(methods)

    mat = np.full((n, n), '-', dtype=object)

    for i in range(n):
        for j in range(i + 1, n):
            x = np.array(methods[i][metric])
            y = np.array(methods[j][metric])

            assert len(x) == len(y), "Paired samples must have same length"

            # Remove ties (required by Wilcoxon)
            diff = x - y
            mask = diff != 0
            x_f = x[mask]
            y_f = y[mask]

            if len(x_f) == 0:
                p = 1.0
            else:
                _, p = wilcoxon(x_f, y_f, alternative="two-sided")

            if p < 0.001:
                p = '< 1e-3'
            else:
                p = f"{p:.3f}"
            mat[i, j] = p
            mat[j, i] = p

    return pd.DataFrame(mat, index=names, columns=names)


dolgo = rc("dolgo")

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def convert_to_dolgo(sent):
    sent_tokens = space_separate(sent)
    out_tokens = []
    missing = []
    for tok in sent_tokens:
        if tok in [' ', '.',',']:
            #out_tokens.append(tok)
            pass
        elif tok in dolgo.converter:
            out_tokens.append(dolgo.converter[tok])
        elif len(tok) > 1 and tok[:-1] in dolgo.converter:
            out_tokens.append(dolgo.converter[tok[:-1]])
        elif len(tok) > 2 and tok[:-2] in dolgo.converter:
            out_tokens.append(dolgo.converter[tok[:-2]])
        elif len(tok) > 3 and tok[:-3] in dolgo.converter:
            out_tokens.append(dolgo.converter[tok[:-3]])
        else:
            out_tokens.append(tok)
            missing.append(tok)
    #if len(missing) > 0:
        #print(missing)
    return ''.join(out_tokens)

def tabulate_results(results_root="results", output_csv="results/tabulated_results.csv"):
    results_root = Path(results_root)
    all_results = []
    to_compare = ['custom_split', 'custom_split_lm', 'custom_split_noavg', 'custom_split_noinit', 'wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns_split','whisper-large-v3_split', 'Qwen2-Audio-7B-Instruct_split', 'Qwen2.5-Omni-7B_split']
    
    # iterate over languages
    for lang_dir in results_root.iterdir():
        if lang_dir.is_dir():
            lang = lang_dir.name
            pval_methods = []
            # iterate over model names
            print(lang)
            for model_dir in tqdm(lang_dir.iterdir()):
                if model_dir.is_dir():
                    model_name = model_dir.name
                    # iterate over split names
                    for split_dir in model_dir.iterdir():
                        if split_dir.is_dir():
                            split_name = split_dir.name
                            stats_file = split_dir / "stats.json"
                            preds_file = split_dir / "predictions.tsv"
                            if stats_file.exists():
                                with open(stats_file, "r", encoding="utf-8") as f:
                                    stats = json.load(f)
                                    wer = stats.get("wer", None)
                                    cer = stats.get("cer", None)
                                    per = stats.get("char_stats", {}).get("per", None)
                                    
                                    all_results.append({
                                        "lang": lang,
                                        "model_name": model_name,
                                        "split_name": split_name,
                                        "wer": round(wer, 3),
                                        "cer": round(cer,3),
                                        "per": round(per,3)
                                    })
                            else:
                                print(f"Warning: {stats_file} does not exist.")
                            if preds_file.exists():
                                preds_df = pd.read_csv(preds_file, sep='\t')

                                if model_name + '_' + split_name.replace('split1', 'split') in to_compare:
                                    preds_df['wer'] = preds_df.apply(lambda x: wer_metric.compute(predictions=[x['Prediction']], references=[x['Reference']]), axis=1)
                                    preds_df['cer'] = preds_df.apply(lambda x: cer_metric.compute(predictions=[x['Prediction']], references=[x['Reference']]), axis=1)
                                    pval_methods.append({
                                        "model": f"{model_name}_{split_name}",
                                        "wer": preds_df['wer'].to_list(),
                                        "cer": preds_df['cer'].to_list()
                                    })

                                ## compute Dolgopolsky-class error rate
                                for col in ['Reference', 'Prediction']:
                                    preds_df[col] = preds_df.apply(lambda x: convert_to_dolgo(x[col]),axis=1)
                                refs = preds_df["Reference"].to_list()
                                preds = preds_df["Prediction"].to_list()
                                der = cer_metric.compute(predictions=preds, references=refs)
                                all_results[-1]['der'] = round(der, 3)
            ## compute p-vals
            if len(pval_methods) == 0:
                continue
            pval_methods.sort(key= lambda x: to_compare.index(x['model'].replace('split1','split')))
            pval_wer = pvalue_matrix(pval_methods, metric='wer')
            pval_cer = pvalue_matrix(pval_methods, metric='cer')
            pval_wer.to_csv(f"results/pval_{lang}_wer.tsv", sep='\t')
            pval_cer.to_csv(f"results/pval_{lang}_cer.tsv", sep='\t')

    
    df = pd.DataFrame(all_results)
    # sort by lang, model_name, split_name incrementally
    df = df.sort_values(by=["lang", "model_name", "split_name"])
    df.to_csv(output_csv, index=False)
    print(f"Tabulated results saved to {output_csv}")
    print(df)
    
if __name__ == "__main__":
    tabulate_results()