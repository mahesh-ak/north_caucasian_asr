# with the results folder structure results/<lang>/<model_name>/<split_name> tabulate wer, cer and char_stats['per'] from stats.json files in each split folder into a csv file
# The columns are: lang, model_name, split_name, wer, cer, per
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import evaluate
from lingpy import rc
import numpy as np
import matplotlib.pyplot as plt
from utils import space_separate
from scipy.stats import wilcoxon
from scipy.optimize import curve_fit
from adjustText import adjust_text

def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

base_vowels = {'a', 'e', 'i', 'o', 'u', 'ɨ', 'ə', 'y'}
long = 'ː'
pharyn = 'ˤ'

base_consonants = {'b', 'd', 'd͡ʒ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 's', 't', 't͡s', 't͡ʃ', 'w', 'x', 'ɢ', 'ʁ', 'ʃ', 'χ', 'ɬ', 'ʒ', 'ʟ', 'ɣ', 'ʔ', 'ʕ', 'z', 'r', 'j', 'ħ', 'ɮ'}
lab = 'ʷ'
eject = 'ʼ'
pal = 'ʲ'

def phoneme_category_stats(char_stats_dict):
    ph_cat_stats = {'V': [], 'C': []}
    all_marks = [long, pharyn, lab, eject, pal]
    for k, v in char_stats_dict.items():
        base_form = k
        suffix = ''
        for i, c in enumerate(list(k)):
            if c in all_marks:
                suffix = str(k[i:])
                base_form = str(k[:i])
                break
        if base_form in base_vowels:
            cat_str = 'V' + suffix
        elif base_form in base_consonants:
            cat_str = 'C' + suffix
        else:
            continue
        if v['support'] == 0:
            continue

        if cat_str not in ph_cat_stats:
            ph_cat_stats[cat_str] = [v['f1-score']]
        else:
            ph_cat_stats[cat_str].append(v['f1-score'])
    return {k: (round(np.mean(v),3), round(np.std(v),3), len(v)) for k,v in ph_cat_stats.items()}
        
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


dolgo = rc("asjp")

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def convert_to_dolgo(sent):
    sent_tokens = space_separate(sent)
    out_tokens = []
    missing = []
    for tok in sent_tokens:
        if tok in [' ', '.',',']:
            out_tokens.append(tok)
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
            top_errors = []
            # iterate over model names
            if lang == 'RutulOld':
                continue
            print(lang)
            with open(f"processed_data/{lang}/train_phonemes.json") as fp:
                train_phonemes = json.load(fp)
                train_phonemes = {k: {'f1-score': 1.0, 'support':v} for k,v in train_phonemes.items()}
                phoneme_class_dict = phoneme_category_stats(train_phonemes)
                phoneme_class_lst = list(phoneme_class_dict.keys())
            phoneme_class_lst.sort()

            phoneme_cat_scores = []

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

                                    classification_report = stats.get("char_stats",{}).get("classification_report", None)
                                    f1_scores = [(k, round(v['f1-score'],2), int(v['support'])) for k,v in classification_report.items() if k not in ['<eps>', ' ', 'accuracy', 'macro avg','weighted avg'] and v['support'] > 0]
                                    f1_scores.sort(key=lambda x: x[1])
                                    top_errors.append({'model': model_name+'_'+split_name, 'f1': f1_scores[:25]})
                                    phoneme_cat_scores.append((model_name, split_name, phoneme_category_stats(classification_report)))

                                    # Collect data 
                                    xs = []
                                    ys = []
                                    labels = []
                                    point_sizes = []

                                    for phoneme, stats_p in classification_report.items():
                                        if phoneme in ['<eps>', ' ', 'accuracy', 'macro avg', 'weighted avg']:
                                            continue
                                        if phoneme not in train_phonemes:
                                            continue

                                        train_sup = train_phonemes[phoneme]['support']
                                        test_sup = stats_p.get('support', 0)

                                        if train_sup > 0 and test_sup > 0:
                                            xs.append(train_sup)
                                            ys.append(stats_p['f1-score'])
                                            point_sizes.append(np.log10(test_sup + 1) * 20)  # size proportional to log of test support
                                            labels.append(phoneme)

                                    if len(xs) > 0:
                                        xs = np.array(xs)
                                        ys = np.array(ys)

                                        plt.figure(figsize=(7, 6))
                                        plt.scatter(xs, ys, alpha=0.75, s= point_sizes)

                                        # Fit in log10-support space
                                        x_log = np.log10(xs)

                                        # Initial guesses: max F1, slope, midpoint
                                        p0 = [1.0, 1.0, np.median(x_log)]

                                        params, _ = curve_fit(
                                            logistic,
                                            x_log,
                                            ys,
                                            p0=p0,
                                            maxfev=10000
                                        )

                                        L, k, x0 = params

                                        # Plot fitted curve
                                        x_fit = np.linspace(x_log.min(), x_log.max(), 300)
                                        y_fit = logistic(x_fit, L, k, x0)

                                        plt.plot(10 ** x_fit, y_fit, color="red", linewidth=2, label="Sigmoid fit")
                                        plt.legend()

                                        # Report parameters on plot
                                        plt.text(
                                            0.05,
                                            0.45,
                                            f"L={L:.2f}\nk={k:.2f}\nlog₁₀(x₀)={x0:.2f}",
                                            transform=plt.gca().transAxes,
                                            fontsize=9,
                                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
                                        )
                                        # Log scale for Zipfian phoneme frequencies
                                        plt.xscale('log')

                                        plt.xlabel("Training support (log scale)")
                                        plt.ylabel("F1-score")
                                        plt.title(f"({lang}) {model_name} – {split_name}\nPhoneme learning difficulty")

                                        plt.grid(True, which="both", linestyle="--", alpha=0.3)

                                        # Annotate ALL phonemes
                                        label_texts = []

                                        for x, y, lab in zip(xs, ys, labels):
                                            #lab_txt = plt.annotate(
                                            #    lab,
                                            #    (x, y),
                                            #    fontsize=7,
                                            #    alpha=0.85,
                                            #    textcoords="offset points",
                                            #    xytext=(2, 2)
                                            #)
                                            lab_txt = plt.text(x, y, lab, fontsize=7, alpha=0.85)
                                            label_texts.append(lab_txt)
                                        adjust_text(label_texts) 


                                        # Save plot
                                        out_path = split_dir / "phoneme_f1_vs_train_support.png"
                                        plt.tight_layout()
                                        plt.savefig(out_path, dpi=300)
                                        plt.close()

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
                                all_results[-1]['aer'] = round(der, 3)
            ## compute p-vals
            if len(pval_methods) == 0:
                continue
            pval_methods.sort(key= lambda x: to_compare.index(x['model'].replace('split1','split')))
            pval_wer = pvalue_matrix(pval_methods, metric='wer')
            pval_cer = pvalue_matrix(pval_methods, metric='cer')
            pval_wer.to_csv(f"results/pval_{lang}_wer.tsv", sep='\t')
            pval_cer.to_csv(f"results/pval_{lang}_cer.tsv", sep='\t')

            errors_lines = ['\t'.join(["model"] + [str(num) for num in range(1,26)])]
            for err in top_errors:
                errors_lines.append('\t'.join([err['model']] + [str(sc) for sc in err['f1']]))
            with open(f"results/top_errors_{lang}.tsv",'w',encoding='utf-8') as fp:
                fp.write('\n'.join(errors_lines))

            phoneme_cat_lines = ['\t'.join(["model", "split"] + [f"{p} ({phoneme_cat_scores[0][2][p][2]}/{phoneme_class_dict[p][2]})" for p in phoneme_class_lst])]
            for ph_cat in phoneme_cat_scores:
                phoneme_cat_lines.append('\t'.join([ph_cat[0], ph_cat[1]] + [str(ph_cat[2][p][0]) for p in phoneme_class_lst]))
            with open(f"results/phoneme_catwise_{lang}.tsv",'w',encoding='utf-8') as fp:
                fp.write('\n'.join(phoneme_cat_lines))

    df = pd.DataFrame(all_results)
    # sort by lang, model_name, split_name incrementally
    df = df.sort_values(by=["lang", "model_name", "split_name"])
    df.to_csv(output_csv, index=False)
    print(f"Tabulated results saved to {output_csv}")
    print(df)
    
if __name__ == "__main__":
    tabulate_results()