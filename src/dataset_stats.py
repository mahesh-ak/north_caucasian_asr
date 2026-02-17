import pandas as pd
import argparse
from pathlib import Path
from collections import Counter
import json
from scipy.stats import pearsonr

def create_wordforms(lang_path):
    dfs_list = []
    for subfolder in lang_path.iterdir():
        if subfolder.is_dir():
            df_path = subfolder / "dataset.csv"
            if df_path.exists():
                dfs_list.append(pd.read_csv(str(df_path)))
    
    df = pd.concat(dfs_list)
    text = ' '.join(df['transcript'].tolist()).split()
    word_forms = Counter(text)
    wf_dict = {'word_forms':[], 'frequency': []}
    for k,v in word_forms.items():
        wf_dict['word_forms'].append(k)
        wf_dict['frequency'].append(v)
    wf_df = pd.DataFrame(wf_dict)
    wf_df.sort_values(by=['word_forms'], inplace=True)
    wf_df.to_csv(str(lang_path / f"{lang_path.parts[-1]}_word_forms.tsv"), sep='\t', index=False)

    
## read results/Archi/<model>/split/stats.json for <model> in ['whisper-large-v3', 'wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns']
## the dict containing f1 scores of phonemes is in entry "char_stats" and sub-entry "classification_report"
## each phoneme has entries "f1-score", "support", look at f1_scores of phonemes with support < 10^(1.6) and find correlation with those of their base forms with support > 10^(2)
## a base form for example of p' is p, arrive here like if phoneme[:k] in some list, keep on decreasing k until a match is found
def analyze_fewshot_results():
    models = ['whisper-large-v3', 'wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns']
    print("Language: Archi")
    for model in models:
        with open(f"results/Archi/{model}/split/stats.json") as f:
            stats = json.load(f)
        char_stats = stats["char_stats"]["classification_report"]
        # Analyze the character stats here
        low_support_phonemes = {phoneme: metrics for phoneme, metrics in char_stats.items() if phoneme not in ['accuracy', 'macro avg', 'weighted avg', ' '] and metrics['support'] < 10**1.6}
        high_support_phonemes = {phoneme: metrics for phoneme, metrics in char_stats.items() if phoneme not in ['accuracy', 'macro avg', 'weighted avg', ' '] and metrics['support'] > 10**1.9}
        low_support_f1_scores = []
        high_support_f1_scores = []
        matches = []
        for phoneme, metrics in low_support_phonemes.items():
            base_form = None
            for k in range(len(phoneme)-1, 0, -1):
                if phoneme[:k] in high_support_phonemes:
                    base_form = phoneme[:k]
                    break
            if base_form:
                low_support_f1_scores.append(metrics['f1-score'])
                high_support_f1_scores.append(high_support_phonemes[base_form]['f1-score'])
                matches.append((phoneme, base_form))

        corr, _ = pearsonr(low_support_f1_scores, high_support_f1_scores)
        lucky_phonemes = [p for p,v in low_support_phonemes.items() if v['f1-score'] > 0.8]
        print(f"Model: {model}, Correlation: {round(corr,2)}")
        print("matched low-support (< 10^1.6), high-support (> 10^1.9 = chosen to maximize rho) phoneme pairs:", matches)
        print("low-support phonemes with f1-score > 0.8:", lucky_phonemes)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform miscellaneous analysis on the datasets."
    )
    
    parser.add_argument(
        "--lang",
        type=str,
        required=False,
        help="Language to analyse, the directory analyzed will be processed_data/<lang>/",
    )

    parser.add_argument(
        "--fs-res",
        action='store_true',
        help="analyze the few-shot learned phonemes from results on Archi for specific models: whisper-large-v3 and wav2vec2-large-ipa",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    if args.lang:
        lang = args.lang
    
        lang_path = Path(f"processed_data/{lang}")
        create_wordforms(lang_path)
    
    if args.fs_res:
        analyze_fewshot_results()


if __name__=='__main__':
    main()