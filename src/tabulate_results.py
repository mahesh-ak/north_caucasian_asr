# with the results folder structure results/<lang>/<model_name>/<split_name> tabulate wer, cer and char_stats['per'] from stats.json files in each split folder into a csv file
# The columns are: lang, model_name, split_name, wer, cer, per
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import evaluate
from lingpy import rc
from utils import space_separate

dolgo = rc("dolgo")

cer_metric = evaluate.load("cer")

def convert_to_dolgo(sent):
    sent_tokens = space_separate(sent)
    out_tokens = []
    missing = []
    for tok in sent_tokens:
        if tok in [' ', '.']:
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
            missing.append(tok)
    if len(missing) > 0:
        print(missing)
    return ''.join(out_tokens)

def tabulate_results(results_root="results", output_csv="results/tabulated_results.csv"):
    results_root = Path(results_root)
    all_results = []
    
    # iterate over languages
    for lang_dir in results_root.iterdir():
        if lang_dir.is_dir():
            lang = lang_dir.name
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
                                for col in ['Reference', 'Prediction']:
                                    preds_df[col] = preds_df.apply(lambda x: convert_to_dolgo(x[col]),axis=1)
                                refs = preds_df["Reference"].to_list()
                                preds = preds_df["Prediction"].to_list()
                                der = cer_metric.compute(predictions=preds, references=refs)
                                all_results[-1]['der'] = round(der, 3)

    
    df = pd.DataFrame(all_results)
    # sort by lang, model_name, split_name incrementally
    df = df.sort_values(by=["lang", "model_name", "split_name"])
    df.to_csv(output_csv, index=False)
    print(f"Tabulated results saved to {output_csv}")
    print(df)
    
if __name__ == "__main__":
    tabulate_results()