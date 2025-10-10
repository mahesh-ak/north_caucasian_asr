# with the results folder structure results/<lang>/<model_name>/<split_name> tabulate wer, cer and char_stats['per'] from stats.json files in each split folder into a csv file
# The columns are: lang, model_name, split_name, wer, cer, per
import os
import json
import pandas as pd
from pathlib import Path

def tabulate_results(results_root="results", output_csv="results/tabulated_results.csv"):
    results_root = Path(results_root)
    all_results = []
    
    # iterate over languages
    for lang_dir in results_root.iterdir():
        if lang_dir.is_dir():
            lang = lang_dir.name
            # iterate over model names
            for model_dir in lang_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    # iterate over split names
                    for split_dir in model_dir.iterdir():
                        if split_dir.is_dir():
                            split_name = split_dir.name
                            stats_file = split_dir / "stats.json"
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
    
    df = pd.DataFrame(all_results)
    # sort by lang, model_name, split_name incrementally
    df = df.sort_values(by=["lang", "model_name", "split_name"])
    df.to_csv(output_csv, index=False)
    print(f"Tabulated results saved to {output_csv}")
    print(df)
    
if __name__ == "__main__":
    tabulate_results()