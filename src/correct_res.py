from prompt_llm import compute_metrics_openai
from utils import invert_mapping
from pathlib import Path
import json
import argparse
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="Folder with inference results to be corrected")
    p.add_argument("--ipa2cyrl", required=False, help="Path to ipa2cyrl.json mapping file")
    return p.parse_args()



def correct_res_openai(results_dir, cyrl2ipa=None):
    res_df = pd.read_csv(Path(results_dir) / "predictions.tsv", sep="\t")
    pred_str = res_df["Prediction"].tolist()
    ref_str = res_df["Reference"].tolist()
    eval_dict = compute_metrics_openai(pred_str=pred_str, ref_str=ref_str, save_results=True, results_folder=str(results_dir), cyrl2ipa=cyrl2ipa)
    print(eval_dict)

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    if args.ipa2cyrl:
        with open(args.ipa2cyrl, "r", encoding="utf-8") as f:
            ipa2cyrl = json.load(f)
        cyrl2ipa = invert_mapping(ipa2cyrl)
    
    correct_res_openai(results_dir, cyrl2ipa=cyrl2ipa)

if __name__ == "__main__":
    main()

     
