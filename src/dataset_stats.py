import pandas as pd
import argparse
from pathlib import Path
from collections import Counter

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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform miscellaneous analysis on the datasets."
    )
    
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language to analyse, the directory analyzed will be processed_data/<lang>/",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    lang = args.lang
    
    lang_path = Path(f"processed_data/{lang}")
    create_wordforms(lang_path)


if __name__=='__main__':
    main()