import argparse
from pathlib import Path
import os
import sys
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, AutoProcessor

# Function to process data
def prepare_dataset(batch):
    # Load and resample audio data
    audio = batch["audio_path"]
    # Check if audio_path exists and is a file
    if not os.path.isfile(audio):
        # Try to find the file with absolute path
        if not os.path.isabs(audio):
            audio = os.path.join('north_caucasus_data/Transcribed', audio)
    
    # Load audio file
    #with open(audio, "rb") as f:
    #    audio_data = f.read()
    
    # Convert audio to array
    audio_array, _ = librosa.load(audio)#np.frombuffer(audio_data, dtype=np.int16)
    
    # Process audio
    batch["input_values"] = processor(
        audio_array, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values[0]
    
    # Process transcript for IPA
    batch["transcript"] = batch["transcript"].replace(' ','#')
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
    
    return batch

def tokenize_transcripts(data_dir, processor, tokenizer, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # A directory is usually structured with subfolders each containing a dataset.csv
    # Subfolder allocated to train, test should be present in data_dir / split.json with structure {'train': ['subfolder1', 'subfolder2'], 'test': ['subfolder3']}
    split_file = data_dir / "split.json"
    if not split_file.exists():
        print(f"Split file {split_file} does not exist. Exiting.")
        sys.exit(1)
    
    splits = json.load(open(split_file))
    for split, subfolders in splits.items():
        all_transcripts = []
        # create an list of dfs and concatenate all transcripts
        dfs = []
        for subfolder in subfolders:
            csv_path = data_dir / subfolder / "dataset.csv"
            if not csv_path.exists():
                print(f"CSV file {csv_path} does not exist. Skipping.")
                continue
            df = pd.read_csv(csv_path)
            if 'transcript' not in df.columns:
                print(f"'transcript' column not found in {csv_path}. Skipping.")
                continue
            dfs.append(df[['transcript']])
        
        if not dfs: 
            print(f"No valid CSV files found for split {split}. Skipping.")
            continue
        
        all_transcripts_df = pd.concat(dfs, ignore_index=True)
        
    

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize transcripts using a specified tokenizer."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing the dataset CSV files.",
    )
    parser.add_argument(
        "--processor",
        type=str,
        required=True,
        help="Pretrained tokenizer name or path from Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Directory to save the tokenized transcripts.",
    )
    parser.add_argument(
        "--create-new-tokenizer",
        action="store_true",
        help="Flag to create a new tokenizer based on vocab.json in data_dir.",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else "tokenized_data" / Path(*data_dir.parts[1:])
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.processor)
    # Load the tokenizer
    if args.create_new_tokenizer is False:
        tokenizer = processor.tokenizer
    else:
        ## create a tokenizer based on vocab.json in data_dir
        vocab_file = data_dir / "vocab.json"
        if not vocab_file.exists():
            print(f"Vocab file {vocab_file} does not exist. Exiting.")
            sys.exit(1)
        ## As of now only Wav2Vec2CTCTokenizer is supported
        tokenizer = Wav2Vec2CTCTokenizer(vocab_file=vocab_file)        



if __name__ == "__main__":
    main()
