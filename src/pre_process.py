import argparse
from pathlib import Path
import os
import sys
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2CTCTokenizer, AutoProcessor
import librosa
from datasets import Dataset

# Function to process data
def prepare_dataset(batch, processor, word_delimiter_token):
    # Load and resample audio data
    audio = batch["audio_path"]
    # Check if audio_path exists and is a file
    if not os.path.isfile(audio):
        raise FileNotFoundError(f"Audio file {audio} not found.")
    
    # Convert audio to array
    audio_array, _ = librosa.load(audio)
    
    # Process audio, data created by data.py is already at 16kHz
    batch["input_values"] = processor(
        audio_array, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values[0]
    
    # Process transcript for IPA, first condense multiple spaces to single space
    batch["transcript"] = ' '.join(batch["transcript"].strip().split())
    batch["labels"] = batch["transcript"].replace(' ', word_delimiter_token)
    with processor.as_target_processor():
        batch["labels"] = processor(batch["labels"]).input_ids
    
    return batch

def tokenize_transcripts(data_dir, processor, output_dir, word_delimiter_token="|"):
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
            dfs.append(df)
        
        if not dfs: 
            print(f"No valid CSV files found for split {split}. Skipping.")
            continue
        
        all_transcripts_df = pd.concat(dfs, ignore_index=True)
        if split == 'train':
            # Shuffle and take 95% for train, 5% for validation
            train_df, val_df = train_test_split(all_transcripts_df, test_size=0.05, random_state=42)
            train_dataset = Dataset.from_pandas(train_df)
            val_dataset = Dataset.from_pandas(val_df)
            # Process datasets
            datasets = [("train", train_dataset), ("validation", val_dataset)]
        else:
            test_dataset = Dataset.from_pandas(all_transcripts_df)
            datasets = [(split, test_dataset)]
        
        
        for ds_name, dataset in datasets:
            # Map the dataset with the processor, retain these columns: textgrid_path, tier_name, interval_id for traceability or remove audio_path and transcript
            dataset = dataset.map(lambda batch: prepare_dataset(batch, processor, word_delimiter_token), remove_columns=[col for col in ["audio_path"] if col in dataset.column_names], num_proc=os.cpu_count())
            # Save the processed dataset
            dataset.save_to_disk(output_dir / ds_name)
            print(f"Saved {ds_name} dataset to {output_dir / ds_name}")
    

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
        "--new-tokenizer",
        type=str,
        required=False,
        help="Name for the new tokenizer to be created based on vocab.json in data_dir.",
    )
    
    parser.add_argument(
        "--word-delimiter-token",
        type=str,
        default="|",
        help="Token to use as word delimiter in transcripts.",
    )

    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else "tokenized_data" / Path(*data_dir.parts[1:])
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.processor)
    # Load the tokenizer
    if args.new_tokenizer:
        print("Creating a new tokenizer based on vocab.json in data_dir")
        ## create a tokenizer based on vocab.json in data_dir
        vocab_file = data_dir / "vocab.json"
        if not vocab_file.exists():
            print(f"Vocab file {vocab_file} does not exist. Exiting.")
            sys.exit(1)
        ## As of now only Wav2Vec2CTCTokenizer is supported
        tokenizer = Wav2Vec2CTCTokenizer(vocab_file=vocab_file, 
                                        word_delimiter_token=args.word_delimiter_token, 
                                        unk_token="<unk>",
                                        pad_token="<pad>",
                                        bos_token="<s>",
                                        eos_token="</s>",
                                        model_max_length=1024)        

        processor.tokenizer = tokenizer
        ## save processor to models / data_dir.parts[1:] / args.new_tokenizer
        save_dir = Path("models") / Path(*data_dir.parts[1:]) / args.new_tokenizer
        save_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(save_dir)
        print(f"Saved new processor to {save_dir}")
        
        ## output_dir should contain a subfolder with name args.new_tokenizer
        output_dir = output_dir / args.new_tokenizer
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        ## processor is of format username/model_name, use only model_name as subfolder
        model_name = args.processor.split("/")[-1]
        output_dir = output_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
    print(f"Using tokenizer with vocab size: {processor.tokenizer.vocab_size}")
    tokenize_transcripts(data_dir, processor, output_dir, args.word_delimiter_token)
    
if __name__ == "__main__":
    main()
