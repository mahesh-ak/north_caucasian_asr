import argparse
from pathlib import Path
import os
import sys
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2CTCTokenizer, AutoProcessor, WhisperTokenizerFast, WhisperFeatureExtractor, WhisperProcessor
import librosa
from datasets import Dataset, concatenate_datasets
import numpy as np
import torch

def ipa_to_cyrillic(text: str, ipa2cyrl: dict) -> str:
    keys = sorted(ipa2cyrl.keys(), key=len, reverse=True)  # longest first
    i, n = 0, len(text)
    out = []

    while i < n:
        matched = False
        for k in keys:
            if text.startswith(k, i):
                out.append(ipa2cyrl[k])
                i += len(k)
                matched = True
                break
        if not matched:                      # pass through unknown chars
            out.append(text[i])
            i += 1

    return "".join(out)

conversations = lambda lang: {
    'qwen_audio': [{"role": "user", "content": [
                        {"type": "audio"},     # audio will be bound by processor(audio=..)
                        {"type": "text", "text": f"Transcribe the audio in {lang} (a North Caucasian language) into IPA (Internation Phonetic Alphabet). Do not translate, interpret, or add punctuation. Output only the phonetic transcription."},
                    ]}],
    'qwen_omni': [ {   
                            "role": "system",
                            "content": [
                                {"type": "text", "text":
                                    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio"},                # binds to processor(..., audio=audios)
                                {"type": "text", "text": f"Transcribe the audio in {lang} (a North Caucasian language) into IPA (Internation Phonetic Alphabet). Do not translate, interpret, or add punctuation. Output only the phonetic transcription."},
                            ],
                }],
    'phi': [
            {
                "role": "user",
                "content": f"<|audio_1|>Transcribe the audio in {lang} (a North Caucasian language) into IPA (International Phonetic Alphabet). Do not translate, interpret, or add punctuation. Output only the phonetic transcription."
            }
        ]

}

# Function to process data
def prepare_dataset(example, processor, word_delimiter_token, mode="wav2vec2", transcriber = None, lang=None, split=None):
    # Load and resample audio data
    audio = example["audio_path"]
    if lang:
        if mode.startswith("qwen") or mode == "phi":
            prompt = conversations(lang)[mode]
    # Check if audio_path exists and is a file
    if not os.path.isfile(audio):
        raise FileNotFoundError(f"Audio file {audio} not found.")
    
    # Convert audio to array
    audio_array, sr = librosa.load(audio, sr=16000)
    
    # clean + tokenize text
    example["transcript"] = " ".join(example["transcript"].strip().split())
    text = example["transcript"].replace(" ", word_delimiter_token)
    if transcriber:
        text = ipa_to_cyrillic(text, transcriber)
        
    if mode == "wav2vec2":
        example["input_values"] = processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_values[0]

        with processor.as_target_processor():
            example["labels"] = processor(text).input_ids
    elif mode == "whisper":
        example["input_features"] = processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features[0]

        # Whisper special tokens
        start_token = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        lang_token  = processor.tokenizer.convert_tokens_to_ids("<|ru|>")      # closest language
        task_token  = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        no_ts_token = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

        tokens = processor.tokenizer(
            text,
            add_special_tokens=False
        ).input_ids

        # prepend special whisper tokens
        example["labels"] = [start_token, lang_token, task_token, no_ts_token] + tokens
    elif "qwen" in mode:
        
        example["audio_array"] = audio_array.astype(np.float32)

        if split == 'train':
            prompt.append({"role": "assistant", "content": [{"type": "text", "text": text}]})  # for generation position
            example["prompts"] = processor.apply_chat_template(prompt, add_generation_prompt=False, tokenize=False)
            example["labels"] = text
        else:
            example["prompts"] = processor.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)

        

    elif mode == "phi":
        # Convert audio path to waveform & tokenize prompt via chat-template
        prompt_ids = processor.apply_chat_template(
            prompt,
            add_generation_prompt=True,   # ensures <assistant> position for generation/loss
            tokenize=False,
        )

        inputs = processor(
            text=prompt_ids,
            audios=[(audio_array.astype(np.float32), 16000)],
            return_tensors="pt",
        )

        # input_ids for model forward()
        example = {k: v[0] for k, v in inputs.items() if v is not None and len(v) > 0}

        # Tokenize answer (transcript) as labels
        answer_ids = processor.tokenizer(text, add_special_tokens=True).input_ids

        if split == "train":
            # prompt_ids first, answer_ids after
            prompt_ids = example["input_ids"]

            full_ids = torch.tensor(
                prompt_ids.tolist() + answer_ids,
                dtype=torch.long
            )

            # labels = ignore prompt, supervise only answer tokens
            labels = torch.tensor(
                [-100] * len(prompt_ids) + answer_ids,
                dtype=torch.long
            )

            example["input_ids"] = full_ids
            example["labels"] = labels
        else:
            # evaluation mode: no concatenation
            example["labels"] = torch.tensor(answer_ids, dtype=torch.long)

    return example

def tokenize_transcripts(data_dir, processor, output_dir, split_file, mode, word_delimiter_token="|", transcriber = None):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    # A directory is usually structured with subfolders each containing a dataset.csv
    # Subfolder allocated to train, test should be present in data_dir / split.json with structure {'train': ['subfolder1', 'subfolder2'], 'test': ['subfolder3']}
    split_file = Path(split_file)
    if not split_file.exists():
        print(f"Split file {split_file} does not exist. Exiting.")
        sys.exit(1)
    
    splits = json.load(open(split_file))
    # get the name without extension and preceeding path, e.g. data/Rutul/ --->
    split_name = split_file.stem
    output_dir = output_dir / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lang = data_dir.parts[-1]  # assuming data_dir is like data/Rutul, get 'Rutul' as language name
    print(f"Processing language: {lang}")
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
            train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
            val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
            # Process datasets
            datas = [("train", train_dataset), ("validation", val_dataset)]
        else:
            test_dataset = Dataset.from_pandas(all_transcripts_df, preserve_index=False)
            datas = [(split, test_dataset)]

        for ds_name, dataset in datas:
            # Map the dataset with the processor, retain these columns: textgrid_path, tier_name, interval_id for traceability or remove audio_path and transcript
            batch_size = 500  # or tune depending on memory
            processed_chunks = []

            for start_idx in range(0, len(dataset), batch_size):
                end_idx = min(start_idx + batch_size, len(dataset))
                chunk = dataset.select(range(start_idx, end_idx))
    
                chunk = chunk.map(
                    lambda example: prepare_dataset(
                        example,
                        processor,
                        word_delimiter_token,
                        mode=mode,
                        transcriber=transcriber,
                        lang=lang,
                        split=ds_name
                    ),
                    remove_columns=[col for col in ["audio_path"] if col in chunk.column_names],
                    num_proc=None,  # single process avoids multiprocessing issues
                    batched=False,
                    load_from_cache_file=False
                )
                processed_chunks.append(chunk)

            # Concatenate all processed chunks back into a single dataset
            dataset = concatenate_datasets(processed_chunks)

            # Save the processed dataset to output_dir / ds_name
            dataset.save_to_disk(output_dir / ds_name)
            print(f"Saved {ds_name} dataset with {len(dataset)} samples to {output_dir / ds_name}")
    

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
        "--split-file",
        type=str,
        default="data/split1.json",
        help="Path to the JSON file defining train/test splits.",
    )
    
    parser.add_argument(
        "--word-delimiter-token",
        type=str,
        default="|",
        help="Token to use as word delimiter in transcripts.",
    )

    parser.add_argument(
        "--ipa-to-cyrillic",
        type=str,
        default=None,
        help="Path to JSON file mapping IPA symbols to Cyrillic characters.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else "tokenized_data" / Path(*data_dir.parts[1:])
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.processor, trust_remote_code=True)
    mode = ""
    is_whisper = 'whisper' in processor.__class__.__name__.lower()
    is_qwen = 'qwen' in processor.__class__.__name__.lower()
    is_phi = 'phi' in processor.__class__.__name__.lower()
    mode = "whisper" if is_whisper else mode
    if is_whisper:
        word_delimiter_token = " "
    elif is_qwen:
        mode = "qwen_audio"
        if 'omni' in processor.__class__.__name__.lower():
            mode = "qwen_omni"
        word_delimiter_token = " "  
    elif is_phi:
        mode = "phi"
        word_delimiter_token = " "
    else:
        word_delimiter_token = args.word_delimiter_token

    is_wav2vec2 = 'wav2vec2' in processor.__class__.__name__.lower()
    mode = "wav2vec2" if is_wav2vec2 else mode
    if not mode:
        print("Unsupported processor type. Only Whisper and Wav2Vec2 are supported.")
        sys.exit(1)
    # Load the tokenizer only for wav2vec2 mode
    if args.new_tokenizer and mode in ["wav2vec2"]:
        print("Creating a new tokenizer based on vocab.json in data_dir")
        vocab_file = data_dir / "vocab.json"
        if not vocab_file.exists():
            print(f"Vocab file {vocab_file} does not exist. Exiting.")
            sys.exit(1)

        # -------------------------------
        # Existing Wav2Vec2 tokenizer
        # -------------------------------
        mode = "wav2vec2"
        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file=vocab_file,
            word_delimiter_token=args.word_delimiter_token,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            model_max_length=1024,
        )
        processor.tokenizer = tokenizer

        # -------------------------------
        # Save processor
        # -------------------------------
        save_dir = (
            Path("models")
            / Path(*data_dir.parts[1:])
            / args.new_tokenizer
            / Path(args.split_file).stem
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        processor.save_pretrained(save_dir)
        print(f"Saved new processor to {save_dir}")

        output_dir = output_dir / args.new_tokenizer
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        ## processor is of format username/model_name, use only model_name as subfolder
        model_name = args.processor.split("/")[-1]
        output_dir = output_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
   
    transcriber = None 
    if args.ipa_to_cyrillic:
        transcriber = json.load(open(args.ipa_to_cyrillic, "r", encoding="utf-8"))
        print(f"Loaded IPA to Cyrillic mapping from {args.ipa_to_cyrillic}")
        
    print(f"Using tokenizer with vocab size: {processor.tokenizer.vocab_size}")
    tokenize_transcripts(data_dir, processor, output_dir, args.split_file, mode, word_delimiter_token, transcriber=transcriber)
    
if __name__ == "__main__":
    main()
