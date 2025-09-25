import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import csv
from pydub import AudioSegment
from textgrid import TextGrid
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer
import random
import json

cyrillic = "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯяр"

class CorrMap:
    """Class to handle character correction mapping."""
    
    ignore = ['cyrillic', 'mixed', 'unclear']
    delim = ['delimiter', 'glossing']
    punct = ['punctuation']
    
    def __init__(self):
        self.map = []

    def create_map(self, row):
        src = row['src']
        dst = row['dst']
        op = row['type']

        if op not in self.ignore:
            if op in self.punct:
                dst = ' '
            elif op in self.delim:
                dst = ''

            self.map.append((src, dst))

    def correct(self, txt):
        for src, dst in self.map:
            txt = txt.replace(src, dst)
        txt = txt.replace('"','')
        txt = ' '.join(txt.split())
        return txt.strip()



def TextGrid_to_Wav(data_folder, output_audio_folder, corr_map, tier_names):
    # Convert TextGrid annotations to segmented WAV files and create a CSV mapping file.
    # CSV file to store mapping from segment file to transcript
    csv_file = os.path.join(output_audio_folder, "dataset.csv")

    # if output_audio_folder/segments is not empty, delete all files in it
    if os.path.exists(os.path.join(output_audio_folder, "segments")) and len(os.listdir(os.path.join(output_audio_folder, "segments"))) > 0:
        print(f"Output folder {os.path.join(output_audio_folder, 'segments')} is not empty. Deleting all files in it.")
        for f in os.listdir(os.path.join(output_audio_folder, "segments")):
            os.remove(os.path.join(output_audio_folder, "segments", f))
            
    with open(csv_file, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "transcript"])  # header row

        # Process each TextGrid file in the folder
        
        # tier names are dependent on the language and annotation scheme
        # hence we try to infer it from the first file by printing the contents and use human in the loop
        # if somewhere down the line, this tier name is not found, we again print the tier names and ask for human input
        
        for file in tqdm(os.listdir(data_folder)):
            if file.lower().endswith(".textgrid"):
                base_name = os.path.splitext(file)[0]
                textgrid_path = os.path.join(data_folder, file)
                wav_path = os.path.join(data_folder, base_name + ".wav")
                
                if not os.path.exists(wav_path):
                    wav_path = os.path.join(data_folder, base_name + ".WAV")
                    if not os.path.exists(wav_path):
                        print(f"WAV file not found for {textgrid_path}")
                        continue

                # Load the corresponding audio
                audio = AudioSegment.from_file(wav_path, format="wav").set_frame_rate(16000)

                # Load the TextGrid annotations
                tg = TextGrid.fromFile(textgrid_path)

                if len(tier_names) == 0: # first file
                    ## print the tier names, contents and index and ask for human input as index of the tier to use
                    ## also use -1 as index to skip this file
                    print(f"Tier names in {textgrid_path}:")
                    for idx, tier in enumerate(tg.tiers):
                        print(f"{idx}: {tier.name}")
                        ## print first 3 contents concatenated
                        contents = ' | '.join([interval.mark for interval in tier.intervals[:3]])
                        print(f"   Contents: {contents} ...")
                        
                    tier_idxs = input("Enter the index of the tier to use for transcription (separated by , in case of multiple): ")
                    ## convert to list of int
                    tier_idxs = [int(x) for x in tier_idxs.split(',')]
                    if -1 in tier_idxs:
                        print(f"Skipping file {textgrid_path}")
                        continue
                    for tier_idx in tier_idxs:
                        tier_names.append(tg.tiers[tier_idx].name)
                        
                ## take intersection of tier_names and tg.tiers as current_tier_names
                current_tier_names = [tier.name for tier in tg.tiers if tier.name in tier_names]
                if len(current_tier_names) == 0: # no intersection
                    print(f"Tier names in {textgrid_path}:")
                    for idx, tier in enumerate(tg.tiers):
                        print(f"{idx}: {tier.name}")
                        ## print first 3 contents concatenated
                        contents = ' | '.join([interval.mark for interval in tier.intervals[:3]])
                        print(f"   Contents: {contents} ...")
                    tier_idxs = input("Enter the index of the tier to use for transcription (separated by , in case of multiple): ")
                    ## convert to list of int
                    tier_idxs = [int(x) for x in tier_idxs.split(',')]
                    if -1 in tier_idxs:
                        print(f"Skipping file {textgrid_path}")
                        continue
                    for tier_idx in tier_idxs:
                        tier_names.append(tg.tiers[tier_idx].name)
                        current_tier_names.append(tg.tiers[tier_idx].name)

                else:
                    if len(current_tier_names) > 1:
                        print(f"Warning! Multiple matching tier names found in {textgrid_path}: {current_tier_names}")
                        
                for tier in tg.tiers:
                    if not tier.name in current_tier_names:
                        continue
                    segment_index = 0
                    for interval in tier.intervals:
                        # Use the 'mark' attribute (or 'text' if your TextGrid uses that field)
                        transcript = interval.mark.strip()
                        if corr_map:
                            transcript = corr_map.correct(transcript)
                            
                        if transcript == "":
                            continue  # skip unannotated or mute segments

                        # Extract the segment from the audio (convert seconds to milliseconds)
                        start_ms = int(interval.minTime * 1000)
                        end_ms = int(interval.maxTime * 1000)
                        if (end_ms - start_ms) < 0.15*1000:
                            print(f"Skipped: {base_name}_segment{segment_index}")
                            continue
                        segment_audio = audio[start_ms:end_ms]

                        # Save the segment as a new wav file
                        segment_filename = f"{base_name}_segment{segment_index}.wav"
                        segment_path = os.path.join(output_audio_folder, "segments", segment_filename)
                        segment_audio.export(segment_path, format="wav")

                        # Write the file path and transcript to CSV
                        writer.writerow([segment_path, transcript])
                        segment_index += 1

    print(f"Dataset preparation complete. CSV saved to {csv_file}")
    return tier_names

def space_separate(sent):
    # Separate phonemes in a given sentence
    phonemes = []

    i = 0
    while i < len(sent):
        if i < len(sent)-1:
            if sent[i+1] in ['ʲ', 'ʷ', 'ʼ', 'ː','ˤ', "'"]:
                if i < len(sent) - 2 and sent[i+2] in ['ʲ', 'ʷ', 'ʼ', 'ː','ˤ', "'"]:
                    phonemes.append(sent[i]+sent[i+1]+sent[i+2])
                    i += 3
                else:
                    phonemes.append(sent[i]+sent[i+1])
                    i += 2
                continue
            elif sent[i+1] in ['͡']:
                if i < len(sent) - 3 and sent[i+3] in ['ʲ', 'ʷ', 'ʼ', 'ː','ˤ', "'"]:
                    phonemes.append(sent[i]+sent[i+1]+sent[i+2]+sent[i+3])
                    i += 4
                else:
                    phonemes.append(sent[i]+sent[i+1]+sent[i+2])
                    i += 3
                continue
        phonemes.append(sent[i])
        if sent[i] in ['ʲ', 'ʷ', 'ʼ', 'ː','ˤ', "'"]:
            print('\t',sent)
        i += 1
    return phonemes


def GenerateCharMap(tokenizer_name, output_path, output_audio_folder):
    # Generate a character mapping file based on the tokenizer and dataset
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    gold = set(tokenizer.vocab.keys())
    
    # Read the dataset.csv file from each subfolder of output_audio_folder and concatenate them
    dataset_df = pd.DataFrame(columns=['audio_path', 'transcript'])
    for subfolder in Path(output_audio_folder).iterdir():
        if subfolder.is_dir():
            csv_path = subfolder / "dataset.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                dataset_df = pd.concat([dataset_df, df], ignore_index=True)

    # Get all unique characters in the transcripts
    chars = []
    for sent in dataset_df['transcript']:
        chars += space_separate(sent)
    IPA_CHARS = sorted(list(set(chars).difference(set(cyrillic))))
    diff_char_counts = Counter(c for c in chars if tokenizer.unk_token_id in tokenizer(c)['input_ids'] and c not in cyrillic) 
    
    # Create a DataFrame for the character mapping
    mapping = {'src': [], 'dst': ['']*len(diff_char_counts), 'type': ['']*len(diff_char_counts), 'count': [], 'sample_wordforms':[]}
    for k,v in diff_char_counts.items():
        mapping['src'].append(k)
        mapping['count'].append(v)
        words = dataset_df[dataset_df['transcript'].str.contains(k, regex=False)]
        # Get unique words containing the character
        words = words.apply(lambda x: [y for y in x['transcript'].split() if k in y], axis=1).tolist()
        unique_words = set()
        for w_lst in words:
            unique_words = unique_words.union(set(w_lst))
        mapping['sample_wordforms'].append(' '.join(unique_words))

    mapping_df = pd.DataFrame(mapping)
    mapping_df = mapping_df.sort_values(by='count', ascending=False)
    mapping_df.to_csv(output_path, sep="\t", index=False)
    print(f"Character mapping file saved to {output_path}")
    return IPA_CHARS
    
    
def parse_args():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Prepare data for North Caucasian ASR project"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=False,
        help="Path to the directory containing the data files",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Path to the output directory where processed data will be saved",
    )

    parser.add_argument(
        "--char-map-file",
        type=str,
        required=False,
        help="Path to the character mapping file",
    )
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        help="Tokenizer to use for processing text",
    )

    parser.add_argument(
        "--tier-names-file",
        type=str,
        required=False,
        help="Path to a file containing tier names (one per line)",
    )
    
    return parser.parse_args()


    
def main():
    """Main function to process data."""
    args = parse_args()
   
    if args.data_dir: 
        data_dir = Path(args.data_dir)
   
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("processed_data") / Path(*data_dir.parts[1:])
    

    # Load character mapping if provided
    if args.char_map_file:
        char_map_path = Path(args.char_map_file)
        if not char_map_path.exists():
            print(f"Character map file {char_map_path} does not exist.")
            sys.exit(1)
        
        char_map_df = pd.read_csv(char_map_path, sep="\t")
        char_map_df.dropna(subset=['count'], inplace=True)
        corr_map = CorrMap()
        char_map_df.apply(corr_map.create_map, axis=1)
    else:
        corr_map = None
    
    ## data_dir contains subfolders, we process each subfolder by looping over them
    if args.data_dir:
        # maintain a common tier names pool across a run
        if args.tier_names_file:
            tier_names_path = Path(args.tier_names_file)
            if not tier_names_path.exists():
                print(f"Tier names file {tier_names_path} does not exist.")
                sys.exit(1)
            with open(tier_names_path, 'r', encoding='utf-8') as f:
                tier_names = [line.strip() for line in f.readlines() if line.strip()]
        else:
            tier_names = []
        for subfolder in data_dir.iterdir():
            if subfolder.is_dir():
                print(f"Processing subfolder: {subfolder}")
                sub_output_dir = output_dir / subfolder.name
                os.makedirs(sub_output_dir, exist_ok=True)
                os.makedirs(sub_output_dir / "segments", exist_ok=True)
                tier_names = TextGrid_to_Wav(
                    data_folder=str(subfolder),
                    output_audio_folder=str(sub_output_dir),
                    corr_map=corr_map,
                    tier_names=tier_names,
                ) 
        # save the tier names to a file in output_dir
        tier_names_path = output_dir / "tier_names.txt"
        with open(tier_names_path, 'w', encoding='utf-8') as f:
            for name in tier_names:
                f.write(name + '\n')
        print(f"Tier names saved to {tier_names_path}")
    
    # If tokenizer is provided, generate character mapping file and new vocab file
    if args.tokenizer:
        if not args.char_map_file:
            char_map_path = output_dir / "char_map.tsv"
        else:
            char_map_path = Path(args.char_map_file)
            
        CHARS = GenerateCharMap(
            tokenizer_name=args.tokenizer,
            output_path=str(char_map_path),
            output_audio_folder=str(output_dir),
        )
        
        ## save the vocab to a json file
        vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}

        init_no = max([v for k,v in vocab.items()]) + 1
        for i, c in enumerate(CHARS):
            vocab[c] = i + init_no
        
        vocab_path = output_dir / "vocab.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)
        print(f"Vocab file saved to {vocab_path}")


if __name__ == "__main__":
    main()
    