# north_caucasian_asr

python version `3.12`

## `data/` structure

Each folder is labeled by the name of the language. Eg. `data/Rutul/`

Each sub-folder is a possible split and contains `.TextGrid` and corresponding `.wav` files.


## `data/<LANG>/char_map.tsv` notation

This file should contain the correction map of characters in the transcription to characters recognizable by the tokenizer (say IPA).
If no such file is given to `src/data.py`, it is automatically generated based on a given tokenizer.

This file should have columns `src`, `dst` and `type`
`type` should be in `cyrillic`, `mixed`, `unclear`, `delimiter`, `glossing`, `punctuation`

> Sample command for data processing

 `python src/data.py --data-dir data/Rutul --char-map-file processed_data/Rutul/char_map_annotated.tsv --tokenizer ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns --tier-names-file processed_data/Rutul/tier_names.txt`

> Sample command for tokenized data creation

 `python src/pre_process.py --data-dir processed_data/Rutul --processor ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns --new-tokenizer custom`


## To do

* run training with accelerate
* Take note of the delimiter '#' of the default model (ctaguchi)
* Add feature to handle Archi data
* Fix identifier for tiers containing IPA transcripts
* Fix the notation of `data/<LANG>/char_map.tsv` f