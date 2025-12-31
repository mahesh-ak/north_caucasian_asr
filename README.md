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

 `python src/pre_process.py --data-dir processed_data/Rutul --processor ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns --new-tokenizer custom --split-file processed_data/Rutul/split1.json`

> Sample command for training and evaluating with multiple GPUs

 `torchrun --nproc_per_node=2 src/train.py --model-dir models/Rutul/custom/ --data-dir tokenized_data/Rutul/custom/ --num-epochs 5 --batch-size 1`

> Sample command to run gpt-4o-transcribe:
 `python src/prompt_llm.py --data-dir processed_data/Archi --split-json processed_data/Archi/split.json --vocab processed_data/Archi/vocab.json`

> Sample command to run wav2vec2-ipa with 3-gram lm decoding:
`python src/infer_with_lm.py --data-dir tokenized_data/Archi/custom/split/test --model models/Archi/custom/split --lm-path models/Archi/lm/split/lm.klm --beam-size 10 --alpha 0.3 --beta 0.3 --results-dir results/Archi/custom/split_lm`

> Command to train 3-gram lm:
` python src/train_lm.py --data-dir processed_data/Rutul --split-file processed_data/Rutul/split1.json --order 3`



base_vowels = {'a', 'e', 'i', 'o', 'u', 'ɨ', 'ə', 'y'}
long = 'ː'
pharyn = 'ˤ'

base_consonants = {'b', 'd', 'd͡ʒ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 's', 't', 't͡s', 't͡ʃ', 'w', 'x', 'ɢ', 'ʁ', 'ʃ', 'χ', 'ɬ', 'ʒ', 'ʟ', 'ɣ', 'ʔ', 'ʕ', 'z', 'r', 'j', 'ħ', 'ɮ'}
lab = 'ʷ'
eject = 'ʼ'
pal = 'ʲ'

## IPA-Cyrillic Map
Rutul:  a-а aː-аа  aˤ-аӀ b-б d-д dʲ-дʼ d͡ʒ-дж d͡ʒʷ-джв e-е eˤ-еӀ f-ф g-г gʲ-гʼ gʷ-гв h-гь hʷ-гьв i-и iː-ии iˤ-иӀ j-й k-к kʲ-кʼ kʷ-кв kʼ-кӀ kʼʲ-кӀʼ kʼʷ-кӀв l-л lʲ-лʼ m-м mʲ-мʼ n-н nʲ-нʼ o-о oˤ-оӀ p-п pʲ-пʼ pʼ-пӀ q-хъ qʷ-хъв qʼ-хъӀ qʼʷ-хъӀв r-р s-с sʲ-сʼ t-т tʲ-тʼ tʼ-тӀ t͡s-ц t͡sʼ-цӀ t͡ʃ-ч t͡ʃʷ-чв t͡ʃʼ-чӀ u-у uː-уу uˤ-уӀ v-в w-в wʲ-вʼ x-хь xʲ-хьʼ xʷ-хьв y-уь z-з ø-ё ɢ-къ ɢʷ-къв ɣ-гӀ ɨ-ы ɨː-ыы ɨˤ-ыӀ ʁ-гъ ʁʷ-гъв ʃ-ш ʃʲː-щʼ ʒ-ж ʔ-Ӏъ χ-х χʷ-хв


Archi: a-a aː-аа aːˤ-ааӏ aˤ-аӏ b-б d-д e-е eː-ее eːˤ-ееӏ eˤ-еӏ g-г gʷ-гв h-гь i-и iː-ии iˤ-иӏ j-й k-к kʷ-кв kʼ-кӏ kʼʷ-кӏв kː-кк kːʷ-ккв l-л m-м n-н o-о oː-оо oːˤ-ооӏ oˤ-оӏ p-п pʼ-пӏ pː-пп q-хъ qʷ-хъв qʼ-хъӏ qʼʷ-къв qʼː-ккъ qʼːˤ-ккъӏ qʼˤ-къӏ qʼˤʷ-къӏв qˤ-хъӏ qˤʷ-хъв r-р s-с sː-сс t-т tʼ-тӏ tː-тт t͡s-ц t͡sʼ-цӏ t͡sʼː-ццӏ t͡ʃ-ч t͡ʃʼ-чӏ u-у uː-уу uˤ-уӏ w-в z-з ħ-гӏ ə-a ɬ-лъ ɬː-ллъ ɬːʷ-ллъв ɮ-лъ ʁ-гъ ʁˤ-гъӏ ʃ-ш ʃʷ-шв ʃː-щ ʃːʷ-щв ʒ-ж ʒʷ-жв ʔ-ъ ʕ-ӏ ʟ-лӏ ʟʼ-кь ʟʼʷ-кьв χ-х χʷ-хв χː-хх χːʷ-ххв χːˤ-ххьӏ χˤ-хьӏ