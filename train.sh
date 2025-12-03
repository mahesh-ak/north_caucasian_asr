#python src/modelling_custom.py
#torchrun --nproc_per_node=2 src/train.py --model-dir models/Rutul/custom/split1 --data-dir tokenized_data/Rutul/custom/split3 --results-dir results/Rutul/custom/zero_shot --num-epochs 0 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir models/Archi/custom/split --data-dir tokenized_data/Archi/custom/split --results-dir results/Archi/custom/zero_shot --num-epochs 0 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir models/Rutul/custom/split1 --data-dir tokenized_data/Rutul/custom/split1 --num-epochs 20 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir models/Rutul/custom/split1_noinit --data-dir tokenized_data/Rutul/custom/split1 --results-dir results/Rutul/custom/split1_noinit --num-epochs 20 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir models/Rutul/custom/split2 --data-dir tokenized_data/Rutul/custom/split2 --num-epochs 20 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir models/Rutul/custom/split3 --data-dir tokenized_data/Rutul/custom/split3 --num-epochs 20 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir models/Archi/custom/split --data-dir tokenized_data/Archi/custom/split --num-epochs 20 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir models/Archi/custom/split_noinit --data-dir tokenized_data/Archi/custom/split --results-dir results/Archi/custom/split_noinit --num-epochs 20 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns --data-dir tokenized_data/Rutul/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns/split1 --results-dir results/Rutul/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns/zero_shot --num-epochs 0 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns --data-dir tokenized_data/Archi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns/split --results-dir results/Archi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns/zero_shot --num-epochs 0 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns --data-dir tokenized_data/Rutul/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns/split1 --num-epochs 20 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns --data-dir tokenized_data/Rutul/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns/split2 --num-epochs 20 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns --data-dir tokenized_data/Rutul/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns/split3 --num-epochs 20 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns --data-dir tokenized_data/Archi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns/split --num-epochs 20 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir openai/whisper-small --data-dir tokenized_data/Rutul/whisper-small/split1 --num-epochs 8 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir openai/whisper-small --data-dir tokenized_data/Archi/whisper-small/split --num-epochs 8 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir openai/whisper-large-v3 --data-dir tokenized_data/Rutul/whisper-large-v3/split1 --num-epochs 8 --batch-size 1
#torchrun --nproc_per_node=2 src/train.py --model-dir openai/whisper-large-v3 --data-dir tokenized_data/Archi/whisper-large-v3/split --num-epochs 8 --batch-size 1
torchrun --nproc_per_node=2 src/train.py --model-dir Qwen/Qwen2-Audio-7B-Instruct --data-dir tokenized_data/Archi/Qwen2-Audio-7B-Instruct/split --num-epochs 8 --batch-size 2 #--full-shard
torchrun --nproc_per_node=2 src/train.py --model-dir Qwen/Qwen2-Audio-7B-Instruct --data-dir tokenized_data/Rutul/Qwen2-Audio-7B-Instruct/split1 --num-epochs 8 --batch-size 2 #--full-shard
#torchrun --nproc_per_node=2 src/train.py --model-dir Qwen/Qwen2.5-Omni-7B --data-dir tokenized_data/Archi/Qwen2.5-Omni-7B/split --num-epochs 0 --batch-size 1 --full-shard

