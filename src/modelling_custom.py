from transformers import AutoConfig, Wav2Vec2ForCTC, AutoProcessor, WhisperForConditionalGeneration, WhisperConfig, WhisperProcessor
import torch

# modeling_whisper.py
# Exact mirror of modeling_custom.py but for Whisper encoder–decoder
# No CLI, only a single function: update_model_for_custom_tokenizer()

def update_model_for_custom_tokenizer_whisper(model_name, new_processor_path):
    """
    Mirror of modeling_custom.py but for Whisper encoder–decoder.

    Steps:
    1. Load new processor (already created in preprocess.py, with IPA vocab).
    2. Load base Whisper model.
    3. Replace decoder embeddings to match new vocab size.
    4. Heuristically initialize new embedding rows by mapping IPA tokens
       to substring-matching Whisper tokens.
    5. Save updated processor + model to new_processor_path.
    """

    # ------------------------------------------------------------
    # Load new processor (created externally in preprocess.py)
    # ------------------------------------------------------------
    new_processor = WhisperProcessor.from_pretrained(new_processor_path)
    new_tokenizer = new_processor.tokenizer

    # ------------------------------------------------------------
    # Load old processor & model
    # ------------------------------------------------------------
    old_processor = WhisperProcessor.from_pretrained(model_name)
    old_model = WhisperForConditionalGeneration.from_pretrained(model_name)
    old_tokenizer = old_processor.tokenizer

    # ------------------------------------------------------------
    # New config inherits from base but updates pad/bos/eos & vocab size
    # ------------------------------------------------------------
    new_config = WhisperConfig.from_pretrained(model_name)
    new_config.pad_token_id = new_tokenizer.pad_token_id
    new_config.bos_token_id = new_tokenizer.bos_token_id
    new_config.eos_token_id = new_tokenizer.eos_token_id
    new_config.vocab_size = new_tokenizer.vocab_size
    new_config.forced_decoder_ids = None
    new_config.suppress_tokens = []

    # Create new model with resized embeddings
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        config=new_config,
        ignore_mismatched_sizes=True
    )

    # ------------------------------------------------------------
    # Build heuristic mapping from new IPA tokens → old Whisper tokens
    # ------------------------------------------------------------
    new_vocab = new_tokenizer.get_vocab()          # token → id
    old_vocab = old_tokenizer.get_vocab()          # token → id

    mapping = {}                                   # new_id → [old_ids]

    for token, new_id in new_vocab.items():
        # special tokens
        if token in ["<s>", "</s>"]:
            continue

        # 1: direct match
        if token in old_vocab:
            mapping[new_id] = [old_vocab[token]]
            continue

        # 2: substring heuristics (mirror of modeling_custom.py)
        assigned = False
        for L in [3, 2, 1]:
            prefix = token[:L]
            suffix = token[L:]
            if prefix in old_vocab:
                mapping[new_id] = [old_vocab[prefix]]
                if suffix in old_vocab:
                    mapping[new_id].append(old_vocab[suffix])
                assigned = True
                break
        if assigned:
            continue

        # 3: map <pad>, <unk> etc.
        if token in ["<pad>", "<unk>"]:
            key = token.upper().replace("<", "[").replace(">", "]")
            if key in old_vocab:
                mapping[new_id] = [old_vocab[key]]
                continue

        # 4: fallback — map to Whisper's pad token
        mapping[new_id] = [old_tokenizer.pad_token_id]

    # ------------------------------------------------------------
    # Copy decoder embeddings using mapping (averaging if multiple)
    # ------------------------------------------------------------
    old_emb = old_model.model.decoder.embed_tokens.weight.data
    new_emb = model.model.decoder.embed_tokens.weight.data

    with torch.no_grad():
        for new_id, old_ids in mapping.items():
            w = old_emb[old_ids[0]].clone()
            if len(old_ids) == 2:
                w = 0.5 * (old_emb[old_ids[0]] + old_emb[old_ids[1]])
            new_emb[new_id] = w

    # ------------------------------------------------------------
    # Copy output projection (lm_head) similarly
    # ------------------------------------------------------------
    old_lm = old_model.lm_head.weight.data
    new_lm = model.lm_head.weight.data

    with torch.no_grad():
        for new_id, old_ids in mapping.items():
            w = old_lm[old_ids[0]].clone()
            if len(old_ids) == 2:
                w = 0.5 * (old_lm[old_ids[0]] + old_lm[old_ids[1]])
            new_lm[new_id] = w

    # ------------------------------------------------------------
    # Save model + processor
    # ------------------------------------------------------------
    model.save_pretrained(new_processor_path)
    new_processor.save_pretrained(new_processor_path)
    print(f"Saved updated Whisper model to {new_processor_path}")



def update_model_for_custom_tokenizer_wav2vec2(model_name, new_processor_path, avg=True):
    """
    Update the model's token embeddings to match the tokenizer's vocabulary size.
    This is necessary when a new tokenizer with a different vocabulary size is created.
    """
    init = True
    if 'noinit' in new_processor_path:
        init = False
    
    new_processor = AutoProcessor.from_pretrained(new_processor_path.replace('_noinit','').replace('_noavg',''))
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.pad_token_id = new_processor.tokenizer.pad_token_id
    model_config.bos_token_id = new_processor.tokenizer.bos_token_id
    model_config.eos_token_id = new_processor.tokenizer.eos_token_id
    model_config.vocab_size = new_processor.tokenizer.vocab_size

    model = Wav2Vec2ForCTC.from_pretrained(model_name,
                                        config= model_config,
                                        ignore_mismatched_sizes=True)
    
    model.freeze_feature_encoder=True
    
    old_processor = AutoProcessor.from_pretrained(model_name)
    
    
    # Create an approximate mapping from new tokenizer vocab ids to old tokenizer vocab ids
    vocab = new_processor.tokenizer.get_vocab()
    old_vocab = old_processor.tokenizer.get_vocab()

    ## new: old
    mapping = {}

    for k in vocab:
        if k == '<s>':
            continue
        if k == '</s>':
            mapping[vocab[k]] = [old_vocab['[PAD]']]
            continue
            
        if k not in old_vocab:
            if len(k)>3 and k[:3] in old_vocab:
                mapping[vocab[k]] = [old_vocab[k[:3]]]
                if k[3:] in old_vocab:
                    mapping[vocab[k]].append(old_vocab[k[3:]])
                continue

            if len(k)>2 and k[:2] in old_vocab:
                mapping[vocab[k]] = [old_vocab[k[:2]]]
                if k[2:] in old_vocab:
                    mapping[vocab[k]].append(old_vocab[k[2:]])
                continue

            if len(k)>1 and k[:1] in old_vocab:
                mapping[vocab[k]] = [old_vocab[k[:1]]]
                if k[1:] in old_vocab:
                    mapping[vocab[k]].append(old_vocab[k[1:]])
                continue

            if k in ['<pad>', '<unk>']:
                mapping[vocab[k]] = [old_vocab[k.upper().replace('<','[').replace('>',']')]]
                
        else:
            mapping[vocab[k]] = [old_vocab[k]]
    
    # Now copy the weights from old model to new model based on this mapping
    # strategy: if multiple old tokens map to a new token, average their weights
    if init:
        old_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        with torch.no_grad():
            for new_idx, old_idx in mapping.items():
                model.lm_head.weight[new_idx, :] = old_model.lm_head.weight[old_idx[0],:]
                model.lm_head.bias[new_idx]      = old_model.lm_head.bias[old_idx[0]]
                if len(old_idx) > 2 and avg:
                    for i in range(1, len(old_idx)):
                        model.lm_head.weight[new_idx, :] += old_model.lm_head.weight[old_idx[i],:]
                        model.lm_head.bias[new_idx]      += old_model.lm_head.bias[old_idx[i]]
                    model.lm_head.weight[new_idx, :] *= (1.0/len(old_idx))
                    model.lm_head.bias[new_idx]      *= (1.0/len(old_idx))
                        
    
    ## save to new_processor_path
    model.save_pretrained(new_processor_path)
    new_processor.save_pretrained(new_processor_path)
    print(f"Saved updated model to {new_processor_path}")
    


if __name__ == "__main__":
    #update_model_for_custom_tokenizer_wav2vec2("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", "models/Rutul/custom/split1") 
    #update_model_for_custom_tokenizer_wav2vec2("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", "models/Rutul/custom/split2")
    #update_model_for_custom_tokenizer_wav2vec2("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", "models/Rutul/custom/split3")
    update_model_for_custom_tokenizer_wav2vec2("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", "models/Archi/custom/split")
    #update_model_for_custom_tokenizer_wav2vec2("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", "models/Rutul/custom/split1_noinit") 
    #update_model_for_custom_tokenizer_wav2vec2("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", "models/Archi/custom/split_noinit")
    #update_model_for_custom_tokenizer_wav2vec2("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", "models/Rutul/custom/split1_noavg", avg=False) 
    #update_model_for_custom_tokenizer_wav2vec2("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", "models/Archi/custom/split_noavg", avg=False)





