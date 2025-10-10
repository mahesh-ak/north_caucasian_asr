from transformers import AutoConfig, Wav2Vec2ForCTC, AutoProcessor
import torch


def update_model_for_custom_tokenizer(model_name, new_processor_path):
    """
    Update the model's token embeddings to match the tokenizer's vocabulary size.
    This is necessary when a new tokenizer with a different vocabulary size is created.
    """
    
    new_processor = AutoProcessor.from_pretrained(new_processor_path)
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
    old_model = Wav2Vec2ForCTC.from_pretrained(model_name)
    with torch.no_grad():
        for new_idx, old_idx in mapping.items():
            model.lm_head.weight[new_idx, :] = old_model.lm_head.weight[old_idx[0],:]
            model.lm_head.bias[new_idx]      = old_model.lm_head.bias[old_idx[0]]
            if len(old_idx) > 2:
                model.lm_head.weight[new_idx, :] += old_model.lm_head.weight[old_idx[1],:]
                model.lm_head.bias[new_idx]      += old_model.lm_head.bias[old_idx[1]]
                model.lm_head.weight[new_idx, :] *= 0.5
                model.lm_head.bias[new_idx]      *= 0.5
    
    ## save to new_processor_path
    model.save_pretrained(new_processor_path)
    print(f"Saved updated model to {new_processor_path}")
    


if __name__ == "__main__":
    update_model_for_custom_tokenizer("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", "models/Rutul/custom/split1") 
    update_model_for_custom_tokenizer("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", "models/Rutul/custom/split2")
    update_model_for_custom_tokenizer("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns", "models/Archi/custom/split")


