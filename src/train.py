import argparse
from pathlib import Path
from utils import *
from datasets import load_from_disk
from transformers import (
    AutoModelForCTC,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Trainer,
    Seq2SeqTrainer,
    AutoProcessor,
)
from transformers.models.whisper import WhisperProcessor
import torch
import inspect


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Wav2Vec2 model for ASR on a given dataset."
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Pretrained model name or path from Hugging Face or local directory. Processor should also be present in the same directory. Checkpoints will be saved in this directory.",
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing the subfolders train, dev and test with huggingface dataset files",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=False,
        help="Directory to save the training results. If not provided, results will be saved in results/<lang>/<model_name>/<split_name>, <lang> and <split_name> is inferred from data_dir name, model_name is inferred from model argument.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training. (default: 2)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of training epochs. (default: 30, 10 for whisper models)",
    )
    
    return parser.parse_args()

_original_prediction_step = Trainer.prediction_step
def patched_prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    # Save ids before passing to model
    ids = inputs.get("id")
    inputs = {k: v for k, v in inputs.items() if k != "id"}

    # Standard HF prediction step
    loss, logits, labels = _original_prediction_step(self, model, inputs, prediction_loss_only, ignore_keys)

    # Attach ids to logits (so they appear inside pred.predictions)
    if isinstance(logits, dict):
        logits["id"] = ids
    else:
        logits = {"logits": logits, "id": ids}

    return loss, logits, labels


_original_seq2seq_prediction_step = Seq2SeqTrainer.prediction_step

def patched_seq2seq_prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    # Save ids before passing to model
    ids = inputs.get("id")
    labels = inputs.get("labels")
    # Remove ids from inputs so model.generate doesn't complain
    inputs = {k: v for k, v in inputs.items() if k not in ("id","labels")}

    # Standard Seq2SeqTrainer prediction step
    loss, generated_tokens, _ = _original_seq2seq_prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    # Wrap predictions in dict to include IDs
    if generated_tokens is not None:
        if isinstance(generated_tokens, dict):
            generated_tokens["id"] = ids
            generated_tokens["labels"] = labels
        else:
            generated_tokens = {"generated_tokens": generated_tokens, "id": ids, "labels": labels}

    return loss, generated_tokens, labels

def main():
    args = parse_args()
    model_name = args.model_dir
    data_dir = Path(args.data_dir)

    model_type = "ctc"
    lr = 3e-4
    num_epochs = args.num_epochs

    if 'whisper' in model_name.lower():
        model_type = "whisper"
        lr = 1e-5
        num_epochs = min(num_epochs, 10)  # limit epochs for whisper fine-tuning
        if 'large' in model_name.lower():
            lr = 5e-6
    ## data_dir format: tokenized_data/<lang>/<partial_model_name>/<split_name> and contains subdirs train, dev, test
    ## partial_model_name is the model name without the path
    results_dir = Path(args.results_dir) if args.results_dir else Path("results") / Path(*data_dir.parts[1:])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = args.batch_size
        
    print(f"Loading dataset from {data_dir}")
    tokenized_dataset = { split: load_from_disk(data_dir / split) for split in ['train','validation','test'] }

    # create ids for each split for traceability
    for split in ['train','validation','test']:
        tokenized_dataset[split] = tokenized_dataset[split].add_column("id", list(range(len(tokenized_dataset[split]))))

    ## created input_dataset with only features "input_values", "labels" and "id" for memory efficiency
    input_dataset = {}
    for split in ['train','validation','test']:
        if model_type == "ctc":
            keep_cols = ['input_values', 'labels', 'id']
        else:  # whisper
            keep_cols = ['input_features', 'labels', 'id']

        cols_to_remove = [col for col in tokenized_dataset[split].column_names if col not in keep_cols]
        input_dataset[split] = tokenized_dataset[split].remove_columns(cols_to_remove)
    
    print(f"Loading processor from {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)

    print(f"Loading model ({model_type}) from {model_name}")
    if model_type == "ctc":
        model = AutoModelForCTC.from_pretrained(model_name)
    else:
        # Whisper encoder-decoder
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        # ensure special tokens are set
        if hasattr(processor, "tokenizer"):
            model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
            model.config.eos_token_id = processor.tokenizer.eos_token_id
            model.config.pad_token_id = processor.tokenizer.pad_token_id

    # include id in model forward pass to be able to access it in compute_metrics
    orig_forward = model.forward
    model.forward = lambda *a, **kw: (out := orig_forward(*a, **{k:v for k,v in kw.items() if k in inspect.signature(orig_forward).parameters})) or {**out, "id": kw.get("id")}

    #model.freeze_feature_encoder = True
    
    ## output_dir is model_name if model_name is a path, else create a directory in results_dir
    if Path(model_name).exists():
        output_dir = model_name
    else:
        output_dir = Path('models') / Path(*data_dir.parts[1:])
        output_dir.mkdir(parents=True, exist_ok=True)
        model.config.bos_token_id = processor.tokenizer.bos_token_id
        model.config.eos_token_id = processor.tokenizer.eos_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        
    # define data collator
    if model_type == "ctc":
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    else:
        data_collator = DataCollatorWhisperWithPadding(processor=processor, padding=True)


    # define training args
    if model_type == "ctc":
        training_args = TrainingArguments(
            output_dir=output_dir,
            group_by_length=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=16//batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=num_epochs,
            fp16=torch.cuda.is_available(),
            learning_rate=lr,
            save_total_limit=1,
            push_to_hub=False,
            remove_unused_columns=False,
        )
    else:
        # Whisper uses seq2seq generation
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=16//batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=num_epochs,
            fp16=torch.cuda.is_available(),
            learning_rate=lr,
            warmup_steps=100,
            weight_decay=0.01,
            predict_with_generate=True,
            generation_max_length=128,
            generation_num_beams=1,  # beam search is expensive; keep 1 for training
            save_total_limit=1,
            remove_unused_columns=False,
        )
    Trainer.prediction_step = patched_prediction_step
    Seq2SeqTrainer.prediction_step = patched_seq2seq_prediction_step
    # define trainer
    if model_type == "ctc":
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=lambda pred: compute_metrics(pred, processor, tokenized_dataset['validation'], model_type=model_type),
            train_dataset=input_dataset["train"],
            eval_dataset=input_dataset["validation"],
            processing_class=processor,
        )
    else:
        # Whisper: model returns seq2seq predictions
        trainer = Seq2SeqTrainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=lambda pred: compute_metrics(pred, processor, tokenized_dataset['validation'], model_type=model_type),
            train_dataset=input_dataset["train"],
            eval_dataset=input_dataset["validation"],
            tokenizer=processor.tokenizer,
        ) 

    trainer.train()
    ## save model and processor to model_dir
    if num_epochs > 0:
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
    
    print("Evaluating on test set")
    trainer.compute_metrics = lambda pred: compute_metrics(pred, processor, tokenized_dataset['test'], model_type=model_type, save_results=True, results_folder=str(results_dir))
    trainer.eval_dataset = input_dataset["test"]
    test_metrics = trainer.evaluate()
    print(f"Test set metrics: {test_metrics}")
    print(f"Evaluation results saved in {results_dir}")
    
    
if __name__ == "__main__":
    main()
     
