import argparse
from pathlib import Path
from utils import *
from datasets import load_dataset_from_disk
from transformers import AutoModelForCTC, TrainingArguments, Trainer, AutoProcessor
import torch


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
        help="Directory to save the training results. If not provided, results will be saved in results/<lang>/<model_name>, <lang> is inferred from data_dir name, model_name is inferred from model argument.",
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
        default=20,
        help="Number of training epochs. (default: 20)",
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = args.model_dir
    data_dir = Path(args.data_dir)
    
    ## data_dir format: tokenized_data/<lang>/<partial_model_name>/ and contains subdirs train, dev, test
    ## partial_model_name is the model name without the path
    results_dir = Path(args.results_dir) if args.results_dir else Path("results") / Path(*data_dir.parts[1:3])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    
    print(f"Loading dataset from {data_dir}")
    tokenized_dataset = load_dataset_from_disk(data_dir)
    
    print(f"Loading model and processor from {model_name}")
    model = AutoModelForCTC.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # define training args
    training_args = TrainingArguments(
        output_dir=results_dir,
        group_by_length=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=16//batch_size, # to simulate batch size of 16
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_epochs,
        fp16=torch.cuda.is_available(),
        learning_rate=3e-4,
        save_total_limit=2,
        push_to_hub=False,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    # define data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # define trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics= lambda pred: compute_metrics(pred, processor, tokenized_dataset['dev']),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        tokenizer=processor.feature_extractor,
    )

    print("Starting training")
    trainer.train()
    print("Training completed")
    
    print("Evaluating on test set")
    trainer.compute_metrics = lambda pred: compute_metrics(pred, processor, tokenized_dataset['test'], save_results=True, results_folder=results_dir)
    trainer.eval_dataset = tokenized_dataset["test"]
    test_metrics = trainer.evaluate()
    print(f"Test set metrics: {test_metrics}")
    print(f"Training and evaluation results saved in {results_dir}")
    
    
if __name__ == "__main__":
    main()
     
