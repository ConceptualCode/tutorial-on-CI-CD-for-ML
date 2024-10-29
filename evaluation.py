import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from evaluate import load
import json
import argparse
import os

def compute_metrics(eval_pred):
    accuracy_metric = load("accuracy")
    f1_metric = load("f1")
    precision_metric = load("precision")
    recall_metric = load("recall")

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"]
    }

def evaluate_model(data_dir, model_checkpoint, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RobertaForSequenceClassification.from_pretrained(model_checkpoint)
    tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
    
    model.to(device)
    
    test_dataset = load_dataset('csv', data_files=f'{data_dir}/test.csv')
    
    test_dataset = test_dataset.map(lambda x: tokenizer(x['tweet'], padding="max_length", truncation=True), batched=True)
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    results = trainer.evaluate(eval_dataset=test_dataset['train']) 
    
    print(f"Test set evaluation results: {results}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(f"{output_dir}/evaluation_results.json", "w") as f:
        json.dump(results, f)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Roberta model for sentiment analysis")
    
    parser.add_argument('--data_dir', type=str, required=True, help="Directory where the test dataset is located")
    parser.add_argument('--model_checkpoint', type=str, required=True, help="Path to the model checkpoint for evaluation")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory where evaluation results will be saved")

    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.model_checkpoint, args.output_dir)
