import json
import optuna
import argparse
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import numpy as np
import torch
from evaluation import compute_metrics


def objective(trial, data_dir, model_name, output_dir, lr_range, epoch_range, batch_sizes):

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. Please make sure you have GPU for fast training.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    learning_rate = trial.suggest_loguniform('learning_rate', lr_range[0], lr_range[1])
    num_train_epochs = trial.suggest_int('num_train_epochs', epoch_range[0], epoch_range[1])
    per_device_batch_size = trial.suggest_categorical('per_device_batch_size', batch_sizes)

    dataset = DatasetDict({
        'train': load_dataset('csv', data_files=f'{data_dir}/train.csv')['train'],
        'validation': load_dataset('csv', data_files=f'{data_dir}/val.csv')['train']
    })

    num_labels = len(set(dataset['train']['label']))

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    model.to(device)

    # Tokenize datasets
    max_length = 128
    dataset = dataset.map(lambda x: tokenizer(x['tweet'], padding="max_length", truncation=True, max_length=max_length), batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_total_limit=1
    )

    # Initialize Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()

    if 'eval_f1' in eval_results:
        return eval_results['eval_f1']
    else:
        print("F1 score not found in evaluation results. Using accuracy instead.")
        return eval_results['eval_accuracy']


def main(args):
    lr_range = [float(x) for x in args.lr_range.split(',')]
    epoch_range = [int(x) for x in args.epoch_range.split(',')]
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    # Create Optuna study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    study.optimize(lambda trial: objective(trial, args.data_dir, args.model_name, args.output_dir, lr_range, epoch_range, batch_sizes), n_trials=args.n_trials)
    
    with open(args.best_params_file, "w") as f:
        json.dump(study.best_params, f)
    
    print(f"Best hyperparameters: {study.best_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna for Transformers model")
    
    parser.add_argument('--data_dir', type=str, required=True, help="Directory where the dataset is located")
    parser.add_argument('--model_name', type=str, required=True, help="Pre-trained model name or path")
    parser.add_argument('--output_dir', type=str, default="./pm_search", help="Directory where the model output will be saved")
    parser.add_argument('--best_params_file', type=str, default="best_hyperparameters.json", help="File where the best hyperparameters will be saved")
    parser.add_argument('--n_trials', type=int, default=5, help="Number of trials for Optuna optimization")
    parser.add_argument('--lr_range', type=str, default="1e-5,1e-3", help="Learning rate range for log uniform sampling (e.g., '1e-5,1e-3')")
    parser.add_argument('--epoch_range', type=str, default="1,5", help="Number of epochs range (e.g., '1,5')")
    parser.add_argument('--batch_sizes', type=str, default="8,16,32", help="Batch sizes to sample from (e.g., '8,16,32')")
    
    args = parser.parse_args()
    main(args)
