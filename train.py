import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from evaluation import compute_metrics
import json
import matplotlib.pyplot as plt
import argparse


def save_metrics(trainer, output_dir="./results"):
    metrics = trainer.evaluate()
    with open(f"{output_dir}/metrics.txt", "w") as f:
        f.write(str(metrics))
    plt.plot(metrics['eval_loss'])
    plt.savefig(f"{output_dir}/plots.png")


def plot_metrics(trainer, output_dir="./results"):
    metrics_history = trainer.state.log_history
    
    eval_losses = [entry['eval_loss'] for entry in metrics_history if 'eval_loss' in entry]
    
    if eval_losses:
        epochs = range(1, len(eval_losses) + 1)
        plt.plot(epochs, eval_losses, 'bo-', label='Evaluation loss')
        plt.title('Evaluation loss across epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{output_dir}/plots.png")
        plt.close()
    else:
        print("No evaluation losses found in the training logs.")


def train_model(data_dir, model_name, output_dir, best_params_file):
    # Load best hyperparameters from file
    with open(best_params_file, "r") as f:
        best_params = json.load(f)

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. Please make sure you have a GPU enabled environment.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DatasetDict({
        'train': load_dataset('csv', data_files=f'{data_dir}/train.csv')['train'],
        'validation': load_dataset('csv', data_files=f'{data_dir}/val.csv')['train'],
        'test': load_dataset('csv', data_files=f'{data_dir}/test.csv')['train']
    })

    num_labels = len(set(dataset['train']['label']))
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    model.to(device)

    dataset = dataset.map(lambda x: tokenizer(x['tweet'], padding="max_length", max_length=512, truncation=True), batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=best_params['num_train_epochs'],
        learning_rate=best_params['learning_rate'],
        per_device_train_batch_size=16,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    save_metrics(trainer, output_dir)
    plot_metrics(trainer, output_dir)

    model.save_pretrained(f"{output_dir}/fine_tuned_model")
    tokenizer.save_pretrained(f"{output_dir}/fine_tuned_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Roberta model for sentiment analysis")

    parser.add_argument('--data_dir', type=str, required=True, help="Directory where the dataset is located")
    parser.add_argument('--model_name', type=str, required=True, help="Pre-trained model name or path")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory where model output and results will be saved")
    parser.add_argument('--best_params_file', type=str, default="best_hyperparameters.json", help="File with best hyperparameters")

    args = parser.parse_args()
    
    train_model(args.data_dir, args.model_name, args.output_dir, args.best_params_file)