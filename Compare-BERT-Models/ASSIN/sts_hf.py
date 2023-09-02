"""
Fine-tune a pretrained BERT model on the STS task using the huggingface transformers library.
"""
import argparse
import os

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from utils import get_sts_dataset, get_embeddings, prepare_datasets


pearson = evaluate.load("pearsonr")
mse = evaluate.load("mse")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.squeeze(predictions)
    return {
        "pearsonr": pearson.compute(predictions=predictions, references=labels),
        "mse": mse.compute(predictions=predictions, references=labels),
    }


def compute_sts(model_path, train, dev, test, output_dir):
    set_seed(42)

    train_df = get_sts_dataset(train)
    dev_df = get_sts_dataset(dev)
    test_df = get_sts_dataset(test)

    config = AutoConfig.from_pretrained(
        model_path, 
        num_labels=1
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        config=config
    )

    def tokenize(batch):
        return tokenizer(batch['text_a'], batch['text_b'], truncation=True, padding=True)

    assin_datasets = prepare_datasets(train_df, dev_df, test_df, tokenize)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,       
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=assin_datasets['train'],
        eval_dataset=assin_datasets['dev'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate(assin_datasets['test'])
    print(metrics)

    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write(f"{metrics['eval_pearsonr']['pearsonr']:.4f},{metrics['eval_mse']['mse']:.4f}")


def compute_sts_feature_based(model_path, train, dev, test, output):
    train_df = get_sts_dataset(train)
    dev_df = get_sts_dataset(dev)

    # Validation set is used as training set
    train_df = pd.concat([train_df, dev_df])
    test_df = get_sts_dataset(test)

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModel.from_pretrained(model_path, config=config)

    embeddings = get_embeddings(train_df, 
                                ["text_a", "text_b"],
                                model, 
                                tokenizer)

    X_train = embeddings.numpy()
    y_train = train_df['label'].values

    embeddings = get_embeddings(test_df,
                                ["text_a", "text_b"],
                                model,
                                tokenizer)

    X_test = embeddings.numpy()
    y_test = test_df['label'].values

    model = MLPRegressor(hidden_layer_sizes=(100, 100), 
                         max_iter=1000, 
                         random_state=42,
                         validation_fraction=0.2)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    result_mse = mse.compute(predictions=y_pred, references=y_test)
    result_pearson = pearson.compute(predictions=y_pred, references=y_test)

    print(f"pearson: {result_pearson['pearsonr']}, mse: {result_mse['mse']}")

    if not os.path.exists(output):
        os.makedirs(output)

    with open(f'{output}/metrics.txt', 'w') as f:
        f.write(f"{result_pearson['pearsonr']:.4f},{result_mse['mse']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str)
    parser.add_argument('--train', type=str, default='data/assin2-train-only.xml')
    parser.add_argument('--dev', type=str, default='data/assin2-dev.xml')
    parser.add_argument('--test', type=str, default='data/assin2-test.xml')
    args = parser.parse_args()

    # compute_sts(args.model, args.train, args.dev, args.test, './results/')
    compute_sts_feature_based(args.model, args.train, args.dev, args.test)
