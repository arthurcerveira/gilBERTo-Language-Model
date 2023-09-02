"""
Fine-tune a pretrained BERT model on the RTE task using the huggingface transformers library.
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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

from utils import get_rte_dataset, get_embeddings, prepare_datasets


id2label = {0: "none", 1: "entailment"}
label2id = {"none": 0, "entailment": 1}

f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "f1": f1.compute(predictions=predictions, references=labels),
        "accuracy": accuracy.compute(predictions=predictions, references=labels),
    }


def compute_rte(model_path, train, dev, test, output_dir):
    set_seed(42)

    train_df = get_rte_dataset(train)
    dev_df = get_rte_dataset(dev)
    test_df = get_rte_dataset(test)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )

    def tokenize(batch):
        return tokenizer(batch['text_a'], batch['text_b'], 
                         truncation=True, max_length=256, padding='max_length')

    assin_datasets = prepare_datasets(train_df, dev_df, test_df, tokenize)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,           # output directory
        learning_rate=2e-5,              # learning rate
        num_train_epochs=5,              # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        weight_decay=0.01,               # strength of weight decay
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,                                # the instantiated 🤗 Transformers model to be trained
        args=training_args,                         # training arguments, defined above
        train_dataset=assin_datasets['train'],      # training dataset
        eval_dataset=assin_datasets['dev'],         # evaluation dataset
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate(assin_datasets['test'])
    print(metrics)

    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write(f"{metrics['eval_f1']['f1']:.4f},{metrics['eval_accuracy']['accuracy']:.4f}")

    return metrics


def compute_rte_feature_based(model_path, train, dev, test, output_dir):
    train_df = get_rte_dataset(train)
    dev_df = get_rte_dataset(dev)

    # Validation set is used as training set
    train_df = pd.concat([train_df, dev_df])
    test_df = get_rte_dataset(test)

    config = AutoConfig.from_pretrained(model_path, num_labels=1)
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

    model = MLPClassifier(hidden_layer_sizes=(100, 100), 
                          max_iter=1000, 
                          random_state=42,
                          validation_fraction=0.2,
                          verbose=True)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    f1_score = f1.compute(predictions=y_pred, references=y_test)
    accuracy_score = accuracy.compute(predictions=y_pred, references=y_test)

    print(f1_score, accuracy_score)
    print(classification_report(y_test, y_pred, digits=4))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write(f"{f1_score['f1']:.4f},{accuracy_score['accuracy']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--train', type=str, default='data/assin2-train-only.xml')
    parser.add_argument('--dev', type=str, default='data/assin2-dev.xml')
    parser.add_argument('--test', type=str, default='data/assin2-test.xml')
    args = parser.parse_args()

    # compute_rte(args.model, args.train, args.dev, args.test, './results')
    compute_rte_feature_based(args.model, args.train, args.dev, args.test, './results')
