import os
from datetime import datetime

from rte_hf import compute_rte, compute_rte_feature_based
from sts_hf import compute_sts, compute_sts_feature_based


MODELS = ["M-BERT", "BERTimbau", "BertPT", "gilBERTo-phase-1", 
          "gilBERTo-phase-2", "gilBERTo-phase-3", "gilBERTo-phase-4"]

TRAIN = "data/assin2-train-only.xml"
DEV = "data/assin2-dev.xml"
TEST = "data/assin2-test.xml"

TASKS = {
    "rte-fine-tuning": {
        "function": compute_rte,
        "output": "ft_rte.csv",
        "metrics": ["f1", "accuracy"]
    },
    "sts-fine-tuning": {
        "function": compute_sts,
        "output": "ft_sts.csv",
        "metrics": ["pearson", "mse"]
    },
    "sts-feature-based": {
        "function": compute_sts_feature_based,
        "output": "fb_sts.csv",
        "metrics": ["pearson", "mse"]
    },
    "rte-feature-based": {
        "function": compute_rte_feature_based,
        "output": "fb_rte.csv",
        "metrics": ["f1", "accuracy"]
    }
}

def compare_task(model, task, results):
    print(f"[{datetime.now():%H:%M}] Running {model} on {task} task with ASSIN dataset")

    model_path = f"../Models/{model}"
    output = f"output/{task}_{model.lower().replace('-', '_')}"
    compute_task = TASKS[task]["function"]

    if os.path.exists(output):
        print(f"Output directory already exists: {output}")
    else:
        compute_task(model_path, TRAIN, DEV, TEST, output)

    with open(f"{output}/metrics.txt") as f:
        metrics = f.read()

    with open(results, "a") as f:
        f.write(f"{model},{metrics}\n")


if __name__ == "__main__":
    for task in TASKS:
        output = f"results/{TASKS[task]['output']}"

        if os.path.exists(output):
            print(f"Output file already exists: {output}")
            continue

        metrics = ",".join(TASKS[task]["metrics"])

        with open(output, "w") as f:
            f.write(f"model,{metrics}\n")

        for model in MODELS:
            compare_task(model, task, output)

