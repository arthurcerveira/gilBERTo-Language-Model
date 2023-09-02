import subprocess
from datetime import datetime
from itertools import product


# MODELS = ["BERTimbau", "BertPT", "M-BERT", "ckpt-216000", "ckpt-584000"]
# MODELS = ["BERTimbau", "BertPT", "M-BERT", "gilBERTo-phase-1", "ckpt-384-bs512",
#           "ckpt-704-bs256", "ckpt-728-bs384", "gilBERTo-phase-2"]
MODELS = ["BERTimbau", "BertPT", "M-BERT", "gilBERTo-phase-1", "gilBERTo-phase-2",
          "gilBERTo-phase-3", "gilBERTo-phase-4"]
DATASETS = ["total", "selective"]
CRF = [False, True]
FEATURE_BASED = [False, True]


def compare_ner(models, dataset="total", use_crf=False, use_feature_based=False):
    # Create a DataFrame to store the results
    ner_command = """
        python run_bert_harem.py \
        --bert_model {model} \
        --labels_file data/classes-{dataset}.txt \
        --do_train \
        --train_file data/FirstHAREM-{dataset}-train.json \
        --valid_file data/FirstHAREM-{dataset}-dev.json \
        {crf} \
        {feature_based} \
        --num_train_epochs {epochs} \
        --per_gpu_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --do_eval \
        --eval_file data/MiniHAREM-{dataset}.json \
        --output_dir {output}
    """

    crf = "" if use_crf else "--no_crf"
    epochs = 15 if use_crf else 50
    output_file = f"results/ner_bert_{dataset}_{epochs}.csv"
    feature_based = ""

    # If feature-based, overwrite epochs and output file
    if use_feature_based is True:
        feature_based = "--freeze_bert --pooler sum"
        epochs = 50 if use_crf else 100
        output_file = f"results/lstm_ner_bert_{dataset}_{epochs}.csv"

    with open(output_file, "w") as f:
        f.write("Model,F1,Precision,Recall\n")

    # Iterate over the models
    for model in models:
        # Create the output directory
        model_key = model.lower().replace("-", "_")

        if use_feature_based is True:
            output = f"output/lstm_output_{dataset}_{model_key}_{epochs}"
        else:
            output = f"output/output_{dataset}_{model_key}_{epochs}"

        print(
            f"[{datetime.now():%H:%M}] Running {model} on {dataset} dataset " + 
            f"with CRF={use_crf} and feature-based={use_feature_based}"
        )

        model_path = f"../Models/{model}"

        command = ner_command.format(
            model=model_path,
            dataset=dataset,
            crf=crf,
            feature_based=feature_based,
            epochs=epochs,
            output=output
        )

        # Run the command
        subprocess.run(
            command.split(),
            stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL
        )

        # Read the results
        with open(f"{output}/metrics.txt", "r") as f:
            line = f.readline()
            f1, precision, recall = line.split(",")

        # Write the results to the file
        with open(output_file, "a") as f:
            f.write(f"{model},{float(f1):.4f},{float(precision):.4f},{float(recall):.4f}\n")


if __name__ == "__main__":
    for dataset, use_crf, use_feature_based in product(DATASETS, CRF, FEATURE_BASED):
        compare_ner(MODELS, dataset, use_crf, use_feature_based)
