import xml.etree.ElementTree as ET

import pandas as pd
import torch
from tqdm import tqdm
import datasets


def get_embeddings(dataset, columns, model, tokenizer, n=float("inf")):
    input_ids = list()
    attention_masks = list()

    # Limit the number of samples
    n = len(dataset) if n > len(dataset) else n

    for index, line in tqdm(dataset.iterrows(), total=n, desc="Creating embeddings..."):
        input_id = list()
        attention_mask = list()

        # Concatenate the columns sentences input_ids and attention_masks
        for column in columns:
            sentence = line[column]

            dictionary = tokenizer.encode_plus(
                            sentence,
                            add_special_tokens = True,
                            max_length = 128,
                            truncation=True,
                            padding='max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                        )

            # Method encode_plus returns a dictionary
            input_id.append(dictionary['input_ids'])
            attention_mask.append(dictionary['attention_mask'])

        input_id = torch.cat(input_id, dim=-1)
        attention_mask = torch.cat(attention_mask, dim=-1)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)

    embeddings = outputs.last_hidden_state[:,0,:]

    return embeddings


def get_dataset(file):
    tree = ET.parse(file)
    root = tree.getroot()

    data = list()

    for pair in root.findall('pair'):
        data.append((
            pair.get('id'),
            pair.get('entailment'),
            pair.get('similarity'),
            pair.find('t').text,
            pair.find('h').text
        ))

    df = pd.DataFrame(
        data,
        columns=['id', 'entailment', 'similarity', 't', 'h']
    )

    df = df.replace({
        "Entailment": 1,
        "None": 0
    })

    df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce')

    return df


def get_rte_dataset(file):
    df = get_dataset(file)
    df_rte = df[["t", "h", "entailment"]]

    df_rte.columns = ['text_a','text_b','label']

    return df_rte


def get_sts_dataset(file):
    df = get_dataset(file)
    df_sts = df[["t", "h", "similarity"]]

    df_sts.columns = ['text_a','text_b','label']

    return df_sts


def prepare_datasets(train_df, dev_df, test_df, tokenize):
    assin_datasets = dict()

    assin_datasets['train'] = datasets.Dataset.from_pandas(train_df)
    assin_datasets['dev'] = datasets.Dataset.from_pandas(dev_df)
    assin_datasets['test'] = datasets.Dataset.from_pandas(test_df)

    for split in ['train', 'dev', 'test']:
        assin_datasets[split] = assin_datasets[split].map(
            tokenize, batched=True, batch_size=len(assin_datasets[split])
        )

    assin_datasets = datasets.DatasetDict(assin_datasets)

    return assin_datasets
