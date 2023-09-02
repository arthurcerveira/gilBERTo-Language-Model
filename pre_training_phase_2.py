import torch
from torch.utils.data import IterableDataset

import transformers
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import BertForMaskedLM, BertTokenizer, BertConfig


transformers.logging.set_verbosity_info()

# CHECKPOINT = "./ckpt-384000-phase-2"
CHECKPOINT = "./gilBERTo-phase-3-model"

print(f"Starting from checkpoint {CHECKPOINT}")

config = BertConfig().from_json_file(f"{CHECKPOINT}/config.json")
model = BertForMaskedLM(config=config)
tokenizer = BertTokenizer.from_pretrained(
    CHECKPOINT,
    max_len=512,
    do_lower_case=False,
)

class IterableLineByLineTextDataset(IterableDataset):
    def __init__(self, tokenizer, file_path: str, block_size: int):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size
        self.file = open(self.file_path, 'r', encoding='utf-8')

    def __iter__(self):
        for line in self.file:
            if len(line) == 0:
                continue
            # lines = [line for line in self.file.read().splitlines() if (len(line) > 0 and not line.isspace())]
            batch_encoding = self.tokenizer(line, add_special_tokens=True, truncation=True, max_length=self.block_size, 
                                            truncation_strategy='only_first_token', padding=True)
            yield {"input_ids": torch.tensor(batch_encoding["input_ids"], dtype=torch.long)}

    def __len__(self):
        if self.file_path == "./Validation.txt":
            return 7_851_663
        elif self.file_path=="./brWaC.txt":
            return 59_615_919  # 64_640_252 oscar

        return None

train_data = IterableLineByLineTextDataset(
    file_path="./brWaC.txt", tokenizer=tokenizer, block_size=384  # 256
)

validation = IterableLineByLineTextDataset(
    tokenizer=tokenizer, file_path="./Validation.txt", block_size=384  # 256
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# num_gpu = 1

training_args = TrainingArguments(
    output_dir="./gilBERTo-phase-4",
    overwrite_output_dir=True,
    num_train_epochs=1,
    # max_steps=total_steps,
    per_device_train_batch_size=32, # 8,
    per_device_eval_batch_size=32, # 8,
    save_steps=8_000,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=1e-4,
    do_eval=True,
    lr_scheduler_type="cosine",
    # ignore_data_skip=True
)

# optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, eps=1e-8)
# total_steps = training_args.num_train_epochs * len(train_data)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=validation,
    # optimizers=(optimizer, scheduler)
    # optimizers=optimizer
)

trainer.train(resume_from_checkpoint=CHECKPOINT)
# trainer.train()

trainer.save_model("./gilBERT-phase-4-model")

# print(training_args)
# print('\n\n\n', trainer)
