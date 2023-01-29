from datasets import DatasetDict, load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, GPT2LMHeadModel, AutoConfig

train_data = load_dataset("text", data_files="./data/Urban/words_train.txt", split="train")
test_data = load_dataset("text", data_files="./data/Urban/words_validate.txt", split="validation")

datasets = DatasetDict({"train": train_data, "valid": test_data})

context_length = 512

tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

def tokenize(element):
    outputs = tokenizer(
        element,
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokenized_datasets = datasets.map(tokenize, batched=True, remove_columns=datasets["train"].column_names)

print(tokenized_datasets)

config = AutoConfig.from_pretrained_config(
    "gpt2", vocab_size = len(tokenizer),
    n_ctx = context_length, bos_token_id = tokenizer.bos_token_id,
    eos_token_id = tokenizer.eos_token_id
)

model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train()