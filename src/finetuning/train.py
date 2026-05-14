"""LoRA fine-tuning script for Mistral-7B using QLoRA + SFTTrainer.

Run:
    python -m src.finetuning.train \
        --train_file data/train.jsonl \
        --val_file data/val.jsonl \
        --output_dir checkpoints/mistral-7b-lora-v1
"""

import argparse
import mlflow
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def train(train_file: str, val_file: str, output_dir: str) -> None:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = get_peft_model(prepare_model_for_kbit_training(model), LORA_CONFIG)
    model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files=train_file, split="train")
    val_ds = load_dataset("json", data_files=val_file, split="train")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        bf16=True,
        report_to="none",
    )

    mlflow.set_experiment("llm-platform-finetuning")
    with mlflow.start_run(run_name="mistral-7b-lora-v1"):
        mlflow.log_params({"r": 16, "alpha": 32, "lr": 2e-4, "epochs": 3, "base_model": BASE_MODEL})

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            dataset_text_field="text",
            max_seq_length=2048,
            args=args,
        )
        trainer.train()
        trainer.save_model(output_dir)
        mlflow.log_artifact(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", default="./checkpoints/mistral-7b-lora-v1")
    args = parser.parse_args()
    train(args.train_file, args.val_file, args.output_dir)
