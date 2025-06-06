# Engineering Document Summarizer with Mistral LLM

# ==========================================================
# 📌 Step 1: Install dependencies (uncomment in new environments)
# ==========================================================
# !pip install transformers datasets accelerate peft bitsandbytes rouge-score

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import torch
import pandas as pd
from rouge_score import rouge_scorer

# ==========================================================
# 📌 Step 2: Load and Prepare Data (Simulated Engineering Summaries)
# ==========================================================
data = {
    'text': [
        "This document discusses the design considerations of a turbofan engine including fan blade optimization, thermal efficiency, and fuel burn reduction.",
        "The report explains the failure mechanisms in composite materials used in aircraft fuselage under extreme load conditions.",
        "A study on the aerodynamic performance improvements using winglets in commercial aircraft and their fuel efficiency impact.",
    ],
    'summary': [
        "Turbofan engine design focuses on fan blade optimization and fuel efficiency.",
        "Composite fuselage materials fail under extreme loads due to structural weaknesses.",
        "Winglets improve aerodynamics and fuel efficiency in commercial aircraft.",
    ]
}

# Convert to HuggingFace dataset
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Train-test split
dataset = dataset.train_test_split(test_size=0.2)

# ==========================================================
# 📌 Step 3: Load Pre-trained Mistral Model and Tokenizer
# ==========================================================
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# ==========================================================
# 📌 Step 4: Preprocess Data into Prompt + Completion Format
# ==========================================================
def preprocess(example):
    """
    Prepares each example from the dataset by formatting it into a prompt-completion pair
    suitable for causal language model training. The prompt includes the document text,
    and the target is the summary. Tokenization pads/truncates to a fixed length.

    Args:
        example (dict): A dictionary with 'text' and 'summary' fields.

    Returns:
        dict: Tokenized input with input_ids and labels for training.
    """
    prompt = f"### Document:\n{example['text']}\n\n### Summary:\n"
    full_text = prompt + example['summary']
    tokenized = tokenizer(full_text, max_length=512, truncation=True, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

encoded_dataset = dataset.map(preprocess, batched=False)

# ==========================================================
# 📌 Step 5: Define Training Arguments
# ==========================================================
# Define the training configuration for the Trainer API
# Includes model save location, evaluation strategy, batch sizes, epochs, etc.
training_args = TrainingArguments(
    output_dir="./mistral-summarizer",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    learning_rate=2e-5,  # Set learning rate
    lr_scheduler_type="cosine",  # Learning rate decay strategy
    warmup_steps=10,  # Gradual warm-up for stability
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    save_strategy="epoch",
    push_to_hub=False
)

# Data collator dynamically pads inputs for batching during training
# Setting mlm=False because we're training a causal language model (not masked LM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==========================================================
# 📌 Step 6: Fine-Tune the Mistral Model
# ==========================================================
# Trainer manages the training and evaluation loop
# It uses the model, datasets, tokenizer, collator, and training args defined above
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# ==========================================================
# 📌 Step 7: Save Fine-Tuned Model
# ==========================================================
trainer.save_model("mistral-summarizer")

# ==========================================================
# 📌 Step 8: Test Inference on a New Engineering Document
# ==========================================================
sample_text = "This paper presents a new method to reduce aircraft noise using adaptive engine nacelle designs and sound absorption materials."
prompt = f"### Document:\n{sample_text}\n\n### Summary:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
summary_ids = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True)
output_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).split("### Summary:")[-1].strip()

print("\nOriginal:\n", sample_text)
print("\nGenerated Summary:\n", output_summary)

# ==========================================================
# 📌 Step 9: Evaluate with ROUGE Score
# ==========================================================
# Evaluated model performance using ROUGE-1, ROUGE-2, and ROUGE-L metrics;
# Optimized training with half-precision (fp16) and evaluation at each epoch.
reference_summary = "This paper proposes reducing aircraft noise using nacelle design and sound-absorbing materials."
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_summary, output_summary)
print("\nROUGE Evaluation:")
for metric, score in scores.items():
    print(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1={score.fmeasure:.4f}")
