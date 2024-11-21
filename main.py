import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

import wandb
wandb_api_key = "41ec56b69a856d0cf90136bb06297ff522b8dd3c"
wandb.login(key=wandb_api_key)

from datasets import Dataset, DatasetDict

# Tạo dataset từ file CSV
dataset = Dataset.from_csv("Palette_Data.csv")

# Chia dữ liệu
split_data = dataset.train_test_split(test_size=0.3, seed=42)
test_valid = split_data["test"].train_test_split(test_size=0.5, seed=42)

# Tạo DatasetDict
dataset_dict = DatasetDict({
    "train": split_data["train"],
    "validation": test_valid["test"],
    "test": test_valid["train"],
})

print(type(dataset_dict))  # Phải in ra <class 'datasets.dataset_dict.DatasetDict'>

def formatting_fc(example):
  if example.get("context", "") != "" : # check trong dist có feild context ko ?
    input_prompt = ("<|im_start|>system\n"
                    "Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.<|im_end|>\n\n"
                    "<|im_start|>user\n"
                    f"{example['Question']}<|im_end|>\n\n"
                    "<|im_start|>assistant"
                    f"{example['Answer']}<|im_end|>"
                    )
  else :
    input_prompt = (
                    "<|im_start|>system\n"
                    "Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.<|im_end|>\n\n"
                    "<|im_start|>user\n"
                    f"{example['Question']}<|im_end|>\n\n"
                    "<|im_start|>assistant\n"
                    f"{example['Answer']}<|im_end|>"
                    )

  return {
      'text' : input_prompt
  }
# Áp dụng `map` lên từng tập con
ColorGenQA = dataset_dict.map(formatting_fc, remove_columns=['ID'])

# In dữ liệu từ train
print(ColorGenQA['train'][0]['text'])

model_id_cp = "vilm/vinallama-2.7b-chat"

bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)



model = AutoModelForCausalLM.from_pretrained(
    model_id_cp,
    quantization_config=bnb_config,
    device_map={"": 0}
)

tokenizer = AutoTokenizer.from_pretrained(model_id_cp, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.config.use_cache = False
model.config.pretraining_tp = 1

print(ColorGenQA)

ColorGenQA['train'][0]['text']


# Set training parameters
training_arguments = TrainingArguments(
    output_dir='model_train_dir',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="all",
)
trainer = SFTTrainer(
    model=model,
    train_dataset=ColorGenQA['train'],
    eval_dataset=ColorGenQA['validation'],  
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

import time

# Số epoch bạn muốn train
num_epochs = 50

print("Starting training for multiple epochs...")
total_start_time = time.time()  # Ghi lại thời gian bắt đầu tổng quát

for epoch in range(1, num_epochs + 1):
    print(f"\nStarting training for epoch {epoch}...")
    start_time = time.time()  # Thời gian bắt đầu cho từng epoch
    
    trainer.train()  # Train 1 epoch
    
    end_time = time.time()  # Thời gian kết thúc từng epoch
    training_time = end_time - start_time
    
    # Chuyển đổi thời gian training thành giờ và phút
    training_time_hours = int(training_time // 3600)
    training_time_minutes = int((training_time % 3600) // 60)
    
    print(f"Epoch {epoch} completed!")
    print(f"Time taken for epoch {epoch}: {training_time_hours}h {training_time_minutes}m")

# Lưu mô hình sau khi train xong tất cả các epoch
trainer.model.save_pretrained('/kaggle/working/model_adapter')
total_end_time = time.time()  # Ghi lại thời gian kết thúc tổng quát

# Thời gian đào tạo tổng
total_training_time = total_end_time - total_start_time
total_training_time_hours = int(total_training_time // 3600)
total_training_time_minutes = int((total_training_time % 3600) // 60)

print(f"\nTraining success!")
print(f"Total training time for {num_epochs} epochs: {total_training_time_hours}h {total_training_time_minutes}m")

base_model = AutoModelForCausalLM.from_pretrained(
    model_id_cp,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, 'model_adapter')
model = model.merge_and_unload()

model.push_to_hub('vinallama-2b-custom-color-verfinal')
