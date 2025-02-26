import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Device 설정 (GPU 사용 여부 확인)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 및 토크나이저 로드 (KLUE-RoBERTa)
MODEL_NAME = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# LoRA 설정
lora_config = LoraConfig(
    r=8,                # LoRA 랭크
    lora_alpha=32,      # LoRA 스케일링 계수
    target_modules=["query", "value"],  # LoRA 적용 대상 모듈
    lora_dropout=0.1,
)

# 8-bit 양자화된 모델 로드
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.float16,  # 16-bit 연산
    load_in_8bit=True  # 8-bit 양자화 적용
).to(device)

# LoRA 적용
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 데이터셋 로드 (감성 분류 예제)
dataset = load_dataset("nsmc")  # 네이버 영화 리뷰 감성 분석 데이터

# 데이터 전처리 함수
def preprocess_function(examples):
    return tokenizer(examples["document"], padding="max_length", truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# 훈련 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,  # 16-bit 연산
)

# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
)

# 모델 학습
trainer.train()

# 모델 저장
trainer.save_model("./lora_finetuned_model")
print("Fine-tuning 완료! 모델이 저장되었습니다.")

