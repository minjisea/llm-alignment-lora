# 🚀 Optimized LoRA Fine-Tuning on KLUE-RoBERTa
This repository provides an efficient LoRA fine-tuning implementation for KLUE-RoBERTa, optimized with 8-bit quantization for low-resource environments like Google Colab.


# 📌 Key Features
✅ LoRA-based Fine-Tuning: Efficient parameter tuning with reduced memory footprint.

✅ 8-bit Quantization: Optimized for low-resource environments using bitsandbytes.

✅ Sentiment Analysis Task: Pretrained on NSMC (Korean Movie Reviews).

✅ Colab-Ready: Designed for free-tier Google Colab execution.


# 📂 llm-alignment-lora
 ├── lora_finetune.py      # LoRA fine-tuning script for KLUE-RoBERTa
 ├── requirements.txt      # Required dependencies
 ├── README.md             # Documentation
 ├── dataset_loader.py     # NSMC dataset loader
 ├── inference.py          # Inference script for evaluation
 ├── utils.py              # Helper functions (quantization, logging)

 
# 🔗 References
LoRA: Low-Rank Adaptation of Large Language Models (https://arxiv.org/abs/2106.09685)
KLUE: Korean Language Understanding Evaluation (https://arxiv.org/abs/2105.09680)
Hugging Face Transformers (https://huggingface.co/docs/transformers/index)


# 📝 To-Do
- Evaluate on additional Korean NLP tasks
- Explore other lightweight tuning methods (e.g., QLoRA)
- Deploy as an API using FastAPI


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  

# 🚀llm-alignment-lora
KLUE-RoBERTa 기반 LoRA + 8-bit 양자화 적용 LLM


# KLUE-RoBERTa + LoRA Fine-Tuning (8-bit 양자화)
Colab 환경에서 실행 가능한 경량화된 KLUE-RoBERTa LoRA Fine-Tuning 코드입니다.


## 📌 주요 내용
✅ **LoRA (Low-Rank Adaptation)** 적용

✅ **8-bit 양자화** (`bitsandbytes` 활용)로 경량화

✅ **KLUE-RoBERTa** 모델 기반 Fine-Tuning

✅ **Colab 실행 가능 (무료 환경 지원)**


## 실행 방법
1. Colab에서 notebook 실행
2. 'requirements.txt' 패키지 설치 후 'lora_finetune.py' 실행


## 🔗 참고 자료
- [LoRA 논문](https://arxiv.org/abs/2106.09685)
- [KLUE 논문](https://arxiv.org/abs/2105.09680)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
