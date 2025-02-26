# 🚀 Efficient LoRA Fine-Tuning on KLUE-RoBERTa  

This repository contains an optimized **Low-Rank Adaptation (LoRA) fine-tuning** implementation for **KLUE-RoBERTa**, utilizing **8-bit quantization** for efficient training on limited computational resources such as Google Colab.  

---

## 📌 Features
✅ **LoRA-based Fine-Tuning**: Reduces trainable parameters for efficient optimization.  
✅ **8-bit Quantization**: Lowers memory usage while maintaining performance.  
✅ **Sentiment Analysis Task**: Trained on the NSMC (Korean Movie Reviews) dataset.  
✅ **Colab-Ready**: Designed to run on Google Colab with minimal modifications.  

---

## 📂 File Structure
lora-finetuning ├── lora_finetune.py # Main script for fine-tuning KLUE-RoBERTa ├── requirements.txt # List of required dependencies ├── README.md # Project documentation

📝 To-Do
- Evaluate on additional Korean NLP tasks
- Explore other lightweight tuning methods (e.g., QLoRA)
- Deploy as an API using FastAPI


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
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
