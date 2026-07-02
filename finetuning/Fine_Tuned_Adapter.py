from transformers import AutoModelForCausalLM,AutoTokenizer,TrainingArguments,Trainer
from peft import LoraConfig,get_peft_model
from datasets import Dataset

model_name="mistralai/Mistral-7B-Instruct-v0.2"

tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")

lora_config=LoraConfig(r=8,lora_alpha=16,
target_modules=["q_proj","v_proj"],
lora_dropout=0.05,
task_type="CAUSAL_LM")


model=get_peft_model(model,lora_config)

data=[{"text":"Question: What is RAG?\nAnswer: Retrieval Augmented Generation"}]
dataset=Dataset.from_list(data)

dataset=dataset.map(lambda x:tokenizer(x["text"],truncation=True,padding="max_length",max_length=512))

training_args=TrainingArguments(output_dir="./fine_tuned_model",learning_rate=2e-4,num_train_epochs=3,per_device_train_batch_size=2,logging_steps=10)

trainer=Trainer(model=model,args=training_args,train_dataset=dataset)

trainer.train()

model.save_pretrained("./rag_finetuned")
tokenizer.save_pretrained("./rag_finetuned")