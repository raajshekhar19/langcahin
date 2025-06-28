from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline  # ✅ new import

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Setup pipeline with GPU (device=0)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Use new .invoke() method
response = llm.invoke("### Instruction:\nExplain what a transformer is in simple terms.\n\n### Response:")
print(response)
