from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load local model (replace with your local path or model ID)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # can be a path like "./my-local-model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Create HF pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
)

# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Prompt templates
template1 = PromptTemplate.from_template("### Human:\nwrite a detailed report on the {topic}\n\n### Assistant:")
template2 = PromptTemplate.from_template("### Human:\nwrite a 5 line summary on the following text:\n{text}\n ## Assistant:")

# Create prompts
prompt1 = template1.invoke({"topic": "black hole"})
output1 = llm.invoke(prompt1)

prompt2 = template2.invoke({"text": output1})
output2 = llm.invoke(prompt2)

# Print results
print("\n--- Summary ---\n", output2)
