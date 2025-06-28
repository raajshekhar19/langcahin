from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)
message = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content = "tell me about langchain")
]

model = ChatHuggingFace(llm=llm)
res = model.invoke(message)

message.append(AIMessage(content=res.content))
print(message)