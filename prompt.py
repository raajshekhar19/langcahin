from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)
chat_history = []

model = ChatHuggingFace(llm=llm)

while True:
    user_input = input("you:")
    chat_history.append(user_input)
    if user_input=='exit':
        break
    res = model.invoke(chat_history)
    chat_history.append(res)
    print(f"Ai:{res.content}")
