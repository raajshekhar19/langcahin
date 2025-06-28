from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation'

)

model = ChatHuggingFace(llm = llm)
message = [
    SystemMessage(content='you are a helpful ai assitant')
    
]
while True:
    user_input = input('You:')
    message.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    res = model.invoke(message)
    message.append(AIMessage(content = res.content))
    print("AI :",res)