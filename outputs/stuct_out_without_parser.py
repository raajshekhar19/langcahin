from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='write a detailed report on the {topic}',
    input_variables=['topic']
    )

template2 = PromptTemplate(
    template='write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})

resut = model.invoke(prompt1)

prompt2 = template2.invoke({'text':resut.content})

resut1 = model.invoke(prompt2)

print(resut1.content)