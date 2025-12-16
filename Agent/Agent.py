import os
from Config import Config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

config = Config()
os.environ["OPENAI_API_KEY"] =  config.OPENAI_API_KEY
llm = ChatOpenAI(model_name=config.gen_model, temperature=0)

def ask_agent(prompt, data):
    prompt = ChatPromptTemplate.from_messages([("system", prompt), ("human", "{input}")])
    formatted = prompt.format_messages(input=data)
    response = llm.invoke(formatted)
    return response.content