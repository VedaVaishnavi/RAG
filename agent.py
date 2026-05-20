<<<<<<< HEAD
import re
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrockConverse
=======
import os
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock
>>>>>>> 8ca219e (Schema Defination)
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

<<<<<<< HEAD
AWS_REGION = st.secrets["AWS_REGION"]
BEDROCK_MODEL_ID = st.secrets["BEDROCK_MODEL_ID"]
EMBEDDING_MODEL_ID = st.secrets["EMBEDDING_MODEL_ID"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

llm = ChatBedrockConverse(
    model=BEDROCK_MODEL_ID,
    region_name="us-east-1",
    temperature=0
=======
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

llm = ChatBedrock(
    model_id=BEDROCK_MODEL_ID,
    region_name=AWS_REGION,
    model_kwargs={
        "temperature": 0
    }
>>>>>>> 8ca219e (Schema Defination)
)

embeddings = BedrockEmbeddings(
    model_id=EMBEDDING_MODEL_ID,
<<<<<<< HEAD
    region_name="us-east-1"
)

pc = Pinecone(api_key=PINECONE_API_KEY)
=======
    region_name=AWS_REGION
)

pc = Pinecone(
    api_key=PINECONE_API_KEY
)
>>>>>>> 8ca219e (Schema Defination)

index = pc.Index(PINECONE_INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
<<<<<<< HEAD
    embedding=embeddings
=======
    embedding=embeddings,
    text_key="text",
    namespace="schema"
>>>>>>> 8ca219e (Schema Defination)
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
<<<<<<< HEAD
        "k": 20
=======
        "k": 10
>>>>>>> 8ca219e (Schema Defination)
    }
)

PROMPT_TEMPLATE = """
You are a Nissan Manufacturing AI Assistant.

Use ONLY the provided enterprise manufacturing context
to answer the user's question.

<<<<<<< HEAD
If the answer is not present in the context, say:
=======
Guidelines:
- Answer only from retrieved context
- Be concise and accurate
- If answer is unavailable say:
>>>>>>> 8ca219e (Schema Defination)
"I could not find relevant information in the enterprise database."

Context:
{context}

<<<<<<< HEAD
=======
Chat History:
{chat_history}

>>>>>>> 8ca219e (Schema Defination)
Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
<<<<<<< HEAD
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
=======
    input_variables=[
        "context",
        "question",
        "chat_history"
    ]
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={
>>>>>>> 8ca219e (Schema Defination)
        "prompt": PROMPT
    }
)

def get_agent_response(message):

    response = qa_chain.invoke({
<<<<<<< HEAD
        "query": f"""
        Retrieve manufacturing information related to:
        {message}
        """
    })

    return response["result"]
=======
        "question": message
    })

    return response["answer"]
>>>>>>> 8ca219e (Schema Defination)

if __name__ == "__main__":

    print("\nNISSAN AI AGENT READY\n")

    while True:

        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        try:

            answer = get_agent_response(user_input)

            print("\nAssistant:")
            print(answer)
            print()

        except Exception as e:

            print("\nERROR:")
            print(str(e))