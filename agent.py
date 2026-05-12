import re
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrockConverse
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

AWS_REGION = st.secrets["AWS_REGION"]
BEDROCK_MODEL_ID = st.secrets["BEDROCK_MODEL_ID"]
EMBEDDING_MODEL_ID = st.secrets["EMBEDDING_MODEL_ID"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

llm = ChatBedrockConverse(
    model=BEDROCK_MODEL_ID,
    region_name="us-east-1",
    temperature=0
)

embeddings = BedrockEmbeddings(
    model_id=EMBEDDING_MODEL_ID,
    region_name="us-east-1"
)

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 20
    }
)

PROMPT_TEMPLATE = """
You are a Nissan Manufacturing AI Assistant.

Use ONLY the provided enterprise manufacturing context
to answer the user's question.

If the answer is not present in the context, say:
"I could not find relevant information in the enterprise database."

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": PROMPT
    }
)

def get_agent_response(message):

    response = qa_chain.invoke({
        "query": f"""
        Retrieve manufacturing information related to:
        {message}
        """
    })

    return response["result"]

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