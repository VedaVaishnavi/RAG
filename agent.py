import os
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

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
)

embeddings = BedrockEmbeddings(
    model_id=EMBEDDING_MODEL_ID,
    region_name=AWS_REGION
)

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(PINECONE_INDEX_NAME)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace="schema"
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10
    }
)

PROMPT_TEMPLATE = """
You are a Nissan Manufacturing AI Assistant.

Use ONLY the provided enterprise manufacturing context
to answer the user's question.

Guidelines:
- Answer only from retrieved context
- Be concise and accurate
- If answer is unavailable say:
"I could not find relevant information in the enterprise database."

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
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
        "prompt": PROMPT
    }
)

def get_agent_response(message):

    response = qa_chain.invoke({
        "question": message
    })

    return response["answer"]

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