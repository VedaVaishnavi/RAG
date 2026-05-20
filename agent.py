import os
import streamlit as st
from typing import List
from pydantic import Field
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

load_dotenv()

AWS_REGION = st.secrets["AWS_REGION"]
BEDROCK_MODEL_ID = st.secrets["BEDROCK_LLM_MODEL_ID"]
EMBEDDING_MODEL_ID = st.secrets["BEDROCK_EMBEDDING_MODEL_ID"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)


embeddings = BedrockEmbeddings(
    model_id=EMBEDDING_MODEL_ID,
    region_name=AWS_REGION
)

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(PINECONE_INDEX_NAME)

# ==========================================
# 2. INITIALIZE NAMESPACE VECTOR STORES
# ==========================================

# Vector store pointing to your Column Definitions
schema_vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace="schema"
)

# Vector store pointing to your Raw Row Data
data_vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace="data"
)

# ==========================================
# 3. BUILD VALIDATED DUAL-NAMESPACE RETRIEVER
# ==========================================

class DualNamespaceRetriever(BaseRetriever):
    """Custom Retriever that targets schema and data namespaces sequentially."""
    schema_vectorstore: PineconeVectorStore = Field(exclude=True)
    data_vectorstore: PineconeVectorStore = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        
        # Build underlying standard retrievers 
        schema_retriever = self.schema_vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 3}
        )
        data_retriever = self.data_vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # Pull text contexts matching the query from both spaces
        schema_docs = schema_retriever.invoke(query)
        data_docs = data_retriever.invoke(query)
        
        # Wrap findings with visual boundaries so the LLM parses context correctly
        merged_docs = []
        for doc in schema_docs:
            doc.page_content = f"[SCHEMA] {doc.page_content}"
            merged_docs.append(doc)
            
        for doc in data_docs:
            doc.page_content = f"[DATA RECORD] {doc.page_content}"
            merged_docs.append(doc)
            
        return merged_docs

# Instantiate the compliant retriever object
combined_retriever = DualNamespaceRetriever(
    schema_vectorstore=schema_vectorstore, 
    data_vectorstore=data_vectorstore
)

# ==========================================
# 4. PROMPT DESIGN & CHAIN CONSTRUCTION
# ==========================================

PROMPT_TEMPLATE = """
You are an enterprise data assistant that answers questions using business schema definitions and retrieved database records.
 
Your job is to interpret the user's question using the [SCHEMA] context definitions, then answer the question accurately using the [DATA RECORD] context.
 
-------------------------
RETRIEVED CONTEXT (SCHEMA & DATA RECORDS)
{context}
-------------------------
Chat History:
{chat_history} 
-------------------------
USER QUESTION
{question}
-------------------------
INSTRUCTIONS
- Use [SCHEMA] lines to correctly interpret business terms or column keys.
- Use ONLY the [DATA RECORD] lines to pull statistics, counts, or values to answer the question.
- When the Information is asked about a particular VIN ID, give everydetail in a new line
- If the answer cannot be determined from the provided data records, clearly say so.
- Do NOT invent values, KPIs, or business rules.
- Do not make any assumptions.
- If the question is ambiguous, ask for clarification.

Output format:
Final Answer: 
Explanation:
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

# Initialize the chain with the validated custom retriever
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=combined_retriever,
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

# ==========================================
# 5. EXECUTION ENTRY POINT
# ==========================================
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
