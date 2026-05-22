import os
import re
import streamlit as st
from typing import List
from pydantic import Field
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

# New modern LCEL imports added here to prevent crashing
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# AWS and Pinecone Configurations
AWS_REGION = st.secrets["AWS_REGION"]
BEDROCK_MODEL_ID = st.secrets["BEDROCK_LLM_MODEL_ID"]
EMBEDDING_MODEL_ID = st.secrets["BEDROCK_EMBEDDING_MODEL_ID"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize LLM & Embeddings
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    streaming=True
)

embeddings = BedrockEmbeddings(
    model_id=EMBEDDING_MODEL_ID,
    region_name=AWS_REGION
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Vector Store Setup
schema_vectorstore = PineconeVectorStore(
    index=index, embedding=embeddings, text_key="text", namespace="schema"
)
data_vectorstore = PineconeVectorStore(
    index=index, embedding=embeddings, text_key="text", namespace="data"
)

# Custom Retriever Class
class DualNamespaceRetriever(BaseRetriever):
    """Custom Retriever that targets schema and data namespaces sequentially."""
    schema_vectorstore: PineconeVectorStore = Field(exclude=True)
    data_vectorstore: PineconeVectorStore = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        
        # Extract VIN using a regex match
        vin_match = re.search(r'\b[A-HJ-NPR-Z0-9]{17}\b', query, re.IGNORECASE) or re.search(r'vin[:\s]+(\w+)', query, re.IGNORECASE)
        data_kwargs = {"k": 5}
        
        if vin_match:
            extracted_vin = vin_match.group(1) if 'vin' in vin_match.group(0).lower() else vin_match.group(0)
            data_kwargs["filter"] = {"VIN_ID": {"$eq": extracted_vin.upper()}}
        
        schema_retriever = self.schema_vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 3}
        )
        data_retriever = self.data_vectorstore.as_retriever(
            search_kwargs=data_kwargs
        )
        
        schema_docs = schema_retriever.invoke(query)
        data_docs = data_retriever.invoke(query)
        
        merged_docs = []
        for doc in schema_docs:
            doc.page_content = f"[SCHEMA] {doc.page_content}"
            merged_docs.append(doc)
            
        for doc in data_docs:
            doc.page_content = f"[DATA RECORD] {doc.page_content}"
            merged_docs.append(doc)
            
        return merged_docs

combined_retriever = DualNamespaceRetriever(
    schema_vectorstore=schema_vectorstore, 
    data_vectorstore=data_vectorstore
)

# Prompt Configuration
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
- The VIN_ID is a unique identifier for each vehicle. If the question references a VIN_ID, only use the data record that matches that VIN_ID to answer.
- If the answer cannot be determined from the provided data records, clearly say so.
- Do NOT invent values, KPIs, or business rules.
- Do not make any assumptions.
- If the question is ambiguous, ask for clarification.

Output format:
-When the Information is asked about a particular VIN ID, give everydetail in a new line
Final Answer: 
Explanation:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question", "chat_history"]
)


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


# Modern LCEL Pipeline Configuration
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": combined_retriever | format_docs,
        "question": RunnablePassthrough(),
        "chat_history": lambda x: memory.load_memory_variables({}).get("chat_history", "")
    }

    | PROMPT
    | llm
    | StrOutputParser()
)

# Dynamic Streaming Token Generator Function
def get_agent_response_stream(message):
    full_answer = ""
    for token in rag_chain.stream(message):
        full_answer += token
        yield token
    memory.save_context({"input": message}, {"output": full_answer})
