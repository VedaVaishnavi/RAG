import os
import boto3
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_aws import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

S3_BUCKET = os.getenv("S3_BUCKET")
S3_KEY = os.getenv("S3_KEY")

EMBED_MODEL = os.getenv("EMBEDDING_MODEL_ID")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

print("Connected to S3")

response = s3_client.get_object(
    Bucket=S3_BUCKET,
    Key=S3_KEY
)

csv_content = response["Body"].read().decode("utf-8")

df = pd.read_csv(StringIO(csv_content))

print("CSV Loaded Successfully")
print(f"Shape: {df.shape}")

documents = []

for idx, row in df.iterrows():

    row_text = f"""
Plant ID: {row['Plant_ID']}
Production Line: {row['Production_Line']}
Vehicle Model: {row['Vehicle_Model']}
VIN Number: {row['VIN_Number']}
Batch Number: {row['Batch_Number']}
Production Date: {row['Production_Date']}
Shift: {row['Shift']}

Operator Name: {row['Operator_Name']}
Supervisor Name: {row['Supervisor_Name']}

Machine ID: {row['Machine_ID']}
Machine Identifier: {row['Machine_ID']}
Manufacturing Machine: {row['Machine_ID']}
Machine Reference Number: {row['Machine_ID']}

Machine Status: {row['Machine_Status']}
Maintenance Cost: {row['Maintenance_Cost_USD']}

Units Produced: {row['Units_Produced']}
Defect Count: {row['Defect_Count']}
Quality Score: {row['Quality_Score']}
Downtime Minutes: {row['Downtime_Minutes']}

Temperature: {row['Temperature_C']}
Pressure: {row['Pressure_PSI']}
Humidity: {row['Humidity_Percent']}

Energy Consumption: {row['Energy_Consumption_kWh']}

Raw Material Type: {row['Raw_Material_Type']}
Raw Material Weight: {row['Raw_Material_Weight_kg']}

Supplier Name: {row['Supplier_Name']}
Inventory Level: {row['Inventory_Level']}

Safety Incident: {row['Safety_Incident']}
Maintenance Cost: {row['Maintenance_Cost_USD']}
"""

    doc = Document(
        page_content=row_text,
        metadata={
    "plant_id": str(row["Plant_ID"]).strip(),
    "production_line": str(row["Production_Line"]).strip(),
    "vehicle_model": str(row["Vehicle_Model"]).strip(),
    "batch_number": str(row["Batch_Number"]).strip(),
    "machine_id": str(row["Machine_ID"]).strip(),
    "machine_status": str(row["Machine_Status"]).strip(),
    "shift": str(row["Shift"]).strip(),
    "supplier_name": str(row["Supplier_Name"]).strip(),
    "production_date": str(row["Production_Date"]).strip(),
    "safety_incident": str(row["Safety_Incident"]).strip()
}
    )

    documents.append(doc)

print(f"Created {len(documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

split_docs = text_splitter.split_documents(documents)

print(f"Created {len(split_docs)} chunks")

embeddings = BedrockEmbeddings(
    model_id=EMBED_MODEL,
    region_name="us-east-1"
)

pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [
    index["name"]
    for index in pc.list_indexes()
]

if PINECONE_INDEX_NAME not in existing_indexes:

    print("Creating Pinecone Index...")

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print("Pinecone Index Created")

else:

    print("Using Existing Pinecone Index")

vectorstore = PineconeVectorStore.from_documents(
    documents=split_docs,
    embedding=embeddings,
    index_name=PINECONE_INDEX_NAME
)

print("Data Indexed Successfully!")