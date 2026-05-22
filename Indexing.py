import io
import os
import json
import boto3
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
DATA_FILE_KEY = os.getenv("DATA_FILE_KEY")
COLUMN_DEF_KEY = os.getenv("COLUMN_DEF_KEY")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

print("Reading Data File from S3...")

csv_obj = s3.get_object(
    Bucket=BUCKET_NAME,
    Key=DATA_FILE_KEY
)

data_df = pd.read_csv(
    io.BytesIO(csv_obj["Body"].read())
)
print(f"Loaded Data Rows: {len(data_df)}")

print("Reading column_definitions.xlsx from S3...")
excel_obj = s3.get_object(
    Bucket=BUCKET_NAME,
    Key=COLUMN_DEF_KEY
)
column_df = pd.read_excel(
    io.BytesIO(excel_obj["Body"].read())
)
print(f"Loaded Column Definitions: {len(column_df)}")

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

existing_indexes = [
    index["name"]
    for index in pc.list_indexes()
]

EMBEDDING_DIM = 1024

if PINECONE_INDEX not in existing_indexes:

    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

print("Pinecone index ready.")

index = pc.Index(PINECONE_INDEX)
#Embedding Function
def get_embedding(text):

    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({
            "inputText": text
        })
    )

    response_body = json.loads(
        response["body"].read()
    )

    return response_body["embedding"]

print("Indexing column definitions...")

schema_vectors = []

for idx, row in column_df.iterrows():

    column_name = str(row["COLUMN_NAME"])
    description = str(row["DESCRIPTION"])

    text = f"""
    Column Name: {column_name}
    Description: {description}
    """

    embedding = get_embedding(text)

    metadata = {
        "type": "schema",
        "column_name": column_name,
        "description": description,
        "text": text
    }

    schema_vectors.append(
        (
            f"schema-{idx}",
            embedding,
            metadata
        )
    )

index.upsert(
    vectors=schema_vectors,
    namespace="schema"
)

print(f"Indexed {len(schema_vectors)} schema vectors.")

print("Indexing data rows...")

batch_vectors = []

BATCH_SIZE = 100

for idx, row in data_df.iterrows():

    row_parts = []

    metadata = {
        "type": "data"
    }

    for column in data_df.columns:

        value = str(row[column])

        row_parts.append(
            f"{column}: {value}"
        )

        metadata[column] = value

    row_text = " | ".join(row_parts)

    embedding = get_embedding(row_text)

    batch_vectors.append(
        (
            f"row-{idx}",
            embedding,
            {
                **metadata,
                "text": row_text
            }
        )
    )

    if len(batch_vectors) >= BATCH_SIZE:

        index.upsert(
            vectors=batch_vectors,
            namespace="data"
        )

        print(f"Uploaded batch till row {idx}")

        batch_vectors = []

if batch_vectors:

    index.upsert(
        vectors=batch_vectors,
        namespace="data"
    )

print("All data indexed successfully.")

# query = "vehicle identifier"

# query_embedding = get_embedding(query)

# results = index.query(
#     namespace="schema",
#     vector=query_embedding,
#     top_k=3,
#     include_metadata=True
    
# )

# print("\nTop Results:\n")

# for match in results["matches"]:

#     print("=" * 50)

#     print("Score:", match["score"])

#     print(match["metadata"]["text"])