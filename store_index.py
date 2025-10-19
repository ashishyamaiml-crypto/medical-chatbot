from dotenv import load_dotenv
import os
from src.helpers import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Only set environment variables if they were provided
if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Resolve data path relative to project root (one level up from this script)
here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(here, "."))
data_path = os.path.join(project_root, "data")
print(f"Using data path: {data_path}")

extracted_data = load_pdf_file(data=data_path)
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_split(filter_data)

embeddings = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)