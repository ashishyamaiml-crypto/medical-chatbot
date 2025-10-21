from flask import Flask, render_template, request
from src.helpers import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompts import system_prompt
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

# -------------------- Flask Initialization --------------------
app = Flask(__name__)

# -------------------- Load Environment Variables --------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# -------------------- Initialize Pinecone (v5) --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# Create index if it does not exist
existing_indexes = [idx["name"] for idx in pc.list_indexes().get("indexes", [])]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # match your embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Created Pinecone index: {index_name}")
else:
    print(f"✅ Using existing Pinecone index: {index_name}")

# Connect to index
pinecone_index = pc.Index(index_name)

# -------------------- Load Embeddings --------------------
embeddings = download_hugging_face_embeddings()

# -------------------- Create LangChain Vector Store --------------------
docsearch = PineconeVectorStore(pinecone_index, embeddings)

# -------------------- Create Retriever --------------------
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# -------------------- Load LLM --------------------
chat_model = ChatOpenAI(model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])

qa_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# -------------------- Flask Routes --------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response['answer'])
    return str(response["answer"])

# -------------------- Run Flask --------------------
if __name__ == "__main__":
    host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_RUN_PORT", 8080))
    app.run(host=host, port=port, debug=True)
