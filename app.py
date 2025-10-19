from flask import Flask, render_template, jsonify, request
from src.helpers import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompts import *
import os

#Flask initialization
app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Set env variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

#Embed each chunk and upsert the embedding into your pinecone index
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# Docsearch as a retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4}
)

# Load LLM model
chatModel = ChatOpenAI(model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])

qa_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    qa_chain
)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input" : msg})
    print("Response: ", response['answer'])
    return str(response["answer"])

if __name__ == "__main__":
    # Allow overriding host/port via environment for flexibility
    host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_RUN_PORT", 8080))
    try:
        app.run(host=host, port=port, debug=True)
    except Exception as e:
        # Print full traceback for diagnostics and try a safer fallback
        import traceback
        print(f"Failed to start Flask on {host}:{port} — error: {e}")
        traceback.print_exc()
        if host != "127.0.0.1":
            print("Retrying on 127.0.0.1:8080")
            app.run(host="127.0.0.1", port=port, debug=True)

