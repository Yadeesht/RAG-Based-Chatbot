from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import StorageContext, load_index_from_storage
import google.generativeai as genai
import os

# Load environment variables
if os.environ.get("RAILWAY") != "true":
    from dotenv import load_dotenv
    load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

Settings.llm = None
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

try:
    storage_context = StorageContext.from_defaults(persist_dir="index_store")
    index = load_index_from_storage(storage_context)
except Exception as e:
    print(f"Error loading index: {str(e)}")
    print("Please run initialize_rag.py first to create the index")
    raise

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)]
)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(model_name='gemini-1.5-flash')

def query_rag_engine(comment: str) -> str:
    try:
        response = query_engine.query(comment)

        context = "Context:\n"
        for node in response.source_nodes:
            context += node.text + "\n\n"
        
        if not context.strip():
            context = "No relevant data was found for this query from Yadeesh's portfolio.\n\n"

        prompt = f"""You are J.A.R.V.I.S, an intelligent assistant for exploring Yadeesh's portfolio.
        Your behavior:
        - Provide friendly, human-like, and concise answers.
        - Only extract and include information related to the user's query.
        - If the query is about projects, include ONLY project-related information.
        - Use Yadeesh's name explicitly when referring to him.
        - Do not include general objectives, skills, or achievements unless they are directly related to a project.
        - Format your response with bullet points using "-" for each point.
        - Add a line break between different sections.
        - Keep each point concise and clear.
        - Dont reply anything other than the question asked related to the profile and resume,website.
        - End every response with a line break followed by "--J.A.R.V.I.S" on a new line.

        {context}
        Please respond to the following comment. Use the context above if it is helpful.

        {comment}
        [/INST]
        """
        gemini_response = model.generate_content(prompt)
        return gemini_response.text
    except Exception as e:
        return f"Error processing query: {str(e)}"





