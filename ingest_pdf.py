# import basics
import os
from dotenv import load_dotenv
from tqdm import tqdm  # Import tqdm for progress bar

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# import supabase
from supabase.client import Client, create_client

# load environment variables
load_dotenv()  

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model using HuggingFace with 1536 dimensions
embeddings = HuggingFaceEmbeddings(model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")  # 1536 dimensions

# load pdf docs from folder 'documents'
print("Loading documents...")
loader = PyPDFDirectoryLoader("documents")
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# split the documents in multiple chunks
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks")

# store chunks in vector store with progress bar
print("Embedding documents and storing in Supabase...")
# Process documents in batches with a progress bar
batch_size = 10  # Adjust based on your needs
total_batches = (len(docs) + batch_size - 1) // batch_size  # Calculate total batches

for i in tqdm(range(0, len(docs), batch_size), total=total_batches, desc="Processing document batches"):
    # Get current batch
    batch_docs = docs[i:i+batch_size]
    
    # Process this batch
    SupabaseVectorStore.from_documents(
        batch_docs,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
        chunk_size=1000,
    )

print("Document processing complete!")