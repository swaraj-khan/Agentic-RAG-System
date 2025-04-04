# import basics
import os
from dotenv import load_dotenv
from tqdm import tqdm
import glob
import torch  # Add torch import to check GPU availability

# Import for custom epub loader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re

# import langchain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# import supabase
from supabase.client import Client, create_client

# load environment variables
load_dotenv()  

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model using HuggingFace with 1536 dimensions and GPU
embeddings = HuggingFaceEmbeddings(
    model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja",  # 1536 dimensions
    model_kwargs={"device": device},  # Use GPU if available
    encode_kwargs={"device": device, "batch_size": 32}  # Larger batch size for GPU
)

# Function to extract text from HTML content
def get_text_from_html(html_content):
    """Extract text from HTML content using BeautifulSoup"""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    # Clean the text (remove extra whitespace)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Custom ePub loader class
def load_epub_file(file_path):
    """Load an ePub file and return a list of Documents"""
    documents = []
    
    try:
        # Load the ePub file
        book = epub.read_epub(file_path)
        
        # Get the book title for metadata
        title = book.get_metadata('DC', 'title')
        book_title = title[0][0] if title else os.path.basename(file_path)
        
        # Process each item
        for item in book.get_items():
            # Only process HTML/XHTML documents
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Get content as HTML
                html_content = item.get_content().decode('utf-8', errors='replace')
                
                # Extract text from HTML
                text = get_text_from_html(html_content)
                
                # Skip if text is too short or empty
                if len(text) < 20:
                    continue
                
                # Create a Document with metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "title": book_title,
                        "item_id": item.id,
                        "file_type": "epub"
                    }
                )
                documents.append(doc)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return documents

def load_epub_directory(directory_path):
    """Load all ePub files in a directory"""
    # Get all epub files in the directory
    epub_files = glob.glob(os.path.join(directory_path, "*.epub"))
    documents = []
    
    # Load each epub file with a progress bar
    for file_path in tqdm(epub_files, desc="Loading ePub files"):
        docs = load_epub_file(file_path)
        documents.extend(docs)
    
    return documents

# load epub docs from folder 'documents'
print("Loading documents...")
documents = load_epub_directory("documents")
print(f"Loaded {len(documents)} documents")

# split the documents in multiple chunks
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks")

# store chunks in vector store with progress bar
print("Embedding documents and storing in Supabase...")
# Process documents in batches with a progress bar
batch_size = 32  # Increased batch size for GPU processing
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