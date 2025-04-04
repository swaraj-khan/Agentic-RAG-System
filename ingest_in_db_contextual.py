# import basics
import os
from dotenv import load_dotenv
from tqdm import tqdm
import glob
import torch
import time
from typing import List

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

# import anthropic for contextual chunking
import anthropic

# load environment variables
load_dotenv()  

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initialize Anthropic client
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
claude_client = anthropic.Anthropic(api_key=anthropic_api_key)

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

def contextualize_chunk(chunk: Document, doc_source: str) -> Document:
    """
    Use Claude to contextualize a text chunk within its larger document
    """
    # Get original chunk text and metadata
    chunk_text = chunk.page_content
    metadata = chunk.metadata.copy()
    
    # Create prompt for Claude
    system_prompt = """Your job is to provide concise context for a text chunk to improve retrieval and relevance.
    Examine the chunk and generate a brief summary that places it in context, including:
    - What document or section it comes from
    - Key entities, topics, or concepts mentioned
    - Any relevant timeframes, numbers, or specific details
    - How it relates to the overall document theme
    
    Your response should be 2-3 sentences only, focusing on factual context rather than interpretation."""
    
    user_prompt = f"""I need context for this text chunk from a document titled '{metadata.get('title', 'Unknown')}':
    
    <chunk>
    {chunk_text}
    </chunk>
    
    Generate a brief contextual description (2-3 sentences) that would help place this chunk in context within the overall document. Focus on factual details."""
    
    try:
        # Call Claude to contextualize the chunk
        response = claude_client.messages.create(
            model="claude-3-haiku-20240307",  # Using Haiku for speed and cost efficiency
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        
        # Extract Claude's response
        context = response.content[0].text.strip()
        
        # Combine the context with the original chunk
        contextualized_text = f"{context}\n\n{chunk_text}"
        
        # Add metadata about contextualization
        metadata["contextualized"] = True
        metadata["original_length"] = len(chunk_text)
        metadata["contextualized_length"] = len(contextualized_text)
        
        # Create a new document with the contextualized text
        contextualized_doc = Document(
            page_content=contextualized_text,
            metadata=metadata
        )
        
        return contextualized_doc
    
    except Exception as e:
        print(f"Error contextualizing chunk: {e}")
        # Return original chunk if contextualization fails
        metadata["contextualized"] = False
        return chunk

def contextualize_chunks_batch(chunks: List[Document], batch_size: int = 10) -> List[Document]:
    """
    Contextualize a batch of chunks with rate limiting to avoid API throttling
    """
    contextualized_chunks = []
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Contextualizing chunks"):
        batch = chunks[i:i+batch_size]
        
        # Process each chunk in the batch
        for chunk in batch:
            contextualized_chunk = contextualize_chunk(chunk, chunk.metadata.get("source", ""))
            contextualized_chunks.append(contextualized_chunk)
        
        # Rate limiting - pause between batches to avoid API throttling
        if i + batch_size < len(chunks):
            time.sleep(1)  # 1 second pause between batches
    
    return contextualized_chunks

# load epub docs from folder 'documents'
print("Loading documents...")
documents = load_epub_directory("documents")
print(f"Loaded {len(documents)} documents")

# split the documents in multiple chunks
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks")

# Contextualize the chunks
print("Contextualizing chunks with Claude...")
contextualized_docs = contextualize_chunks_batch(docs, batch_size=5)
print(f"Contextualized {len(contextualized_docs)} chunks")

# store contextualized chunks in vector store with progress bar
print("Embedding contextualized documents and storing in Supabase...")
# Process documents in batches with a progress bar
batch_size = 32  # Increased batch size for GPU processing
total_batches = (len(contextualized_docs) + batch_size - 1) // batch_size  # Calculate total batches

for i in tqdm(range(0, len(contextualized_docs), batch_size), total=total_batches, desc="Processing document batches"):
    # Get current batch
    batch_docs = contextualized_docs[i:i+batch_size]
    
    # Process this batch
    SupabaseVectorStore.from_documents(
        batch_docs,
        embeddings,
        client=supabase,
        table_name="documents_contextual",
        query_name="match_documents",
        chunk_size=1000,
    )

print("Document processing complete!")