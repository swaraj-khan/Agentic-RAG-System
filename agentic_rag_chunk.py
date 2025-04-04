import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from supabase.client import Client, create_client
import json

# Load environment variables
load_dotenv()

# Initialize Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")

# Initialize vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="document_contextual",
    query_name="match_documents_contextual",
)

# Initialize LLM
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama3-70b-8192",
    temperature=0
)

# Create system prompt
system_template = """You are a sophisticated trading and investment assistant. Answer questions based on the provided trading book information.

INSTRUCTIONS:
1. ALWAYS cite the specific book(s) using the format: "According to [Book Title]..."
2. When multiple books provide relevant information, cite each of them clearly.
3. Base your answer primarily on the trading book information provided.
4. Use professional trading terminology.

The following chunks from trading books are relevant to the query:
{chunks}
"""

def retrieve_chunks(query, k=3):
    """Retrieve chunks related to a query"""
    docs = vector_store.similarity_search(query, k=k)
    return docs

def generate_response(query, docs):
    """Generate response based on query and chunks"""
    # Format chunks for the prompt
    chunks_text = "\n\n".join(
        f"CHUNK {i+1}:\nSource: {doc.metadata}\nContent: {doc.page_content}"
        for i, doc in enumerate(docs)
    )
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{query}")
    ])
    
    # Format the prompt with the chunks and query
    formatted_prompt = prompt.format(chunks=chunks_text, query=query)
    
    # Get response from LLM
    response = llm.invoke(formatted_prompt)
    return response.content

# Streamlit UI
st.title("Trading Book Contextual Retrieval System")
st.write("Ask a question about trading and investment to see relevant chunks from books")

# Query input
query = st.text_input("Enter your question:")

# Number of chunks to retrieve
k = st.slider("Number of chunks to retrieve", min_value=1, max_value=5, value=3)

if st.button("Search"):
    if query:
        with st.spinner("Retrieving information..."):
            # Retrieve chunks
            docs = retrieve_chunks(query, k)
            
            # Display chunks
            st.subheader("Retrieved Chunks")
            for i, doc in enumerate(docs):
                title = doc.metadata.get("title", "Unknown Book")
                with st.expander(f"Chunk {i+1} - {title}", expanded=True):
                    # Display content first for better readability
                    st.write("**Content:**")
                    st.write(doc.page_content)
                    
                    # Display full source metadata
                    st.write("**Source Metadata:**")
                    st.json(doc.metadata)
                    
                    # If the document seems to have structured content, try to display it
                    try:
                        # Check if the content might be JSON
                        if doc.page_content.strip().startswith('{') and doc.page_content.strip().endswith('}'):
                            st.write("**Parsed Content (JSON):**")
                            content_data = json.loads(doc.page_content)
                            st.json(content_data)
                    except:
                        # If parsing fails, just continue
                        pass
            
            # Generate and display response
            with st.spinner("Generating response..."):
                response = generate_response(query, docs)
                st.subheader("Answer")
                st.write(response)
    else:
        st.warning("Please enter a question.")