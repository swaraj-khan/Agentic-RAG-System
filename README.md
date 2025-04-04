# 🤖 Agentic RAG System for Trading Knowledge

An advanced Retrieval-Augmented Generation (RAG) system specialized for trading and investment knowledge. This system combines the power of Large Language Models with contextually-enhanced document retrieval to provide accurate, source-cited answers to trading questions.

## 🌟 Features

- 🧠 **Agentic RAG Architecture**: Uses tool-calling agents for smarter, context-aware responses
- 📚 **Multi-Format Document Processing**: Support for both EPUB and PDF trading books
- 🔍 **Contextual Chunking**: Claude-powered contextual enhancement of document chunks
- 💾 **Vector Database Integration**: Supabase + pgvector for efficient similarity search
- 🌐 **Multiple Interface Options**: 
  - Command-line interface
  - Streamlit web application with professional UI
- 🔄 **Model Flexibility**: Works with multiple LLMs (Groq Llama-3-70B, OpenAI, Claude)
- 📊 **Source Attribution**: All answers include citations to source books

## 🏗️ System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│             │    │              │    │             │    │             │
│ EPUB/PDF    │──▶│ Contextual    │──▶│ HuggingFace │──▶│  Supabase   │
│ Documents   │    │ Chunking     │    │ Embeddings  │    │  Vector DB  │
│             │    │              │    │             │    │             │
└─────────────┘    └──────────────┘    └─────────────┘    └──────┬──────┘
                                                                 │
                                                                 ▼
                   ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
                   │              │    │             │    │             │
                   │   LLM-based  │◀──┤ Agentic RAG │◀───┤ User Query  │
                   │   Response   │    │ Processing  │    │ Interface   │
                   │              │    │             │    │             │
                   └──────────────┘    └─────────────┘    └─────────────┘
```

## 🔧 Technical Stack

- **Frontend**: Streamlit with custom CSS/JS for enhanced UX
- **Vector Database**: Supabase (PostgreSQL + pgvector extension)
- **Embedding Models**: 
  - HuggingFace (`sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja`)
  - OpenAI (`text-embedding-3-small`)
- **LLM Options**:
  - Groq (`llama3-70b-8192`)
  - OpenAI (`gpt-4o`)
  - Anthropic (`claude-3-haiku-20240307`)
- **Document Processing**: PyTorch, EbookLib, PyPDF, BeautifulSoup4
- **Frameworks**: LangChain for agent orchestration

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/swaraj-khan/Agentic-RAG-System.git
   cd Agentic-RAG-System
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables in a `.env` file:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_key
   GROQ_API_KEY=your_groq_api_key
   ANTHROPIC_API_KEY=your_claude_api_key  # Optional, for contextual chunking
   OPENAI_API_KEY=your_openai_api_key     # Optional, for OpenAI embeddings
   ```

4. Set up Supabase:
   - Create a Supabase project and enable the pgvector extension
   - Create tables: `documents` and `documents_contextual`

## 📋 Usage

### 1️⃣ Process and Ingest Documents

#### For EPUB books (with contextual chunking):
```bash
# Place your EPUB files in the 'documents' folder
python ingest_in_db_contextual.py
```

#### For EPUB books (standard chunking):
```bash
# Place your EPUB files in the 'documents' folder
python ingest_in_db.py
```

#### For PDF documents:
```bash
# Place your PDF files in the 'documents' folder
python ingest_pdf.py
```

### 2️⃣ Run the System

#### Command-line Interface:
```bash
python agentic_rag.py
```

#### Streamlit Web Application:
```bash
streamlit run str2.py  # Advanced UI
# OR
streamlit run agentic_rag_streamlit.py  # OpenAI version
# OR
streamlit run agentic_rag_chunk.py  # Chunk visualization version
```

## 🧪 Advanced Features

### Contextual Chunking

The system uses Claude to enhance document chunks with contextual information:

```python
def contextualize_chunk(chunk: Document, doc_source: str) -> Document:
    # Get original chunk text and metadata
    chunk_text = chunk.page_content
    metadata = chunk.metadata.copy()
    
    # Create prompt for Claude
    system_prompt = """Your job is to provide concise context for a text chunk to improve retrieval..."""
    
    # Call Claude to contextualize the chunk
    response = claude_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    
    # Extract Claude's response and combine with original chunk
    context = response.content[0].text.strip()
    contextualized_text = f"{context}\n\n{chunk_text}"
    
    # Create a new document with the contextualized text
    return Document(page_content=contextualized_text, metadata=metadata)
```

### Agent-Based Retrieval

The system uses LLM agents to determine when and how to retrieve information:

```python
@tool
def retrieve(query: str):
    """Retrieve information related to a query from trading books."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized

# combine the tools and provide to the llm
tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)
```

## 📊 Performance Considerations

- **GPU Acceleration**: The system detects and utilizes GPU when available for faster embedding generation
- **Batch Processing**: Implements batched document processing to optimize memory usage
- **Rate Limiting**: Built-in rate limiting for API calls to avoid throttling
- **Configurable Context Windows**: Adjustable chunk sizes and retrieval parameters

## 🔐 Security Notes

- Store all API keys in environment variables, not in code
- The Supabase service role key has full access to your database - keep it secure
- Consider implementing user authentication for the Streamlit app in production

## 🛠️ Customization

### Modify Chunking Parameters

```python
# Adjust these parameters in the ingest scripts
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
```

### Change Embedding Model

```python
# For faster but less accurate embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# For higher accuracy
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### Use Different LLM

```python
# Llama 3 via Groq
llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model_name="llama3-70b-8192")

# OpenAI
llm = ChatOpenAI(model="gpt-4o")

# Claude
llm = ChatAnthropic(model="claude-3-sonnet-20240229")
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
