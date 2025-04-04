import streamlit as st
import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from supabase.client import Client, create_client
from langchain_core.tools import tool

# Set page configuration
st.set_page_config(
    page_title="Trading Assistant",
    page_icon="📈",
    layout="wide",
)

# Force dark theme for better readability
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #0e1117;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS for a clean interface resembling Claude
st.markdown("""
<style>
    /* Main container styling */
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        position: relative;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message.user {
        background-color: #2d3748;
        border-left: 5px solid #718096;
        color: #ffffff;
    }
    
    .chat-message.assistant {
        background-color: #1a202c;
        border-left: 5px solid #4299e1;
        color: #ffffff;
    }
    
    .chat-message.agent {
        background-color: #2d3748;
        border: 1px solid #4a5568;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 0.85em;
        color: #a0aec0;
    }
    
    /* Input area styling */
    .stTextArea textarea {
        min-height: 100px;
        border-radius: 10px;
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 20px;
        padding: 0.5rem 1rem;
        background-color: #4299e1 !important;
        color: white !important;
    }
    
    .stExpander {
        border-radius: 10px;
    }
    
    /* Make textareas auto-resize */
    textarea {
        overflow: hidden;
        min-height: 50px;
        resize: none;
    }
    
    /* Hide the default Streamlit footer */
    footer {
        visibility: hidden;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>

<script>
    // JavaScript to auto-resize text areas
    document.addEventListener('DOMContentLoaded', function() {
        const textareas = document.querySelectorAll('textarea');
        textareas.forEach(textarea => {
            textarea.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = (this.scrollHeight) + 'px';
            });
        });
    });
</script>
""", unsafe_allow_html=True)

# Load environment variables
@st.cache_resource
def load_env_vars():
    load_dotenv()
    return {
        "supabase_url": os.environ.get("SUPABASE_URL"),
        "supabase_key": os.environ.get("SUPABASE_SERVICE_KEY"),
        "groq_api_key": os.environ.get("GROQ_API_KEY"),
    }

env_vars = load_env_vars()

# Initialize Supabase client
@st.cache_resource
def init_supabase():
    supabase_url = env_vars["supabase_url"]
    supabase_key = env_vars["supabase_key"]
    return create_client(supabase_url, supabase_key)

supabase = init_supabase()

# Initialize embeddings model
@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")

embeddings = init_embeddings()

# Initialize vector store
@st.cache_resource
def init_vector_store():
    return SupabaseVectorStore(
        embedding=embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
    )

vector_store = init_vector_store()

# Initialize LLM
@st.cache_resource
def init_llm():
    return ChatGroq(
        api_key=env_vars["groq_api_key"],
        model_name="llama3-70b-8192",
        temperature=0
    )

llm = init_llm()

# Create the system prompt
system_template = """You are a sophisticated trading and investment assistant specializing in market analysis, trading strategies, risk management, and financial markets. Your knowledge is grounded in technical analysis, price action, chart patterns, trading psychology, and various methodologies across different markets.

Recent conversation history:
{chat_history}

Question: {input}

Information from trading books:
{context}

CORE INSTRUCTIONS:
1. ALWAYS cite the specific book(s) you're referencing in your answer using the format: "According to [Book Title]..." or "In [Book Title], the author explains that..."
2. When multiple books provide relevant information, cite each of them clearly.
3. Base your answer primarily on the trading book information provided, not general knowledge.
4. Use professional trading terminology appropriate for an audience familiar with financial markets.
5. When discussing risk management or strategy, include practical application advice from the cited books.

Remember: Every substantive answer about trading concepts MUST include at least one specific book citation to show the source of your information.
"""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create a callback handler to capture the agent's steps
class AgentCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.steps = []
        
    def on_agent_action(self, action, **kwargs):
        self.steps.append(f"🔍 Tool: {action.tool}\n📥 Input: {action.tool_input}")
        
    def on_tool_end(self, output, **kwargs):
        self.steps.append(f"📤 Output: {output}\n")

# Create retrieve tool
@tool
def retrieve(query: str):
    """Retrieve information related to a query from trading books."""
    try:
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized
    except Exception as e:
        return f"Error retrieving information: {str(e)}. Make sure your database contains trading books."

# Combine tools and create agent
tools = [retrieve]
agent = create_tool_calling_agent(llm, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    return_intermediate_steps=True,
    handle_parsing_errors=True
)

# Initialize session state for chat history and form submission
if "messages" not in st.session_state:
    st.session_state.messages = []

if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

def submit_message():
    st.session_state.form_submitted = True

# Title for the app
st.title("Trading Book Assistant")

# Sidebar for controls
with st.sidebar:
    st.header("Options")
    
    # Clear history button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user"><p>{message["content"]}</p></div>', unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f'<div class="chat-message assistant"><p>{message["content"]}</p></div>', unsafe_allow_html=True)
        elif message["role"] == "agent":
            with st.expander("Agent Thought Process", expanded=False):
                st.markdown(f'<div class="chat-message agent"><p>{message["content"]}</p></div>', unsafe_allow_html=True)
        elif message["role"] == "system":
            st.markdown(f'<div style="padding: 10px; background-color: #1e3a8a; color: white; border-radius: 5px; margin-bottom: 10px;">{message["content"]}</div>', unsafe_allow_html=True)

# Input area at the bottom
user_input = st.text_area("Type your question here...", height=100, key="user_input", 
                          placeholder="Ask about trading concepts, price action, chart patterns...",
                          on_change=submit_message)

# Add JavaScript to handle Enter key press
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    const textarea = document.querySelector('textarea');
    textarea.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const submitButton = document.querySelector('button[kind="primaryFormSubmit"]');
            if (submitButton) {
                submitButton.click();
            }
        }
    });
});
</script>
""", unsafe_allow_html=True)

# Send button
if st.button("Send", key="send_button") or st.session_state.form_submitted:
    if user_input:
        current_input = user_input  # Store the current input value
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": current_input})
        
        # Convert message history to LangChain format
        chat_history = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
        
        # Create callback handler to capture agent steps
        handler = AgentCallbackHandler()
        
        # Process the query
        with st.spinner("Thinking..."):
            try:
                # Invoke the agent
                response = agent_executor.invoke(
                    {
                        "input": current_input,
                        "chat_history": chat_history,
                        "context": "",  # This will be filled by the retrieve tool
                    },
                    config={"callbacks": [handler]}
                )
                
                # Add agent steps to chat history if there are any
                if handler.steps:
                    agent_thoughts = "\n".join(handler.steps)
                    st.session_state.messages.append({
                        "role": "agent",
                        "content": f"Agent Reasoning:\n\n{agent_thoughts}"
                    })
                
                # Add agent response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["output"]
                })
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}. Please try asking in a different way."
                })
        
        # Reset form submitted state
        st.session_state.form_submitted = False
        
        # Force rerun to refresh the UI and clear the input
        st.rerun()

# JavaScript to make textareas auto-resize and handle Enter key
st.markdown("""
<script>
    // Auto-resize textareas
    const textareas = document.getElementsByTagName('textarea');
    for (let i = 0; i < textareas.length; i++) {
        textareas[i].setAttribute('style', 'height: auto;');
        textareas[i].addEventListener('input', OnInput, false);
    }

    function OnInput() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    }

    // Handle Enter key to submit
    document.addEventListener('keydown', function(e) {
    if (e.target.tagName.toLowerCase() === 'textarea' && e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        
        // Find the closest form element
        const submitButton = document.querySelector('button[kind="primaryFormSubmit"]');
        if (submitButton) {
            submitButton.click();
        }
    }
});
</script>
""", unsafe_allow_html=True)

# Add custom Streamlit component for handling Enter key
components_js = """
<script>
// Add this at the top
if (window.keydownListenerAdded) return;
window.keydownListenerAdded = true;

// Create a custom event listener for the Enter key
document.addEventListener('keydown', function(e) {
    if (e.target.tagName.toLowerCase() === 'textarea' && e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        
        // Find the Send button
        const buttons = document.querySelectorAll('button');
        for (const button of buttons) {
            if (button.innerText.includes('Send')) {
                button.click();
                break;
            }
        }
    }
});
</script>
"""

st.components.v1.html(components_js, height=0)

