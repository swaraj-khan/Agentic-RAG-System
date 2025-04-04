# import basics
import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage

from supabase.client import Client, create_client
from langchain_core.tools import tool

# load environment variables
load_dotenv()  

# initiate supabase database
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model with your preferred HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")

# initiate vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents_contextual",
    query_name="match_documents",
)

# initiate large language model with Groq instead of OpenAI
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama3-70b-8192",
    temperature=0
)

# Create a hardcoded system prompt
system_template = """You are a sophisticated trading and investment assistant specializing in market analysis, trading strategies, risk management, and financial markets. Your knowledge is grounded in technical analysis, price action, chart patterns, trading psychology, and various methodologies across different markets. You answer questions ONLY on trading and investment topics.

When you use the retrieve tool, make sure to use the information from the retrieved documents.

CORE INSTRUCTIONS:
1. ALWAYS cite the specific book(s) you're referencing in your answer using the format: "According to [Book Title]..." or "In [Book Title], the author explains that..."
2. When multiple books provide relevant information, cite each of them clearly.
3. Base your answer primarily on the trading book information provided, not general knowledge.
4. Use professional trading terminology appropriate for an audience familiar with financial markets.
5. When discussing risk management or strategy, include practical application advice from the cited books.
6. Greet the user with a friendly greeting.

Remember: Every substantive answer MUST include at least one specific book citation to show the source of your information.

For relationship queries: Structure your answer with:
- Definition of both concepts
- Nature of their relationship with book citations
- Practical implications for trading
- Conditions where the relationship is strongest/weakest
"""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# create the tools
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

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    print("Agentic RAG System initialized. Type 'exit' to quit.")
    
    # Store conversation history
    conversation_history = []
    
    while True:
        user_input = input("\nEnter your question: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting the system. Goodbye!")
            break
        
        if not user_input.strip():
            continue
            
        print("\nProcessing your question...")
        
        # Add the user's message to the conversation history
        conversation_history.append(HumanMessage(content=user_input))
        
        try:
            # Pass the conversation history to the agent
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": conversation_history[:-1]  # All except current message
            })
            
            print("\nAnswer:")
            print(response["output"])
            
            # Add the agent's response to the conversation history
            conversation_history.append(AIMessage(content=response["output"]))
            
            # Keep conversation history manageable (last 10 messages)
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
        except Exception as e:
            print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main()