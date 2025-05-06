import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
llm = ChatGoogleGenerativeAI(
    api_key='AIzaSyB3-CSt-cs5uY0XUSNT8b6e8lbsjX_Um7o',
    model='gemini-2.5-flash-latest',
    temperature=0.5
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

loader = TextLoader("Data/policy.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(split_docs, embedding, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever()

db = SQLDatabase.from_uri("sqlite:///Database/Amazon.db")

def rag_tool(query: str) -> str:
    """Use this tool for questions about policies, returns, and general information."""
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    return qa_chain.run(query)

def sql_tool(query: str) -> str:
    """Use this tool for questions about products, orders, delivery status, etc."""
    sql_generation_prompt = f"""
    You are an expert SQL query generator. Given the following database schema:
    
    Table: products
    - product_id (TEXT, PRIMARY KEY)
    - name (TEXT)
    - category (TEXT)
    - price (REAL)
    - description (TEXT)
    - stock_quantity (INTEGER)
    - rating (REAL)
    
    Table: orders
    - order_id (TEXT, PRIMARY KEY)
    - customer_id (TEXT)
    - order_date (TEXT)
    - total_amount (REAL)
    - status (TEXT)
    - shipping_address (TEXT)
    
    Table: order_items
    - id (INTEGER, PRIMARY KEY)
    - order_id (TEXT, FOREIGN KEY to orders.order_id)
    - product_id (TEXT, FOREIGN KEY to products.product_id)
    - quantity (INTEGER)
    - price (REAL)
    
    Table: delivery
    - delivery_id (TEXT, PRIMARY KEY)
    - order_id (TEXT, FOREIGN KEY to orders.order_id)
    - carrier (TEXT)
    - tracking_number (TEXT)
    - estimated_delivery_date (TEXT)
    - actual_delivery_date (TEXT)
    - delivery_status (TEXT)
    
    Generate a SQL query to answer this question: {query}
    Return ONLY the SQL query without any explanation.
    """
    
    sql_query = llm.invoke(sql_generation_prompt).content.strip()
    try:
        result = db.run(sql_query)

        result_interpretation_prompt = f"""
        Given the SQL query: {sql_query}
        And the result: {result}
        
        Please provide a human-readable interpretation of these results to answer the question: {query}
        """
        
        interpreted_result = llm.invoke(result_interpretation_prompt).content
        return interpreted_result
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"


tools = [
    Tool(
        name="RAG_Policy_Tool",
        func=rag_tool,
        description="Use this tool for questions about policies, returns, and general information."
    ),
    Tool(
        name="SQL_Database_Tool",
        func=sql_tool,
        description="Use this tool for questions about products, orders, delivery status, etc."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

def process_query(query: str) -> str:
    """Process a query using the appropriate tool."""
    result = agent.invoke(query)
    return result.content.strip()
