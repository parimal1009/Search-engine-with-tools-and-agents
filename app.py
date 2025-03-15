# Import necessary libraries
import streamlit as st  # For building the web app
from langchain_groq import ChatGroq  # Chat model from Groq API
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper  # Academic research tools
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun  # Search tools
from langchain.agents import initialize_agent, AgentType  # LangChain agent setup
from langchain.callbacks import StreamlitCallbackHandler  # Handles AI thoughts in Streamlit UI
import os  # For environment variable handling
from dotenv import load_dotenv  # Load environment variables from .env file

# Load environment variables from .env file
load_dotenv()

# Get Groq API Key from environment variables
API_KEY = os.getenv("GROQ_API_KEY")

# If API key is missing, display an error and stop execution
if not API_KEY:
    st.error("❌ Error: GROQ API Key not found! Make sure it's set in your .env file as 'GROQ_API_KEY'.")
    st.stop()

# 🔹 Step 1: Initialize External Search & Research Tools

# ArXiv API Wrapper (Fetches academic papers, limits to 1 result & max 200 characters)
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)  # Tool to query ArXiv

# Wikipedia API Wrapper (Fetches Wikipedia summaries, limits to 1 result & max 200 characters)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)  # Tool to query Wikipedia

# DuckDuckGo Search (General web search)
search = DuckDuckGoSearchRun(name="Search")

# 🔹 Step 2: Streamlit UI Setup

# Set the page title in Streamlit
st.title("🔎 LangChain - Chat with Search")
st.write("An AI chatbot that searches the web using LangChain & Groq. Try it out!")

# 🔹 Step 3: Initialize Session State for Storing Chat History

# If "messages" does not exist in session state, initialize it with a welcome message
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 🔹 Step 4: Process User Input

# Capture user input from the chat box
if prompt := st.chat_input(placeholder="Ask me anything..."):
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)  # Display user message in chat

    # 🔹 Step 5: Define the Language Model (LLM) - Groq API
    llm = ChatGroq(
        groq_api_key=API_KEY,  # Automatically load API key from .env
        model_name="Llama3-8b-8192",  # LLM model from Groq
        streaming=True  # Enable streaming response
    )

    # Define tools available to the chatbot (Search, Wikipedia, ArXiv)
    tools = [search, arxiv, wiki]

    # Manually define a system message to guide the AI's response behavior
    system_message = (
        "You are an AI assistant that follows the Thought → Action → Observation pattern. "
        "Always output a well-structured response."
    )
    
    # Combine system message with user input
    formatted_prompt = f"{system_message}\n\nUser: {prompt}"

    # 🔹 Step 6: Initialize the LangChain Agent

    search_agent = initialize_agent(
        tools=tools,  # Provide search and research tools
        llm=llm,  # Use the defined Groq Chat model
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # Ensures structured responses
        handle_parsing_errors="retry"  # If parsing fails, retry execution
    )

    # 🔹 Step 7: Generate & Display AI Response

    raw_response = None  # Variable to store AI output for debugging
    try:
        with st.chat_message("assistant"):  # Display assistant's response in chat
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)  # Callback handler for live updates
            raw_response = search_agent.run(formatted_prompt, callbacks=[st_cb])  # Run agent with formatted prompt
            st.session_state.messages.append({"role": "assistant", "content": raw_response})  # Save response
            st.write(raw_response)  # Display response
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")  # Show error message if something goes wrong
        if raw_response:
            st.write("🛠️ Raw LLM Response:", raw_response)  # Debugging output if available
