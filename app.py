# Import necessary libraries
import streamlit as st  # For building the web app
from langchain_groq import ChatGroq  # Chat model from Groq API
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper  # Academic research tools
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun  # Search tools
from langchain.agents import initialize_agent, AgentType  # LangChain agent setup
from langchain.callbacks import StreamlitCallbackHandler  # Handles AI thoughts in Streamlit UI

# 🔹 Step 1: Streamlit UI Setup

# Set the page title in Streamlit
st.title("🔎 LangChain - Chat with Search")
st.write("An AI chatbot that searches the web using LangChain & Groq. Try it out!")

# 🔹 Step 2: API Key Input
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""

st.session_state["groq_api_key"] = st.text_input(
    "Enter your Groq API Key:", 
    value=st.session_state["groq_api_key"], 
    type="password"
)

if not st.session_state["groq_api_key"]:
    st.warning("⚠️ Please enter your Groq API key to continue.")
    st.stop()

# 🔹 Step 3: Initialize External Search & Research Tools

# ArXiv API Wrapper (Fetches academic papers, limits to 1 result & max 200 characters)
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)  # Tool to query ArXiv

# Wikipedia API Wrapper (Fetches Wikipedia summaries, limits to 1 result & max 200 characters)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)  # Tool to query Wikipedia

# DuckDuckGo Search (General web search)
search = DuckDuckGoSearchRun(name="Search")

# 🔹 Step 4: Initialize Session State for Storing Chat History

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 🔹 Step 5: Process User Input

if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 🔹 Step 6: Define the Language Model (LLM) - Groq API
    llm = ChatGroq(
        groq_api_key=st.session_state["groq_api_key"],  # Use the entered API key
        model_name="Llama3-8b-8192",
        streaming=True
    )

    # Define tools available to the chatbot
    tools = [search, arxiv, wiki]

    # System message for structured responses
    system_message = (
        "You are an AI assistant that follows the Thought → Action → Observation pattern. "
        "Always output a well-structured response."
    )

    formatted_prompt = f"{system_message}\n\nUser: {prompt}"

    # 🔹 Step 7: Initialize the LangChain Agent

    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors="retry"
    )

    # 🔹 Step 8: Generate & Display AI Response

    raw_response = None
    try:
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            raw_response = search_agent.run(formatted_prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": raw_response})
            st.write(raw_response)
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        if raw_response:
            st.write("🛠️ Raw LLM Response:", raw_response)
