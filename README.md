# LangChain - Chat with Search

## 📌 Overview
This project is a chatbot powered by **LangChain** and **Groq API**, enabling users to search the web, academic papers (ArXiv), and Wikipedia for relevant information. It utilizes an intelligent agent to generate structured responses based on real-time search results.

## 🚀 Features
- **Conversational AI**: Chatbot responds to user queries in a structured manner.
- **Web Search**: Uses DuckDuckGo to find real-time web results.
- **Academic Research**: Fetches research papers from ArXiv.
- **Wikipedia Lookup**: Retrieves concise summaries from Wikipedia.
- **LangChain Agent**: Implements a structured Thought → Action → Observation workflow.
- **Streaming Responses**: Utilizes Groq API for real-time responses.

## 📜 How It Works
1. The chatbot initializes search and research tools (ArXiv, Wikipedia, DuckDuckGo).
2. It maintains chat history using Streamlit's session state.
3. When a user inputs a query, the chatbot:
   - Uses LangChain’s **Structured Chat Agent**.
   - Queries external sources for accurate answers.
   - Generates a well-structured response.
4. The chatbot's responses are displayed in the Streamlit interface.

## 📌 File Structure
```
📂 langchain-chatbot
 ├── 📄 app.py                 # Main Streamlit application
 ├── 📄 requirements.txt       # List of dependencies
 ├── 📄 .env                   # Environment variables (not included in repo)
 ├── 📂 utils                  # Helper functions (if any)
 ├── 📜 README.md              # Project documentation
```

## 🔧 Dependencies
- **LangChain**: Framework for building LLM-powered applications.
- **Streamlit**: Interactive UI for chatbot interface.
- **Groq API**: LLM for generating AI responses.
- **DuckDuckGo Search API**: Fetches real-time web results.
- **ArXiv API**: Retrieves academic papers.
- **Wikipedia API**: Provides summaries of Wikipedia pages.

## 🚀 Future Enhancements
- Integrate additional AI models for improved responses.
- Expand search capabilities beyond Wikipedia & ArXiv.
- Optimize response time with caching mechanisms.
- Deploy on a cloud platform (e.g., AWS, Google Cloud, or Streamlit Sharing).


## 📜 License
This project is licensed under the **MIT License**.
