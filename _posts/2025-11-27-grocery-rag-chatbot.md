---
layout: post
title: "ABC Grocery AI Assistant – RAG Chatbot"
image: "/posts/ABC.png"
tags: [OpenAI, LLM, RAG, ChatBot, LangChain, ChromaDB, Streamlit, Python]
---

In this project, I built an AI-powered **Retrieval-Augmented Generation (RAG)** chatbot designed to support customers and staff of **ABC Grocery** by providing *accurate, fast, hallucination-free answers* using LangChain, ChromaDB, OpenAI embeddings, and Streamlit.

---

# Table of Contents
- [00. Project Overview](#overview)
  - [Context](#context)
  - [Actions](#actions)
  - [Results](#results)
  - [Growth / Next Steps](#growth)
- [01. System Design Overview](#system-design)
- [02. Document Processing & Vectorisation](#document-processing)
- [03. RAG Architecture](#rag-architecture)
- [04. Prompt Engineering & Safety Rules](#prompt)
- [05. Streamlit Chat UI](#ui)
- [06. Full Code](#code)
- [07. Discussion, Growth & Next Steps](#discussion)

---

# 00. Project Overview <a name="overview"></a>

## **Context** <a name="context"></a>
ABC Grocery maintains an internal help-desk document containing key operational information such as:
- delivery/pickup policies  
- store hours  
- membership programs  
- in-store services  
- payment options  

Store staff needed a **reliable chatbot** that:
- uses *only* company-approved information  
- remembers user preferences  
- prevents hallucinations  
- works like a real support assistant  

This led to building a **RAG chatbot** with strict grounding rules.

---

## **Actions** <a name="actions"></a>

I built a system that:

### ✔️ Loads an internal help-desk `.md` document  
### ✔️ Splits content into semantic chunks  
### ✔️ Embeds each chunk using **OpenAI text-embedding-3-small**  
### ✔️ Stores vectors in **ChromaDB**  
### ✔️ Retrieves relevant chunks during queries  
### ✔️ Runs an LLM with strict guardrails  
### ✔️ Includes message memory  
### ✔️ Provides a modern Streamlit chat UI  

---

## **Results** <a name="results"></a>

The chatbot successfully:
- Answers grocery-related questions *accurately*
- Uses **only the provided internal document**
- Avoids hallucinations  
- Remembers the user’s name and preferences  
- Provides a real-time chat-like experience  

---

## **Growth / Next Steps** <a name="growth"></a>

Next improvements could include:
- Multi-document ingestion (PDF, web pages, policy updates)
- Staff-only mode with secure login
- Analytics dashboard for customer questions
- Replacing ChromaDB with a scalable vector DB (e.g., Azure AI Search)

---

---

# 01. System Design Overview <a name="system-design"></a>

User → Streamlit Chat UI
→ RAG Pipeline
→ ChromaDB Vector Search
→ Prompt Template
→ OpenAI GPT-5
→ Response

Components:
- **LangChain** for orchestration  
- **MarkdownHeaderTextSplitter** for structured document chunking  
- **Chroma Vector DB**  
- **OpenAI Models** for embeddings + chat  
- **Streamlit** for UI  

---

# 02. Document Processing & Vectorisation <a name="document-processing"></a>

We load the internal help-desk file:

```python
raw_filename = "abc-grocery-help-desk-data.md"
loader = TextLoader(raw_filename, encoding="utf-8")
docs = loader.load()
text = docs[0].page_content
Then split based on headers:

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("###", "id")],
    strip_headers=True,
)
chunked_docs = splitter.split_text(text)
Generate embeddings:
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

Store vectors:
vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    persist_directory="abc_vector_db_chroma",
)

03. RAG Architecture <a name="rag-architecture"></a>

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 6, "score_threshold": 0.25},
)
RAG chain:
rag_answer_chain = (
    {
        "context": itemgetter("input") | retriever | RunnableLambda(format_docs),
        "input": itemgetter("input"),
        "history": itemgetter("history"),
    }
    | prompt_template
    | llm
)
04. Prompt Engineering & Safety Rules <a name="prompt"></a>

Key constraints:
1) ONLY answer using <context>  
2) NEVER use world knowledge  
3) If missing info → return fallback message  
4) History is for personalization only  
5) <context> overrides previous chat  
Fallback message:

“I don’t have that information in the provided context. Please email human@abc-grocery.com
 and they will be glad to assist you!.”
05. Streamlit Chat UI <a name="ui"></a>
Header:

st.markdown(
    "<div class='emoji-title'>🛒 ABC Grocery AI Assistant</div>",
    unsafe_allow_html=True
)


Sidebar:

with st.sidebar:
    st.markdown("## 🛒 ABC Grocery AI Assistant")


Chat loop:

user_input = st.chat_input("Type your question here...")

if user_input:
    resp = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )

06. Full Code (RAG + Streamlit) <a name="code"></a>

Full application code is provided below for transparency.

<PASTE YOUR ENTIRE STREAMLIT CODE HERE — already included above>

07. Discussion, Growth & Next Steps <a name="discussion"></a>

This system demonstrates how RAG can:

improve accuracy

avoid hallucination

provide contextual answers

personalize user experience

Future work:

integrate multi-store data

add user authentication

link to live inventory APIs

generate analytics on user questions

🔗 Live Demo

👉 Try the Chatbot

🔗 GitHub Repository

👉 View Code on GitHub

