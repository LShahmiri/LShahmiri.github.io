---
layout: post
title: "ABC Grocery AI Assistant – RAG Chatbot"
image: "/posts/ABC.png"
tags: [OpenAI, LLM, RAG, ChatBot, LangChain, ChromaDB, Streamlit, Python]
---

This project is an AI-powered **Retrieval-Augmented Generation (RAG)** chatbot designed to support customers and staff of **ABC Grocery**.  
It retrieves verified information from an internal help-desk guide using **LangChain**, **ChromaDB**, and **OpenAI embeddings**, ensuring accurate and hallucination-free responses.

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

## Context <a name="context"></a>

ABC Grocery provides an internal help-desk document containing:

- Store hours  
- Delivery/pickup policies  
- Payment options  
- Rewards & membership  
- In-store services  
- Product availability  

The company needed a **safe, accurate, and fast AI assistant** that:

- Answers questions using **only approved internal content**  
- Prevents hallucinations  
- Remembers customer preferences  
- Provides a natural chat experience  

This led to building a **RAG-based chatbot** with strong grounding rules.

---

## Actions <a name="actions"></a>

I implemented a complete RAG system that:

- Loads the internal `.md` policy file  
- Splits content using **MarkdownHeaderTextSplitter**  
- Generates embeddings with **OpenAI text-embedding-3-small**  
- Saves vectors to **ChromaDB**  
- Retrieves relevant chunks using similarity search  
- Runs an LLM with strict grounding + safety guardrails  
- Supports conversation memory  
- Provides a clean **Streamlit chat UI**  

---

## Results <a name="results"></a>

The assistant:

- Responds quickly and accurately  
- Never uses external/world knowledge  
- Personalises conversation (remembers user name)  
- Handles real customer queries  
- Is fully deployable online via Streamlit  

---

## Growth / Next Steps <a name="growth"></a>

Future extensions could include:

- Multiple document ingestion  
- Staff-only admin mode  
- Integration with live inventory APIs  
- Conversation analytics dashboard  
- Role-based access control  
- Azure AI Search integration  

---

# 01. System Design Overview <a name="system-design"></a>

User → Streamlit UI
→ RAG Pipeline
→ ChromaDB Vector Search
→ Prompt Template
→ OpenAI GPT Model
→ Response

yaml
Copy code

Key components:

- **LangChain** for RAG pipeline  
- **ChromaDB** as vector store  
- **OpenAI GPT-5** as reasoning model  
- **Memory system** to personalise chat  
- **Streamlit** for front-end  

---

# 02. Document Processing & Vectorisation <a name="document-processing"></a>

Load help guide:

```python
raw_filename = "abc-grocery-help-desk-data.md"
loader = TextLoader(raw_filename, encoding="utf-8")
docs = loader.load()
text = docs[0].page_content
Split into chunks:

python
Copy code
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("###", "id")],
    strip_headers=True,
)
chunked_docs = splitter.split_text(text)
Embed and persist:

python
Copy code
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    persist_directory="abc_vector_db_chroma",
)
---
# 03. RAG Architecture <a name="rag-architecture"></a>
Retriever:

python
Copy code
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 6, "score_threshold": 0.25},
)
RAG chain:

python
Copy code
rag_answer_chain = (
    {
        "context": itemgetter("input") | retriever | RunnableLambda(format_docs),
        "input": itemgetter("input"),
        "history": itemgetter("history"),
    }
    | prompt_template
    | llm
)
Memory-enabled chain:

python
Copy code
chain_with_history = RunnableWithMessageHistory(
    runnable=rag_answer_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
---
# 04. Prompt Engineering & Safety Rules <a name="prompt"></a>
The system uses a robust guardrailed prompt:

pgsql
Copy code
1. MUST answer ONLY from <context>.
2. MUST NOT use world knowledge.
3. History is only for personalization.
4. If info missing → return fallback message.
5. <context> overrides everything.
Fallback message:

“I don’t have that information in the provided context. Please email human@abc-grocery.com and they will be glad to assist you!.”
---
# 05. Streamlit Chat UI <a name="ui"></a>
Header:

python
Copy code
st.markdown(
    "<div class='emoji-title'>🛒 ABC Grocery AI Assistant</div>",
    unsafe_allow_html=True
)
Sidebar:

python
Copy code
with st.sidebar:
    st.markdown("## 🛒 ABC Grocery AI Assistant")
Chat loop:

python
Copy code
user_input = st.chat_input("Type your question here...")

if user_input:
    resp = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )
---
# 06. Full Code <a name="code"></a>
Below is the complete Streamlit application code powering the chatbot.

python
Copy code
<PASTE YOUR FULL STREAMLIT CODE HERE — the exact code from your project>
---
# 07. Discussion, Growth & Next Steps <a name="discussion"></a>
This project demonstrates:

How RAG prevents hallucinations

How grounding rules keep responses safe

How vector search improves accuracy

How small documents can be turned into intelligent assistants

How conversation memory improves UX

Next steps include multi-document ingestion, staff dashboards, and API integrations.

🔗 Live Demo
👉 Try the Chatbot

🔗 GitHub Repository
👉 View Code on GitHub

