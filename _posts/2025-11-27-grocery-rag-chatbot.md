---
layout: post
title: "ABC Grocery AI Assistant – RAG Chatbot"
image: "/posts/ABC.png"
tags: [OpenAI, LLM, RAG, ChatBot, LangChain, ChromaDB, Streamlit, Python]
---

This project showcases an AI-powered **Retrieval-Augmented Generation (RAG)** chatbot designed for **ABC Grocery**.  
It retrieves internal help-desk information using **LangChain**, **ChromaDB**, and **OpenAI embeddings**, ensuring safe, accurate, and hallucination-free customer support.

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
- [05. Streamlit UI](#ui)
- [06. Full Code](#code)
- [07. Discussion](#discussion)

---

# 00. Project Overview <a name="overview"></a>

## Context <a name="context"></a>
ABC Grocery maintains a detailed internal policy document describing:

- Store opening times  
- Delivery & pickup processes  
- Payment options  
- Membership & rewards  
- In-store services  
- Product availability  

The company required a **safe and grounded AI assistant** that:

- Answers accurately using *only approved text*  
- Prevents hallucinations  
- Personalizes conversations  
- Provides an intuitive chat experience  

---

## Actions <a name="actions"></a>
I developed a complete RAG system that:

- Loads and processes the `.md` help-desk guide  
- Splits content with **MarkdownHeaderTextSplitter**  
- Embeds text using **OpenAI text-embedding-3-small**  
- Stores embeddings in **ChromaDB**  
- Retrieves relevant context through vector search  
- Uses OpenAI GPT-5 with strict grounding rules  
- Implements conversation memory  
- Provides an interactive **Streamlit chat UI**

---

## Results <a name="results"></a>

- Answers remain strictly grounded in internal documentation  
- No world-knowledge or hallucinations  
- Personalized chat using memory  
- Fully deployable via Streamlit Cloud  
- Reliable performance across many query types  

---

## Growth / Next Steps <a name="growth"></a>

Future additions:

- Multi-document ingestion  
- Staff/admin mode  
- Inventory API integration  
- Analytics dashboard  
- Role-based access control  
- Azure Cognitive Search integration  

---

# 01. System Design Overview <a name="system-design"></a>

**Pipeline:**

User → Streamlit UI  
→ RAG Pipeline  
→ ChromaDB Vector Search  
→ Prompt Template  
→ OpenAI GPT-5  
→ Response

**Core Components:**

- **LangChain** – RAG pipeline  
- **ChromaDB** – vector database  
- **OpenAI GPT-5** – reasoning model  
- **Memory** – personalized chat  
- **Streamlit** – front-end  

---

# 02. Document Processing & Vectorisation <a name="document-processing"></a>

### Load help guide

```python
raw_filename = "abc-grocery-help-desk-data.md"
loader = TextLoader(raw_filename, encoding="utf-8")
docs = loader.load()
text = docs[0].page_content
```

---

### Split into chunks

```python
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("###", "id")],
    strip_headers=True,
)
chunked_docs = splitter.split_text(text)
```

---

### Embed and persist vectors

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    persist_directory="abc_vector_db_chroma",
)
```

---

# 03. RAG Architecture <a name="rag-architecture"></a>

### Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 6, "score_threshold": 0.25},
)
```

---

### RAG answer chain

```python
rag_answer_chain = (
    {
        "context": itemgetter("input") | retriever | RunnableLambda(format_docs),
        "input": itemgetter("input"),
        "history": itemgetter("history"),
    }
    | prompt_template
    | llm
)
```

---

### Memory-enabled chain

```python
chain_with_history = RunnableWithMessageHistory(
    runnable=rag_answer_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

---

# 04. Prompt Engineering & Safety Rules <a name="prompt"></a>

The system uses strict grounding rules:

```
1. MUST answer ONLY from <context>.
2. MUST NOT use world knowledge.
3. History ONLY for personalization.
4. If context missing → fallback message:
   "I don’t have that information in the provided context.
    Please email human@abc-grocery.com and they will be glad to assist you!."
5. <context> overrides everything.
```

---

# 05. Streamlit UI <a name="ui"></a>

### Header

```python
st.markdown(
    "<div class='emoji-title'>🛒 ABC Grocery AI Assistant</div>",
    unsafe_allow_html=True
)
```

---

### Sidebar

```python
with st.sidebar:
    st.markdown("## 🛒 ABC Grocery AI Assistant")
```

---

### Chat loop

```python
user_input = st.chat_input("Type your question here...")

if user_input:
    resp = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )
```

---

# 06. Full Code <a name="code"></a>

```python
# FULL STREAMLIT APPLICATION
# (Paste your entire streamlit_app.py file here)
```

---

# 07. Discussion <a name="discussion"></a>

This project demonstrates:

- How RAG prevents hallucinations  
- How grounding rules ensure correct answers  
- How vector search improves retrieval accuracy  
- How memory enhances user experience  
- How internal documents become intelligent assistants  

Next steps include multi-document RAG, staff dashboards, and real-time API integration.

---

### 🔗 Live Demo  
👉 https://grocery-rag-chatbot-with-memory.streamlit.app

### 🔗 GitHub Repository  
👉 https://github.com/LShahmiri/Grocery-RAG-Chatbot

