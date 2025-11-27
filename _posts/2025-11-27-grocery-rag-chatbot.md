---
layout: post
title: "ABC Grocery AI Assistant – RAG Chatbot"
image: "/img/posts/ABC.png"
tags: [OpenAI, LLM, RAG, ChatBot, LangChain, ChromaDB, Streamlit, Python]
---

### ABC Grocery AI Assistant (RAG Chatbot)

This project is an AI-powered **Retrieval-Augmented Generation (RAG)** chatbot built to support customers and staff of **ABC Grocery**.  
It retrieves verified information from an internal help guide using **LangChain, ChromaDB and OpenAI embeddings**, ensuring grounded, hallucination-free answers.

---

## 00. Project Overview  

### Context  
ABC Grocery maintains an internal help guide covering store hours, delivery policies, membership, and in-store services.  
Staff often need to search through this document manually, and customers repeatedly ask the same questions via email or phone.

The goal was to build a **conversational assistant** that:

- Answers questions using **only** the approved help guide  
- Remembers user details within a session (e.g. name, previous questions)  
- Can be easily shared as a **web app** and showcased in my portfolio  

### Actions  

I designed and implemented a complete RAG pipeline:

- Parsed the internal help guide (`abc-grocery-help-desk-data.md`) and split it into semantically meaningful chunks using **markdown-aware text splitting**.  
- Embedded all chunks with **OpenAI’s `text-embedding-3-small`** model.  
- Stored embeddings in a local **ChromaDB** vector store with cosine similarity search.  
- Wrapped retrieval + generation in a **LangChain** pipeline with a strict system prompt that forbids the model from inventing policies.  
- Added **conversational memory** using `RunnableWithMessageHistory`, so the assistant can remember the user’s name and previous context within a session.  
- Built an interactive **Streamlit** front-end with a chat UI and sidebar explaining what the assistant can and cannot do.  
- Deployed the app on **Streamlit Cloud** and published the source code on GitHub.

### Results  

- Users can ask natural-language questions such as *“What’s your returns policy for meat and milk?”* or *“What time do you open on Sundays?”* and receive consistent, policy-correct answers.  
- The assistant **refuses** to answer questions outside the help guide (e.g. “How old am I?”) and instead shows a safe fallback message directing users to a human contact.  
- Session-level memory allows simple personalization: the assistant can remember the user’s name and refer back to earlier questions while still being fully grounded in the document.

### Growth / Next Steps  

- Extend the knowledge base to multiple documents (FAQ, HR policies, training manuals).  
- Add role-based views (customer vs. staff) with different answer styles.  
- Log anonymous queries for analytics and identify gaps in the help documentation.  
- Integrate authentication and connect to live backend APIs (e.g. real product availability).

---

## 01. Data Source – Internal Help Guide  

The chatbot is grounded on a single markdown document:

- **File:** `abc-grocery-help-desk-data.md`  
- **Structure:** headings such as `### Returns & Refunds`, `### Delivery & Pickup`, `### Store Hours`, etc.  
- **Processing:**  
  - Loaded with `TextLoader`  
  - Split using `MarkdownHeaderTextSplitter` so each section becomes a “document chunk” with its own header-based ID  

This structure ensures that retrieval returns **semantically coherent sections** instead of arbitrary text slices.

---

## 02. RAG Architecture  

### Embeddings & Vector Store  

- Embeddings: `OpenAIEmbeddings(model="text-embedding-3-small")`  
- Vector store: **Chroma** with cosine distance and persistent storage in `abc_vector_db_chroma/`.  
- Retrieval strategy:  
  - `similarity_score_threshold` with `k = 6`  
  - Score threshold of `0.25` to filter out irrelevant chunks  

### Prompting & Grounding  

The system prompt enforces strict rules:

- Only use information inside `<context> ... </context>`  
- Do **not** use external or world knowledge  
- If the answer is not in the context, return a fixed fallback message:  
  > “I don’t have that information in the provided context. Please email human@abc-grocery.com and they will be glad to assist you!.”

This gives **predictable, auditable behaviour** suitable for a real help-desk setting.

---

## 03. Conversational Memory  

To make the assistant feel more natural while staying safe:

- I used `RunnableWithMessageHistory` from LangChain.  
- A simple in-memory `ChatMessageHistory` store is keyed by a `session_id` generated in Streamlit.  
- The LLM sees previous turns as `history`, which it can use to:  
  - Remember the user’s name  
  - Keep track of which policy area we are discussing  
  - Provide follow-up clarification without re-asking everything  

Importantly, **history is never treated as a source of company facts** — only as personalization context.

---

## 04. Streamlit Interface  

The UI is implemented in `streamlit_app.py`:

- Wide-layout chat interface using `st.chat_message` and `st.chat_input`.  
- A sidebar titled **“ABC Grocery AI Assistant”** summarises what the bot can help with:  
  - Store hours & locations  
  - Delivery & pickup policies  
  - Membership & rewards  
  - Payment options  
  - In-store services  
  - Product availability (as described in the help guide)  
- “Clear conversation” button resets the Streamlit session state and the LangChain message history.  
- A concise header banner:  
  > “Welcome to ABC Grocery AI Assistant! Ask anything about ABC Grocery.”

The result is a **clean, production-style demo** that non-technical stakeholders can use immediately.

---

## 05. Tech Stack  

- **Language:** Python  
- **LLM & Embeddings:** OpenAI (GPT-5.1 via `ChatOpenAI`, `text-embedding-3-small`)  
- **RAG Framework:** LangChain  
- **Vector Store:** ChromaDB (cosine similarity)  
- **Frontend:** Streamlit  
- **Document Processing:** Markdown-aware splitting with `MarkdownHeaderTextSplitter`  

---

## 06. Live Demo & Code  

- **Live Demo:**  
  👉 [Try the Chatbot](https://grocery-rag-chatbot-with-memory.streamlit.app)

- **Source Code:**  
  👉 [View Code on GitHub](https://github.com/LShahmiri/Grocery-RAG-Chatbot)

---

If you’d like to explore how a RAG-based assistant can be adapted for your own help-desk or internal knowledge base, this project is a compact, end-to-end example of how to go from **raw markdown documentation** to a **usable AI assistant**.
