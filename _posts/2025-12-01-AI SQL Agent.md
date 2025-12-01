---
layout: post
title: "AI SQL Agent – Natural Language to SQL Query Generator"
image: "/posts/SQL-AGENT-.png"
tags: [OpenAI, LangChain, Agents, SQL, PostgreSQL, Flask, LLM]
---

This project delivers a fully functional **AI-powered SQL Agent** capable of understanding **natural-language questions** and converting them into **optimized, safe SQL queries** with human-friendly explanations.

It integrates **LangChain Agents**, **OpenAI GPT-4.1** (or GPT-5 if available), **PostgreSQL**, and **Flask**, enabling intelligent and secure querying of structured data.

A live demo is deployed via **Render Cloud**, supporting real-time interactions through a simple web interface.

---

# Table of Contents
- [00. Project Overview](#overview)
  - [Context](#context)
  - [Actions](#actions)
  - [Results](#results)
- [01. System Design](#design)
- [02. Dataset & Schema](#schema)
- [03. SQL Agent Architecture](#architecture)
- [04. Prompt Engineering](#prompt)
- [05. Flask Web App](#flask)
- [06. Full Code](#code)
- [07. Discussion](#discussion)
- [Live Demo](#demo)

---

# 00. Project Overview <a name="overview"></a>

## Context <a name="context"></a>
Organizations often store structured data in SQL databases, yet **non-technical users cannot write SQL queries** to extract insights.

They need:

- Natural-language querying  
- Accurate & safe SQL conversion  
- Fast access to customer/transaction insights  
- A simple UI  

---

## Actions <a name="actions"></a>
I built a complete **AI SQL Agent** that:

- Loads database schema automatically  
- Converts questions into optimized SQL queries   
- Executes SQL through SQLAlchemy  
- Returns results + a human explanation  
- Provides a clean Flask UI  
- Supports deployment on Render  

---

## Results <a name="results"></a>

- Natural-language → SQL conversion  
- Accurate and schema-aware reasoning  
- Safe SQL enforcement (no dangerous operations)  
- Instant insights from PostgreSQL  
- Clean UI for real-world usability  
- Production-ready deployment  

---

# 01. System Design <a name="design"></a>

**Core Pipeline**

User → Flask UI  
→ LangChain SQL Agent  
→ Schema Aware SQL Generation  
→ SQLAlchemy Execution  
→ Response + Explanation

**Technologies**

- OpenAI GPT-4.1 (LLM reasoning)  
- LangChain SQL Agent Toolkit  
- PostgreSQL   
- SQLAlchemy  
- Flask  
- Render Cloud  

---

# 02. Dataset & Schema <a name="schema"></a>

Two tables were used:

### **1. grocery_db.customer_details**
- customer_id (int)  
- distance_from_store  
- gender  
- credit_score  

### **2. grocery_db.transactions**
- customer_id  
- transaction_id  
- transaction_date  
- product_area_id  
- sales_cost  
- num_items  

**Join:** customer_id

---

# 03. SQL Agent Architecture <a name="architecture"></a>

The SQL agent uses:

- **SQLDatabaseToolkit**  
- **ChatOpenAI model**  
- Strict SELECT-only constraints  
- Automatic schema understanding  
- Query generation & execution  

The system prompt ensures:

- Safe SQL  
- Schema-aware behavior  
- Clear reasoning steps  

---

# 04. Prompt Engineering <a name="prompt"></a>

Example rules enforced:


---

# 05. Flask Web App <a name="flask"></a>

The web interface allows:

- Asking natural-language questions  
- Viewing generated SQL  
- Viewing query results  
- Seeing model explanations  
- Clean responsive layout  

---

# 06. Full Code <a name="code"></a>

Key components include:

- `app.py` → Flask application  
- `agent/sql_agent_01.py` → SQL Agent logic  
- `agent/sql-agent-system-prompt.txt` → System rules  
- `requirements.txt` → Dependencies  

Full code available in the GitHub repo.

---

# 07. Discussion <a name="discussion"></a>

This project demonstrates:

- Practical use of AI agents  
- Safe LLM-to-SQL generation  
- Combining reasoning with real data  
- Creating useful business applications  

Next steps:

- Add authentication  
- Expand schema support  
- Add visualization layer  
- Add caching for performance  

---

# 🔗 Live Demo <a name="demo"></a>

👉 https://sql-agent-1-te6s.onrender.com/

# 🔗 GitHub Repository

👉 https://github.com/LShahmiri/SQL-Agent  
