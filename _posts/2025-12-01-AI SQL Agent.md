---
layout: post
title: "AI SQL Agent – Natural Language to SQL Query Generator"
image: "/posts/SQLAI.png"
tags: [OpenAI, LangChain, Agents, SQL, PostgreSQL, Flask, LLM]
---

This project delivers a fully functional **AI-powered SQL Agent** capable of understanding **natural-language questions** and converting them into **optimized, safe SQL queries** with human-friendly explanations.

It integrates **LangChain Agents**, **OpenAI GPT-4.1**, **PostgreSQL**, and **Flask**, enabling intelligent and secure querying of structured data.

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
- [04. Flask Web App](#flask)
- [05. Full Code](#code)
- [06. Discussion](#discussion)
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

# 04. Flask Web App <a name="flask"></a>

The web interface allows:

- Asking natural-language questions  
- Viewing generated SQL  
- Viewing query results  
- Seeing model explanations  
- Clean responsive layout  

---

# 05. Full Code <a name="code"></a>

```python
##################################################################################################
# 01 - Bring in .env information
##################################################################################################

import os
from dotenv import load_dotenv
load_dotenv()

##################################################################################################
# 02 - Create the connection string for the postgres database
##################################################################################################

POSTGRES_URI = (f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
                f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DBNAME')}?sslmode=require")

##################################################################################################
# 03 - Create the database engine
##################################################################################################

import sqlalchemy as sa

# create the database engine
engine = sa.create_engine(POSTGRES_URI,
                          pool_pre_ping=True,
                          connect_args={"options": "-c statement_timeout=15000"} # 15 second timeout
                          ) 

# check the connection
with engine.connect() as conn:
    conn.exec_driver_sql("select 1")
    
##################################################################################################
# 04 - Setup the database connection
##################################################################################################

from langchain_community.utilities import SQLDatabase

db = SQLDatabase(engine=engine,
                 schema="grocery_db",
                 include_tables=["customer_details", "transactions"],
                 sample_rows_in_table_info=5)

print("Usable tables:", db.get_usable_table_names())

##################################################################################################
# 05 - Create our SQL AI Agent
##################################################################################################

from langchain_openai import ChatOpenAI

sql_agent = ChatOpenAI(model="gpt-4.1",
                       temperature=0)


##################################################################################################
# 06 - Build the SQL Toolkit and tools
##################################################################################################

from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=sql_agent)
tools = toolkit.get_tools()


##################################################################################################
# 07 - Bring In System Prompt
##################################################################################################

# bring in the system instructions
with open("agent/sql-agent-system-prompt.txt", "r", encoding="utf-8") as f:
    system_text = f.read()
    
##################################################################################################
# 08 - Create the Agent
##################################################################################################

from langchain.agents import create_agent

agent = create_agent(model=sql_agent,
                     tools=tools,
                     system_prompt=system_text)


##################################################################################################
# 09 - Run test queries through the agent and extract the response
##################################################################################################

from langchain_core.messages import HumanMessage

user_query = "on average which gender lives furtherest from the store"

result = agent.invoke({"messages": [HumanMessage(content=user_query)]})
print(result["messages"][-1].content)
```

---

# 06. Discussion <a name="discussion"></a>

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
### 🔗 Live Demo <a name="demo"></a>

👉 [APP](https://sql-agent-1-te6s.onrender.com/)

### 🔗 GitHub Repository <a name="GitHub"></a>

👉 [GitHub](https://github.com/LShahmiri/SQL-Agent)

