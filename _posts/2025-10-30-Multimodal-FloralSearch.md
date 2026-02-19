---
layout: post
title: " Multimodal Flower Arrangement Query & Image Retrieval"
image: "/posts/FLOWERMULTI1.png"
tags: [Multimodal AI, Vision-Language, OpenCLIP, ChromaDB, GPT-4o, Streamlit, HuggingFace]
---

# Multimodal Flower Arrangement Query & Image Retrieval  
### A Vision–Language AI for Flower Search, Matching & Bouquet Recommendations

This project is a **multimodal AI system** that retrieves the most visually similar flowers to a user’s natural-language description and then generates **professional bouquet recommendations** using OpenAI’s GPT-4o.

---

#  Table of Contents
- [00. Project Overview](#overview)
- [01. System Architecture](#architecture)
- [02. Dataset & Embeddings](#dataset)
- [03. Multimodal LLM Pipeline](#pipeline)
- [04. Streamlit Web Application](#ui)
- [05. Full Capabilities](#capabilities)
- [06. Discussion](#discussion)

---

# 00. Project Overview <a name="overview"></a>

This application combines:

-  **OpenCLIP image embeddings**
- **ChromaDB vector search**
-  **Flowers-102 dataset**
-  **GPT-4o multimodal reasoning**
-  **A modern Streamlit UI**
-  **AI Query Validator (GPT-4o-mini)**

Users can write a natural language query such as:

> “Elegant pink lilies with soft petals for birthday gift.”

The system retrieves the most visually similar flowers and provides **personalized bouquet arrangement suggestions**.

---

# 01. System Architecture <a name="architecture"></a>

**Pipeline Overview**

User Query  
→ Query Validator (GPT-4o-mini)  
→ OpenCLIP Text Embedding  
→ ChromaDB Vector Search  
→ Retrieve Top Flower Images  
→ GPT-4o Multimodal Florist Recommendation  
→ Streamlit Output  

---

# 02. Dataset & Embeddings <a name="dataset"></a>

###  Flowers-102 (HuggingFace)
Used for visual similarity search.  
Images are cached locally and indexed in **ChromaDB**.

###  Embedding Model
**OpenCLIPEmbeddingFunction**  
Used for both:

- Flower images  
- User text descriptions  

---

# 03. Multimodal LLM Pipeline <a name="pipeline"></a>

GPT-4o receives:

- Text description  
- Two retrieved flower images (Base64)  

And outputs:

- Bouquet arrangement ideas  
- Floral combinations  
- Color harmony explanation  
- Personalized stylist suggestions  

---

# 04. Streamlit Web Application <a name="ui"></a>

### Key UI Features
- Pink/Gold gradient modern dashboard  
- Animated search box  
- 2-column flower gallery  
- Highlighted “Bouquet Suggestion” box  
- Error handling + input filtering  

### Query Validator  
Rejects non-floral queries:

❌ “hi”  
❌ “what is your name”  
❌ “translate this sentence”  

Only real flower descriptions are processed.

---

# 05. Full Capabilities <a name="capabilities"></a>

| Feature | Description |
|--------|-------------|
|  Image Retrieval | CLIP-based similarity search |
|  Recommendations | GPT-4o florist-style suggestions |
|  Natural Language | Free-form text input |
|  Validation | GPT-4o-mini query filtering |
|  Real-Time | Streamlit interactive UI |
|  Dataset | Flowers-102 |

---

# 06. Discussion <a name="discussion"></a>

This project demonstrates:

- Vision–Language fusion  
- Vector retrieval with embeddings  
- Multimodal prompting  
- High-quality UI engineering  
- Real-world assistant for event planners & florists  

---

### 🔗 GitHub Repository <a name="GitHub"></a>

👉 [GitHub](https://github.com/LShahmiri/Multimodal-FloralSearch)
---
