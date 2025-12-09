# Multimodal Flower Arrangement Query & Image Retrieval  
### A Vision–Language AI for Flower Search, Matching, and Bouquet Recommendations

This project is a **multimodal AI system** that retrieves the most visually similar flowers to a user’s natural-language description and then generates **personalized bouquet recommendations** using OpenAI’s GPT-4o.

It integrates:

- 🖼 **OpenCLIP embeddings** for image understanding  
- 🔍 **ChromaDB vector search** for similarity retrieval  
- 🌺 **Flowers-102 dataset** (HuggingFace)  
- 🤖 **GPT-4o Vision + Text** for florist-style recommendations  
- 🎨 **Modern Streamlit UI** for a smooth user experience  
- 🛡️ **AI Query Validator** to ensure only flower-related queries are processed  

---

## 🚀 Key Features

### **1. Text-Based Flower Search**
Users freely describe flowers, e.g.:

- “pink flowers with soft petals”
- “yellow round-petal flower”
- “elegant red flowers for a birthday gift”

The system embeds the text, compares it with image embeddings, and retrieves the closest visual matches.

---

### **2. Intelligent Image Retrieval (OpenCLIP + ChromaDB)**

The backend performs:

- Preprocessing and loading of the Flowers-102 dataset  
- On-disk caching of images  
- Embedding using OpenCLIP  
- Storing + searching vectors with ChromaDB  

This allows **fast and accurate text-to-image retrieval**.

---

### **3. Multimodal LLM Recommendations (GPT-4o)**

The LLM receives:

- User's flower description  
- Top 2 matched images (Base64 encoded)  

And produces:

- High-quality bouquet arrangements  
- Color harmony suggestions  
- Flower combinations  
- Personalized gift messages  

Perfect for event planning, florists, gifting, or creative design inspiration.

---

### **4. AI Query Validator (GPT-4o-mini)**

To maintain product quality, the system rejects irrelevant queries:

❌ “hi”  
❌ “play music”  
❌ “translate this sentence”  

Only genuine *flower descriptions* are accepted.

---

## 🖼 Streamlit UI

The interface includes:

- Gradient header design  
- Search box with validation  
- Display of matched images in a 2-column layout  
- A highlighted suggestion panel  
- Full mobile and desktop responsiveness  

Modern, colorful, elegant — suitable for production demos or portfolios.

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Embeddings | OpenCLIP |
| Vector DB | ChromaDB |
| LLM | OpenAI GPT-4o & GPT-4o-mini |
| Dataset | HuggingFace Flowers-102 |
| Frontend | Streamlit |
| Backend Language | Python |

---

## 📂 Repository Structure
/project
│
├── app.py # Streamlit application
├── data/ # ChromaDB persistent storage
├── images/ # Cached flower images
├── requirements.txt
└── README.md
/project
│
├── app.py # Streamlit application
├── data/ # ChromaDB persistent storage
├── images/ # Cached flower images
├── requirements.txt
└── README.md

---

## 🌼 Why This Project Matters

This project demonstrates a real-world **multimodal AI pipeline**:

- Vision + language fusion  
- Vector search + embeddings  
- Prompt engineering  
- Interactive applications  
- OpenAI GPT-4o multimodal reasoning  

It showcases how AI can support **creative industries** such as floristry, gifting, branding, and design.

---

## 📸 Demo Preview  
*(Add screenshots here if you want — UI looks beautiful!)*

---


