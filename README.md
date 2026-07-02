# Generative AI & Advanced LLM Orchestration Hub

A comprehensive, production-oriented repository showcasing end-to-end implementations of modern Generative AI engineering patterns. This hub covers everything from fundamental data pipeline steps to stateful agent architectures, parameter-efficient fine-tuning (PEFT), and multi-framework retrieval-augmented generation (RAG).

## 🚀 Core Architecture & Modules

The repository is organized into distinct functional layers representing a production-ready AI development lifecycle:

### 1. Data Processing & Knowledge Foundations
* **Data Ingestion:** Pipelines to import, process, and clean diverse raw data formats before sending them to embedding engines.
* **Data Transformer:** Scalable chunking, splitting strategies, and metadata extraction optimized for maintaining context in retrieval systems.
* **Embeddings & VectorDB:** Practical integration of state-of-the-art embedding models using vector databases for rapid semantic search.

### 2. Conversational Frameworks & Memory
* **LangChain Updated & LCEL:** Mastery of LangChain Expression Language (LCEL) to construct declarative, highly optimizable, and easily debuggable custom chains.
* **OpenAI & Hugging Face Integrations:** Benchmarking and implementing both closed-source APIs and open-source models natively or via LangChain wrappers.
* **Gen AI & OpenAI Chatbots:** Core architectures supporting long-form conversational dynamics and persistent multi-session history storage.

### 3. Stateful Graph-Based Agents
* **LangGraph-main:** Stateful, multi-agent frameworks built to handle cyclical workflows, persistent memory threads, human-in-the-loop interactions, and complex decision-making loops.
* **Search Engine Tools + Agents:** Tool-use (function calling) paradigms enabling models to autonomously fetch real-time data from search engines and custom tools.

### 4. Advanced RAG (Retrieval-Augmented Generation) Architectures
* **RAG GROQ API Project:** Blazing-fast inference pipelines utilizing Groq's LPU architecture for near-zero latency retrieval systems.
* **RAG + Q&A + Chat History:** Complete production RAG systems with conversational context condensation (compressing dialogue history into search queries).
* **GraphDB:** Moving beyond semantic similarity toward Structured RAG by leveraging Knowledge Graphs for entity-relationship reasoning.

### 5. Model Fine-Tuning (PEFT)
* **Gemma_Finetuning.ipynb & finetuning/:** Hands-on implementation of Parameter-Efficient Fine-Tuning (PEFT) using **LoRA (Low-Rank Adaptation)** on Google's Gemma models. Optimizes task-specific accuracy while maintaining minimal memory profiles and preventing catastrophic forgetting.

### 6. Local & Edge AI
* **Ollama + Mini Project:** Deploying, serving, and orchestrating smaller open-source language models fully locally for specialized, privacy-first applications.

---

## 🛠️ Tech Stack & Ecosystem

* **Orchestration Frameworks:** LangChain, LangGraph, LCEL
* **Model Ecosystems:** OpenAI, Hugging Face, Google Gemma, Ollama (Local LLMs)
* **Inference Accelerators:** Groq API
* **Vector & Storage Networks:** AstraDB / VectorDB, Graph Databases
* **Development Tools:** Python, Jupyter Notebooks

---

## ⚙️ Setup & Local Development

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/harshit11purohit/Generative-AI.git](https://github.com/harshit11purohit/Generative-AI.git)
   cd Generative-AI
