

## 🔹 Project Description

**LightRAG Replicate + Ollama Integration**

This project provides a clean and extensible implementation of **LightRAG** using:

* **LLM inference via Replicate** (e.g. `openai/gpt-4o-mini`)
* **Local embeddings via Ollama** (e.g. `bge-m3`)
* **Graph-based Retrieval-Augmented Generation (LightRAG)**

It is designed for document ingestion, entity–relation extraction, and hybrid graph + vector retrieval in research and production environments.


## Initialization Example

```python
rag = await init_rag(
    working_dir=working_dir,
    api_key=api_key,
    model=model_llm,
    embed_model=embed_model,
)
```

### Parameters

| Parameter     | Description                                              |
| ------------- | -------------------------------------------------------- |
| `working_dir` | Directory for LightRAG storage (graph, vector DB, cache) |
| `api_key`     | Replicate API token used for LLM inference               |
| `model`       | Replicate model ID (e.g. `openai/gpt-4o-mini`)           |
| `embed_model` | Ollama embedding model (e.g. `bge-m3`)                   |


## Key Features

* **Graph-based RAG** using entity and relationship extraction
* **Hybrid retrieval** (knowledge graph + vector similarity)
* **Local embedding inference** with Ollama (no external embedding API)
* **Cloud LLM inference** via Replicate
* **Async-first design**, compatible with LightRAG pipelines
*  Ready for research projects, NLP experiments, and academic use


## Requirements

* Python 3.10+
* LightRAG
* Ollama (running locally)
* Replicate API token
