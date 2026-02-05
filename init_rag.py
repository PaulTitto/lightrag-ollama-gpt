import os
import replicate
from functools import partial
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc

from utils.replicate_llm import llm_model_func


async def init_rag(working_dir,api_key: str, model: str, embed_model: str) -> LightRAG:
    rag = LightRAG(
        working_dir=f"./{working_dir}",
        llm_model_func=llm_model_func(api_key, model), 
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,
                embed_model=embed_model,
                host="http://ollama:11434"
            ),
        ),
    )

    await rag.initialize_storages()
    return rag