import inspect
import os
from datetime import datetime

from init_rag import init_rag
from utils.configure_logger import configure_logging
from utils.print_stream import print_stream
from lightrag import LightRAG, QueryParam

async def main():
    rag = None
    INPUT_FOLDER = "./2026-01"
    api_key= "adalah-"
    model_llm = "openai/gpt-4o-mini"
    embed_model = "bge-m3"
    working_dir="adalah"
    try:
        rag = await init_rag(working_dir=working_dir,api_key=api_key, model= model_llm, embed_model=embed_model)
        if os.path.exists(INPUT_FOLDER):
            for root, dirs, files in os.walk(INPUT_FOLDER):
                for file_name in files:
                    if file_name.endswith(".txt"):
                        file_path = os.path.join(root, file_name)
                        print(f"Inserting content from: {file_path}")
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if content.strip():
                                await rag.ainsert(content)
        else:
            print(f"Warning: Folder {INPUT_FOLDER} not found.")

        print("\n=====================")
        print("Query mode: naive")
        print("=====================")

        resp = await rag.aquery(
            "What are the Memory Text? and the verse",
            param=QueryParam(mode="mix", stream=True),
        )

        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if rag:
            await rag.llm_response_cache.index_done_callback()
            await rag.finalize_storages()


if __name__ == "__main__":
    start_time = datetime.now()
    configure_logging()
    main()
    print("Done")
    end_time = datetime.now()

    print(f"Waktu yang dibutuhkan eksekusi ini adalah {end_time - start_time}")