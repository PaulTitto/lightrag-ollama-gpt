async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)