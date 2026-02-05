import replicate
import asyncio

def replicate_llm_func(
    api_key: str,
    model: str = "openai/gpt-4o-mini",
):
    client = replicate.Client(api_token=api_key)

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        keyword_extraction: bool = False,
        **kwargs
    ) -> str:
        loop = asyncio.get_running_loop()
        history_messages = history_messages or []

        def _call_replicate():
            full_prompt = ""

            for m in history_messages:
                full_prompt += f"{m['role'].upper()}: {m['content']}\n"

            if keyword_extraction:
                full_prompt += (
                    "Extract entities and relations. "
                    "Return concise structured text only.\n\n"
                )

            full_prompt += prompt

            output = client.run(
                model,
                input={
                    "prompt": full_prompt,
                    "system_prompt": system_prompt
                        or "You are a precise information extraction assistant.",
                    "temperature": 0.1,
                    "max_tokens": 512,
                }
            )

            if output is None:
                return ""

            if isinstance(output, list):
                return "".join(output)

            return str(output)

        result = await loop.run_in_executor(None, _call_replicate)
        return result.strip()

    return llm_model_func
