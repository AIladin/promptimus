from promptimus.dto import Message

from .openai import OpenAIProvider


class OllamaProvider:
    def __init__(self, model_name: str, base_url: str) -> None:
        self.client = OpenAIProvider(
            model_name=model_name,
            base_url=base_url,
            api_key="DUMMY",
        )

    async def achat(self, history: list[Message]) -> Message:
        return await self.client.achat(history)
