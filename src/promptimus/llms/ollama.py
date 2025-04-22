from promptimus.dto import Message, ToolRequest

from .openai import OpenAILike


class OllamaProvider:
    def __init__(self, model_name: str, base_url: str) -> None:
        self.client = OpenAILike(
            model_name=model_name,
            base_url=base_url,
            api_key="DUMMY",
        )

    async def achat(
        self, history: list[Message], **kwargs
    ) -> Message | list[ToolRequest]:
        return await self.client.achat(history, **kwargs)
