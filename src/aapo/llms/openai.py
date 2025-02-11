from openai import AsyncOpenAI
from aapo.dto import Message, History


class OpenAIProvider:
    def __init__(self, model_name: str, **client_kwargs):
        self.client = AsyncOpenAI(**client_kwargs)
        self.model_name = model_name

    async def achat(self, history: list[Message]) -> Message:
        response = await self.client.chat.completions.create(
            messages=History.dump_python(history),
            model=self.model_name,
        )

        return Message.model_validate(response)
