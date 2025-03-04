from openai import AsyncOpenAI

from promptimus.dto import History, Message, MessageRole


class OpenAILike:
    def __init__(
        self,
        model_name: str,
        call_kwargs: dict | None = None,
        **client_kwargs,
    ):
        self.client = AsyncOpenAI(**client_kwargs)
        self.model_name = model_name
        self.call_kwargs = {} if call_kwargs is None else call_kwargs

    async def achat(self, history: list[Message]) -> Message:
        response = await self.client.chat.completions.create(
            messages=History.dump_python(history),
            model=self.model_name,
            **self.call_kwargs,
        )

        assert response.choices, response
        assert response.choices[0].message.content, response

        return Message(
            role=MessageRole.ASSISTANT, content=response.choices[0].message.content
        )
