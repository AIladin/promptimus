from openai import AsyncOpenAI

from promptimus.dto import History, Message, MessageRole, ToolRequest


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

    async def achat(
        self, history: list[Message], **kwargs
    ) -> Message | list[ToolRequest]:
        response = await self.client.chat.completions.create(
            messages=History.dump_python(history),
            model=self.model_name,
            **dict(self.call_kwargs, **kwargs),
        )

        assert response.choices, response

        if raw_tool_calls := response.choices[0].message.tool_calls:
            tool_calls = [
                ToolRequest.model_validate(raw_tool_request, from_attributes=True)
                for raw_tool_request in raw_tool_calls
            ]
        else:
            tool_calls = None

        return Message(
            role=MessageRole.ASSISTANT,
            content=response.choices[0].message.content or "",
            tool_calls=tool_calls,
        )
