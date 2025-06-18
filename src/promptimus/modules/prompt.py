from promptimus.core import Module, Parameter
from promptimus.dto import Message, MessageRole


class Prompt(Module):
    def __init__(
        self,
        prompt: str | None,
        role: MessageRole | str = MessageRole.SYSTEM,
    ) -> None:
        super().__init__()
        self.prompt = Parameter(prompt)
        self.role = Parameter(role)

    async def forward(
        self,
        history: list[Message] | None = None,
        provider_kwargs: dict | None = None,
        **kwargs,
    ) -> Message:
        if history is None:
            history = []

        prediction = await self.llm.achat(
            [
                Message(
                    role=self.role.value, content=self.prompt.value.format_map(kwargs)
                )
            ]
            + history,
            **(provider_kwargs or {}),
        )
        return prediction
