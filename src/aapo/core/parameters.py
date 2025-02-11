from aapo.llms import ProviderProtocol
from typing import Optional, Self
from aapo.dto import Message, MessageRole
from aapo.utils import DEFAULT_SEPARATOR


class TraceNode:
    history: list[Message]
    parents: list["TraceNode"]
    _response: Optional[Message]
    executor: Optional["TrainablePrompt"]
    scores: list[float]

    def __init__(
        self,
        history: list[Message],
    ):
        self._response = None
        self.history = history
        self.executor = None
        self.scores = []
        self.parents = []

    def branch(self) -> Self:
        assert self._response is None and self.parents is None
        return self.__class__(self.history)

    @property
    def response(self) -> Message:
        assert self._response is not None
        return self._response

    @response.setter
    def response(self, val) -> None:
        self._response = val

    def avg_score(self) -> float:
        assert self.scores
        return sum(self.scores) / len(self.scores)

    @classmethod
    def compose(
        cls,
        prompt: str,
        context_sources: Optional[list[tuple[str, "TraceNode"]]] = None,
        history: Optional[list[Message]] = None,
        sep: str = DEFAULT_SEPARATOR,
    ) -> Self:
        if context_sources is None:
            context_sources = []

        context = []
        parents = []
        for header, node in context_sources:
            assert node._response is not None
            context.append(f"{header}\n\n{node._response.content}")
            parents.append(node)

        if history is None:
            history = []

        history = history + [
            Message(role=MessageRole.USER, content=sep.join(context) + sep + prompt)
        ]

        obj = cls(history)
        obj.parents = parents
        return obj


class TrainablePrompt:
    def __init__(
        self,
        prompt: str,
        provider: ProviderProtocol,
        trainable: bool = True,
    ) -> None:
        self.prompt = prompt
        self.provider = provider
        self.trainable = trainable

    async def forward(self, trace: TraceNode) -> TraceNode:
        trace = trace.branch()
        trace.executor = self

        prediction = await self.provider.achat(
            [Message(role=MessageRole.SYSTEM, content=self.prompt)] + trace.history
        )

        trace._response = prediction
        return trace
