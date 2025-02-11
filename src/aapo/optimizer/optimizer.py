from typing import AsyncIterable, Protocol

from aapo.core.parameters import TraceNode, TrainablePrompt


class MetricProtocol(Protocol):
    async def forward(self, gt: str, pred: TraceNode) -> tuple[TraceNode, float]:
        """Calculates a score for a trace node"""
        ...


class PropagatorProtocol(Protocol):
    async def propagate(self, node: TraceNode) -> AsyncIterable[TraceNode]: ...


class JudgeProtocol(Protocol):
    async def update_prompt(
        self,
        trainable_prompt: TrainablePrompt,
        batch: list[TraceNode],
    ) -> TrainablePrompt: ...


class Optimizer:
    def __init__(
        self,
        propagator: PropagatorProtocol,
        judge: JudgeProtocol,
        metric: MetricProtocol,
    ):
        self.propagator = propagator
        self.judge = judge
        self.metric = metric
