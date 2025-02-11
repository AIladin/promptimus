import asyncio

from aapo.core import Module, TraceNode
from aapo.llms import ProviderProtocol

from .precision import Precision
from .recall import Recall


class SemanticF1Score(Module):
    def __init__(self, provider: ProviderProtocol) -> None:
        super().__init__()
        self.precision = Precision(provider=provider)
        self.recall = Recall(provider=provider)

    async def ascore(self, gt: str, pred: TraceNode) -> tuple[TraceNode, float]:
        (_, precision), (_, recall) = await asyncio.gather(
            self.precision.forward(gt, pred), self.recall.forward(gt, pred)
        )

        return pred, 2 * precision * recall / (precision + recall)
