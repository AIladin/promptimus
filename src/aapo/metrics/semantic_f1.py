import asyncio

from aapo.llms import ProviderProtocol

from .precision import Precision
from .recall import Recall


class SemanticF1Score:
    def __init__(self, provider: ProviderProtocol) -> None:
        self.precision = Precision(provider=provider)
        self.recall = Recall(provider=provider)

    async def ascore(self, gt: str, pred: str) -> float:
        precision_score, recall_score = await asyncio.gather(
            self.precision.ascore(gt, pred), self.recall.ascore(gt, pred)
        )
        return precision_score / recall_score
