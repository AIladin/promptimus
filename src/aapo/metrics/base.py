from typing import Protocol


class Metric(Protocol):
    async def ascore(self, gt: str, pred: str) -> float: ...
