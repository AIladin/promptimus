from aapo.core import Module, TraceNode, TrainablePrompt
from aapo.llms import ProviderProtocol

SYSTEM_PROMPT = """
You are a precision agent for evaluating semantic similarity in language model outputs.
Calculate the precision as the fraction of the system's response that is semantically aligned with the ground truth.

This is defined as:
Precision = (Semantic overlap between system response and ground truth) / (Total semantic content of the system response)

Evaluate semantic overlap using meaningful similarity rather than exact matches. Return a single float between 0 and 1, representing the precision value. Ensure the response contains only the numeric value and no additional text or explanation.
"""

INPUT_FORMAT = """
The groud truth:
{gt}
"""


class Precision(Module):
    def __init__(self, provider: ProviderProtocol) -> None:
        super().__init__()
        self.prompt = TrainablePrompt(SYSTEM_PROMPT, provider=provider, trainable=False)

    async def forward(self, gt: str, pred: TraceNode) -> tuple[TraceNode, float]:
        node = await self.prompt.forward(
            TraceNode.compose(
                INPUT_FORMAT.format(gt=gt),
                [
                    ("The predicted output:", pred),
                ],
            )
        )

        assert node._response is not None
        return pred, float(node._response.content)
