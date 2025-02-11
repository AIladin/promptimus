from aapo.core import Module, TraceNode, TrainablePrompt
from aapo.llms import ProviderProtocol

SYSTEM_PROMPT = """
You are a recall agent for evaluating semantic similarity in language model outputs.
Calculate the recall as the fraction of the ground truth that is semantically captured by the system's response.

This is defined as:
Recall = (Semantic overlap between system response and ground truth) / (Total semantic content of the ground truth)

Evaluate semantic overlap using meaningful similarity rather than exact matches.
Return a single float between 0 and 1, representing the recall value.
Ensure the response contains only the numeric value and no additional text or explanation.
"""

INPUT_FORMAT = """
The groud truth:
{gt}
"""


class Recall(Module):
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
