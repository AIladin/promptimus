from typing import AsyncIterable
from aapo.core import TraceNode, TrainablePrompt
from aapo.llms import ProviderProtocol
from aapo.utils import prettify_history

SYSTEM_PROMPT = """
You are an expert evaluator in hierarchical decision systems.

Your task is to assess the contribution of a single parent node's output to a child's prediction quality.

Evaluate the degree to which the parent node's output influenced the child's prediction quality and score. Assign a numerical score (between 0 and 1) to the parent node, where:
- 1 indicates a perfect positive contribution.
- 0.5 indicated no impact.
- 0 indicates a negative impact.

Respond with a single numerical score. Avoid addition text or explanation.
"""

INPUT_FORMAT = """
- Child's history
{}

-----------------------------------------------------

- Child's prediction:
{}

-----------------------------------------------------

- Child's evaluation score:
{}

-----------------------------------------------------

- Parent output:
{}
"""


class DefaultPropagator:
    def __init__(self, provider: ProviderProtocol) -> None:
        self.prompt = TrainablePrompt(SYSTEM_PROMPT, provider=provider, trainable=False)

    async def propagate(
        self,
        node: TraceNode,
    ) -> AsyncIterable[TraceNode]:
        """Updates scores for parents of current trace Node inplace"""

        for parent_node in node.parents:
            response_node = await self.prompt.forward(
                TraceNode.compose(
                    INPUT_FORMAT.format(
                        prettify_history(node.history, sep="\n"),
                        node.response.content,
                        node.avg_score(),
                        parent_node.response.content,
                    )
                )
            )

            parent_node.scores.append(float(response_node.response.content))
            yield parent_node
