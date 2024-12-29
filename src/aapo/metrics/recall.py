from aapo.agents import Agent
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

--------------------------------

The system prediction:
{pred}

--------------------------------
"""


class Recall:
    def __init__(self, provider: ProviderProtocol) -> None:
        self.agent = Agent(
            provider=provider,
            system_prompt=SYSTEM_PROMPT,
        )

    async def ascore(self, gt: str, pred: str) -> float:
        score = await self.agent.achat(INPUT_FORMAT.format(gt=gt, pred=pred))

        return float(score)
