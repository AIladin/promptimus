from aapo.core import TraceNode, TrainablePrompt
from aapo.llms import ProviderProtocol
from aapo.utils import DEFAULT_SEPARATOR, prettify_history

SYSTEM_PROMPT = """
You are a prompt optimization expert tasked with improving the quality of a given trainable prompt for an AI system. 

Your objective is to analyze a set of examples where the AI system has used the prompt to generate outputs. Each example includes:
- The input messages (history) provided to the system.
- The AI's generated output based on the current prompt.
- A numerical score evaluating the quality of the output (higher scores indicate better performance).

Using this information, your task is to:
1. Identify patterns or weaknesses in the current prompt that may have contributed to low-scoring outputs.
2. Propose specific and actionable changes to the prompt to improve the AI's performance on similar examples in the future.
3. Ensure that changes to the prompt do not negatively affect high-scoring examples.

When making changes:
- Focus on making the prompt clearer and more aligned with the intended goals.
- Avoid overfitting the prompt to the provided examples. Maintain generality while addressing the identified weaknesses.
- Use concise and precise language in the prompt.

Respond only with the updated prompt. Avoid additional text or explanations.
"""

EXAMPLE_FORMAT = """
INPUT:

{}

PREDICTED OUTPUT:

{}

SCORE: {}
"""

INPUT_FORMAT = """"

# The original prompt is:
{}

# Evaluation results:
{}
    
"""


class DefaultJudge:
    def __init__(self, provider: ProviderProtocol) -> None:
        self.prompt = TrainablePrompt(SYSTEM_PROMPT, provider=provider, trainable=False)

    async def update_prompt(
        self,
        trainable_prompt: TrainablePrompt,
        batch: list[TraceNode],
    ) -> TrainablePrompt:
        examples = []
        for node in batch:
            assert node.executor == trainable_prompt
            examples.append(
                EXAMPLE_FORMAT.format(
                    prettify_history(node.history, sep="\n"),
                    node.response,
                    node.avg_score(),
                )
            )
        examples_text = DEFAULT_SEPARATOR.join(examples)

        node = await self.prompt.forward(
            TraceNode.compose(
                INPUT_FORMAT.format(trainable_prompt.prompt, examples_text)
            )
        )

        return TrainablePrompt(
            node.response.content, trainable_prompt.provider, trainable_prompt.trainable
        )
