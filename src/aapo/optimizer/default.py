from aapo.agents import AgentProtocol, Agent
from aapo.data import DatasetProtocol
from aapo.llms import ProviderProtocol

SYSTEM_PROMPT = """
    
"""

INPUT_FORMAT = """"
    
"""


class PromptOptimizer:
    def __init__(self, trainable_agent: AgentProtocol, provider: ProviderProtocol):
        self.optimizer_agent = Agent(provider, SYSTEM_PROMPT, memory_size=2)
