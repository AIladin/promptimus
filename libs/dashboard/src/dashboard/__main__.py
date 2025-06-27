import asyncio
import random

from loguru import logger
from tracing import trace

import promptimus as pm
from promptimus.llms.dummy import DummyLLm

from .client import DashboardClient


class TestModule(pm.Module):
    def __init__(self):
        super().__init__()
        self.memory_submodule = pm.modules.MemoryModule(
            5, system_prompt="Im dummy agent"
        )

        self.post_action = pm.Prompt("some post action {kwarg_1} {kwarg_1}")

    async def forward(self, message: str) -> str:
        r = await self.memory_submodule.forward(message)
        r = await self.post_action.forward([r], kwarg_1="param_1", kwarg_2="param_2")
        return r.content


async def call_agent(agent: pm.Module):
    await asyncio.sleep(random.random())
    await agent.forward("DUMMY USER")


async def main(n_concurrent_calls: int = 5):
    client = DashboardClient(2)
    llm = DummyLLm(delay=0.5)
    agent = TestModule().with_llm(llm)

    async with trace([agent], [client]):
        while True:
            logger.info("running agent")
            await asyncio.gather(
                *(call_agent(agent) for _ in range(n_concurrent_calls))
            )


if __name__ == "__main__":
    asyncio.run(main())
