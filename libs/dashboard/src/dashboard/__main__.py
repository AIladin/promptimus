import asyncio

from loguru import logger
from tracing import Tracer

from promptimus.llms.dummy import DummyLLm
from promptimus.modules import MemoryModule

from .client import DashboardClient

llm = DummyLLm()
tracer = Tracer()
agent = MemoryModule(5, system_prompt="Im dummy agent").with_llm(llm)
tracer.decorate(agent)


async def listener():
    async with DashboardClient() as client:
        while True:
            obj = await tracer.queue.get()
            await client.post_log(obj)


async def producer():
    while True:
        logger.info("running agent")
        await agent.forward("DUMMY USER")
        await asyncio.sleep(3)


async def main():
    await asyncio.gather(listener(), producer())


if __name__ == "__main__":
    asyncio.run(main())
