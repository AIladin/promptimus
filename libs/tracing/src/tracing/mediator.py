import asyncio
from asyncio import Queue
from asyncio.exceptions import CancelledError
from contextlib import AsyncExitStack
from typing import Protocol

from loguru import logger

from . import dto
from .tracer import Tracer


class TraceConsumer(Protocol):
    async def post(self, record: dto.Span | dto.Trace) -> None: ...


class Subscriber:
    def __init__(self, consumer: TraceConsumer):
        self.consumer = consumer
        self.buffer = Queue()

        self._task: asyncio.Task | None = None

    async def __aenter__(self):
        assert self._task is None
        self._task = asyncio.create_task(self._task_fn())
        if hasattr(self.consumer, "__aenter__"):
            await self.consumer.__aenter__()  # type: ignore
        return self

    async def __aexit__(self, *args, **kwargs):
        assert self._task is not None
        self._task.cancel()
        self._task = None
        if hasattr(self.consumer, "__aexit__"):
            await self.consumer.__aexit__(*args, **kwargs)  # type: ignore

    async def _task_fn(self):
        while True:
            try:
                record = await self.buffer.get()
                await self.consumer.post(record)
            except CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Subscriber {self.consumer.__class__.__name__} failed to consume record: {e}"
                )


class TraceMediator:
    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer
        self.subscribers: set[Subscriber] = set()
        self.exit_stack = AsyncExitStack()
        self._task: asyncio.Task | None = None

    def subscribe(self, consumer: TraceConsumer):
        self.subscribers.add(Subscriber(consumer))

    async def _broadcast(self):
        while True:
            try:
                record = await self.tracer.queue.get()
                for subscriber in self.subscribers:
                    subscriber.buffer.put_nowait(record)
            except CancelledError:
                break

    async def __aenter__(self):
        assert self._task is None
        for subscriber in self.subscribers:
            await self.exit_stack.enter_async_context(subscriber)
        self._task = asyncio.create_task(self._broadcast())

    async def __aexit__(self, *args, **kwargs):
        assert self._task is not None
        await self.exit_stack.aclose()
        self._task.cancel()
