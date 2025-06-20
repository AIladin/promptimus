from collections.abc import Iterable
from contextlib import asynccontextmanager

from promptimus import Module

from . import dto as dto
from .mediator import TraceConsumer, TraceMediator
from .tracer import Tracer as Tracer


@asynccontextmanager
async def trace(modules: Iterable[Module], clients: Iterable[TraceConsumer]):
    tracer = Tracer()
    for module in modules:
        tracer.decorate(module)

    mediator = TraceMediator(tracer)

    for client in clients:
        mediator.subscribe(client)

    async with mediator:
        yield
