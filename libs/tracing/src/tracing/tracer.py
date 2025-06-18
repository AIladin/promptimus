from asyncio import Queue
from contextvars import ContextVar
from datetime import datetime, timezone
from itertools import chain

from promptimus import Module, Prompt

from . import dto

_parent: ContextVar[dto.Span | None] = ContextVar("parent", default=None)


class Tracer:
    def __init__(self) -> None:
        # TODO limit maxsize
        self.queue: Queue[dto.Span | dto.Trace] = Queue()

    def _decorate_prompt(self, name: str, prompt: Prompt):
        fn = prompt.forward

        async def wrapper(
            history: list[dto.Message] | None = None,
            provider_kwargs: dict | None = None,
            **kwargs,
        ):
            parent = _parent.get(None)
            trace = dto.Trace(
                prompt_name=name,
                prompt_digest=prompt.digest(),
                parent_id=parent.span_id if parent else None,
                prompt=prompt.prompt.value,
                role=prompt.role.value,
                prompt_args=kwargs,
                history=history,
                llm=prompt.llm.model_name,
            )

            self.queue.put_nowait(trace.model_copy())

            try:
                response = await fn(history, provider_kwargs, **kwargs)

                trace.stop = datetime.now(timezone.utc)
                trace.response = response
                trace.status = dto.LogStatus.OK

                self.queue.put_nowait(trace)

                return response
            except Exception as e:
                trace.stop = datetime.now(timezone.utc)
                trace.error = str(e)
                trace.status = dto.LogStatus.ERR

                self.queue.put_nowait(trace)

                raise e

        prompt.forward = wrapper

    def _decorate_module(self, name: str, module: Module):
        fn = module.forward

        async def wrapper(
            *args,
            **kwargs,
        ):
            parent = _parent.get(None)
            span = dto.Span(
                module_name=name,
                module_digest=module.digest(),
                parent_id=parent.span_id if parent else None,
                request="\n".join(
                    chain(
                        (str(arg) for arg in args),
                        (f"{k}: {v}" for k, v in kwargs.items()),
                    )
                ),
            )

            token = _parent.set(span)

            self.queue.put_nowait(span.model_copy())

            try:
                response = await fn(*args, **kwargs)

                if isinstance(response, dto.Message):
                    span.response = response.content
                else:
                    span.response = str(response)

                span.status = dto.LogStatus.OK

                return response
            except Exception as e:
                span.error = str(e)
                span.status = dto.LogStatus.ERR
                raise e

            finally:
                _parent.reset(token)
                span.stop = datetime.now(timezone.utc)
                self.queue.put_nowait(span)

        module.forward = wrapper

    def decorate(self, module: Module, name: str = "root"):
        self._decorate_module(name, module)

        for name, sub in module._submodules.items():
            if isinstance(sub, Prompt):
                self._decorate_prompt(name, sub)
            elif isinstance(sub, Module):
                self.decorate(sub, name)
