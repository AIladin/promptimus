from contextlib import AsyncExitStack
from typing import Self

from aiohttp import ClientSession
from loguru import logger
from rich import print
from tracing import dto


class DashboardClient:
    def __init__(self) -> None:
        self.exit_stack = AsyncExitStack()
        self.http_client = ClientSession(
            base_url="http://localhost:8000",
            headers={
                "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJraW5kIjoiVG9rZW4iLCJqdGkiOiI0ZmJmZWMxNi0yNjg3LTQxYzYtODcwNi0wYWJmNTFhODZjZGMiLCJzdWIiOjEsImV4cCI6MTc1MDM0NDEzNCwibmFtZSI6InRlc3QifQ.DjTxIHaDAIwb34WCLNk61-IU00kyT3i8_UqxY6N-zdg"
            },
        )

    async def __aenter__(self) -> Self:
        await self.exit_stack.enter_async_context(self.http_client)
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.exit_stack.aclose()

    async def post_log(self, record: dto.Span | dto.Trace):
        async with self.http_client.post(
            "/api/traces", data=record.model_dump_json().encode()
        ) as r:
            if r.status != 200:
                logger.warning(f"{r.status} {r.text}")
                print(record)

            else:
                logger.info("successfully sent")
