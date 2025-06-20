from contextlib import AsyncExitStack
from typing import Self

from aiohttp import ClientSession
from loguru import logger
from rich import print
from tracing import dto


class DashboardClient:
    def __init__(self, project_id: int) -> None:
        self.project_id = project_id
        self.exit_stack = AsyncExitStack()
        self.http_client = ClientSession(
            base_url="http://localhost:8000",
            headers={
                # THIS is localhost dummy token for testing purpuses
                "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJraW5kIjoiVG9rZW4iLCJqdGkiOiJhZjczNzkyNi0yY2M5LTQxM2YtYmNkMS1mNTVlZjI2OGJiNWQiLCJzdWIiOjEsImV4cCI6MTc1ODk4OTIwMCwibmFtZSI6InRlc3QifQ.sa507tgjokEhET5UKKdCaqmexb_NosjZy0KP8AKwsOU"
            },
        )

    async def __aenter__(self) -> Self:
        await self.exit_stack.enter_async_context(self.http_client)
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.exit_stack.aclose()

    async def post(self, record: dto.Span | dto.Trace):
        async with self.http_client.post(
            f"/api/project/{self.project_id}/traces",
            data=record.model_dump_json().encode(),
        ) as r:
            if r.status != 200:
                logger.warning(f"{r.status} {await r.text()}")
                print(record)

            else:
                logger.info("successfully sent")
