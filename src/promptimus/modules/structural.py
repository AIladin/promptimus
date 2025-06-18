import json
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from promptimus.core import Parameter
from promptimus.dto import Message, MessageRole
from promptimus.errors import FailedToParseOutput

from .memory import MemoryModule, Module

T = TypeVar("T", bound=BaseModel)

DEFAULT_SYSTEM_PROMPT = """
You are designed to generate structured JSON outputs adhering to a predefined schema without any embellishments or formatting.

Schema Description: {schema_description}

Response Guidelines:
- Ensure all required fields are present and correctly formatted. 
- Enforce any constraints on fields (e.g., length limits, specific formats) strictly.
- Exclude optional fields if they aren't applicable; do not return null for them.
- Provide valid JSON output without additional commentary, formatting markers like ```json, or unnecessary line breaks.
"""

DEFAULT_SYSTEM_PROMPT_NO_SCHEMA = """
You are designed to generate structured JSON outputs adhering to a predefined schema without any embellishments or formatting.

Response Guidelines:
- Ensure all required fields are present and correctly formatted. 
- Enforce any constraints on fields (e.g., length limits, specific formats) strictly.
- Exclude optional fields if they aren't applicable; do not return null for them.
- Provide valid JSON output without additional commentary, formatting markers like ```json, or unnecessary line breaks.
"""

DEFAULT_RETRY_MESSAGE = """
Your response does not conform to the required schema. Please correct your output by ensuring it matches the expected format and constraints. 

**Schema Validation Error:**  
`{error_message}`  
  
Please reattempt the response, ensuring strict adherence to the schema.    
"""


class StructuralOutput(Module, Generic[T]):
    """A module for generating structured outputs based on a predefined schema."""

    def __init__(
        self,
        output_model: type[T],
        n_retries: int = 5,
        system_prompt: str | None = None,
        retry_template: str | None = None,
        retry_message_role: MessageRole = MessageRole.TOOL,
        native: bool = True,
    ):
        super().__init__()

        self.model_json_schema = Parameter(
            json.dumps(output_model.model_json_schema(), indent=4)
        )

        self.native = native

        default_prompt = (
            DEFAULT_SYSTEM_PROMPT if not native else DEFAULT_SYSTEM_PROMPT_NO_SCHEMA
        )

        self.predictor = MemoryModule(
            memory_size=n_retries * 2,
            system_prompt=system_prompt
            if system_prompt is not None
            else default_prompt,
        )

        self.retry_template = Parameter(
            retry_template if retry_template is not None else DEFAULT_RETRY_MESSAGE
        )

        self.retry_message_role = retry_message_role
        self.output_model = output_model

        assert n_retries >= 0
        self.n_retries = n_retries

    async def forward(self, request: list[Message] | Message | str, **kwargs) -> T:
        if not self.native:
            kwargs["schema_description"] = self.model_json_schema.value
            provider_kwargs = {}
        else:
            provider_kwargs = {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.output_model.__name__,
                        "schema": self.output_model.model_json_schema(),
                    },
                }
            }

        with self.predictor.memory:
            for _ in range(self.n_retries):
                response = await self.predictor.forward(
                    request,
                    provider_kwargs=provider_kwargs,
                    **kwargs,
                )

                try:
                    structured_response = self.output_model.model_validate_json(
                        response.content.strip("\n `").removeprefix("json")
                    )
                    return structured_response

                except ValidationError as e:
                    request = Message(
                        role=self.retry_message_role,
                        content=self.retry_template.value.format(error_message=str(e)),
                    )

            raise FailedToParseOutput(
                f"Failed to parse output after {self.n_retries} retries. {self.predictor.memory}"
            )
