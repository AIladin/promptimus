import asyncio
import json
import re
from collections.abc import Awaitable
from inspect import iscoroutinefunction, signature
from typing import Callable, Generic, Self, TypeVar

from pydantic import BaseModel, ValidationError, create_model, validate_call

from promptimus import errors
from promptimus.core import Module, Parameter
from promptimus.core.module import ModuleDict
from promptimus.dto import Message, MessageRole
from promptimus.errors import MaxIterExceeded
from promptimus.modules.memory import MemoryModule

T = TypeVar("T")

DESCRIPTION_TEMPLATE = """
## `{name}` tool.
{description}

Parameters:
{param_block}
"""

PARAM_TEMPLATE = """
- `{name}`: {p_type} 
"""


class Tool(Module, Generic[T]):
    def __init__(
        self,
        fn: Callable[..., T | Awaitable[T]],
        name: str,
        description: str | None = None,
    ):
        super().__init__()

        self.fn = validate_call(fn)
        self.name = name
        self.description = Parameter(description)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    async def forward(self, json_data: str) -> T:
        input_data = json.loads(json_data)
        if iscoroutinefunction(self.fn):
            result = await self.fn(**input_data)
        else:
            result: T = await asyncio.to_thread(self.fn, **input_data)  # type: ignore
        return result

    @classmethod
    def decorate(cls, fn: Callable[..., T | Awaitable[T]]) -> Self:
        sig = signature(fn)

        params_desc = []
        for pname, pvalue in sig.parameters.items():
            params_desc.append(
                PARAM_TEMPLATE.format(
                    name=pname,
                    p_type=pvalue.annotation.__name__,
                ).strip()
            )

        description = DESCRIPTION_TEMPLATE.format(
            name=fn.__name__,
            description=fn.__doc__ if fn.__doc__ is not None else "",
            param_block="\n".join(params_desc),
        ).strip()

        return cls(fn, fn.__name__, description)

    def build_model(self) -> type[BaseModel]:
        sig = signature(self.fn)

        fields = {}
        for pname, pvalue in sig.parameters.items():
            fields[pname] = (pvalue.annotation, pvalue.default)

        return create_model(self.fn.__name__, **fields)

    def to_openai_function(self) -> dict:
        schema = self.build_model().schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description.value,
                "parameters": schema,
            },
        }


# credit: https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/#view-prompts
DEFAULT_PROMPT = """
You are designed to assist with a wide range of tasks—from answering questions and providing summaries to performing detailed analyses—by utilizing a variety of external tools. Follow these strict instructions to ensure correct tool usage and response formatting:

---

## Tools

- **Tool Access:**  
  You have access to multiple tools: 

  {tool_desc}

- **Execution Protocol:**  
  - **One Step at a Time:** In each response, you must either make a single tool call or provide a direct answer to the user.
  - **No Fabrication:** You are strictly forbidden from generating any `Observation:` lines or simulating tool outputs. Do not assume, infer, or create tool responses—always wait for the actual output from the external tool.

---

## Output Format

When processing a user's query, follow this structured format:

1. **Initial Response (Tool Call or Direct Answer):**  
   - **Thought:** Start every response with a clear thought outlining your reasoning.  
   - **Tool Call:**  
     If you decide that a tool is needed, include:
     - **Action:** The tool's name (choose one from {tool_names}).
     - **Action Input:** The tool input in valid JSON format representing the keyword arguments (for example: `{{"input": "hello world", "num_beams": 5}}`).

   **Example:**
   ```
   Thought: I need to use a tool to help me answer the question.
   Action: tool_name
   Action Input: {{"input": "hello world", "num_beams": 5}}
   ```

2. **Waiting for the Tool Response:**  
   - **Do Not Simulate:** Never generate an `Observation:` line or assume the tool's output. After making a tool call, wait for the external tool to return the actual observation.

3. **Final Answer (When Sufficient Information is Obtained):**  
   When you have enough information to answer the question without additional tool calls, respond using one of these formats:

   **Successful Answer:**
   ```
   Thought: I can answer without using any more tools.
   Answer: [your answer here]
   ```

   **Unable to Answer:**
   ```
   Thought: I cannot answer the question with the provided tools.
   Answer: Sorry, I cannot answer your query.
   ```

---

### Important Reminders

- **Always Start with a Thought:** Begin every response with a "Thought:" to describe your reasoning process.
- **Valid JSON Required:** Ensure that the Action Input is formatted in valid JSON (e.g., do not use single quotes or improper brackets).
- **Strict No-Observation Rule:** Under no circumstances should you generate or simulate an `Observation:` line. Only the external tool should produce that.

By following these guidelines, you will ensure clear, consistent, and tool-dependent interactions.
"""


class ToolCallingAgent(Module):
    ANSWER_PATT = re.compile(
        r"^.*?Thought:\s*(?P<thought>.+?)\s*Answer:\s*(?P<answer>.+)(?=.*?Thought:|.*).*",
        re.DOTALL,
    )
    TOOL_CALL_PATT = re.compile(
        r"^.*?Thought:\s*(?P<thought>.+?)\s*Action:\s*(?P<action>\S+)\s*Action Input:\s*(?P<action_input>\{.+?\})(?=.*Thought:|.*).*",
        re.DOTALL,
    )

    INVALID_TOOL_NAME_MESSAGE = (
        "Tool `{name}` name not found. Please provide one name from {tool_names}."
    )
    INVALID_FORMAT_MESSAGE = (
        "Cannot parse output. Please answer in Thought-Action/Answer format."
    )
    TOOL_OUTPUT_MESSAGE = "Observation: {output}"

    def __init__(
        self,
        tools: list[Tool | Callable],
        max_steps: int = 5,
        memory_size: int = 20,
        prompt: str | None = None,
        observation_role: MessageRole = MessageRole.TOOL,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.observation_role = observation_role

        converted_tools: list[Tool] = [
            tool if isinstance(tool, Tool) else Tool.decorate(tool) for tool in tools
        ]

        self.tools = ModuleDict(**{tool.name: tool for tool in converted_tools})
        self.predictor = MemoryModule(
            memory_size, prompt if prompt is not None else DEFAULT_PROMPT
        )

    @property
    def tool_desc(self):
        return "\n".join(
            [tool.description.value for tool in self.tools.objects_map.values()]
        )

    @property
    def tool_names(self):
        return str([tool.name for tool in self.tools.objects_map.values()])

    async def forward(
        self, request: list[Message] | Message | str, **kwargs
    ) -> Message:
        for step in range(self.max_steps):
            response = await self.predictor.forward(
                request,
                tool_desc=self.tool_desc,
                tool_names=self.tool_names,
                **kwargs,
            )

            # TOOL path
            if (match := self.TOOL_CALL_PATT.match(response.content)) is not None:
                tool_name = match.group("action").strip("`'\" \n")

                # cut extra output from memory
                self.predictor.replace_last(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=(
                            f"Thought:{match.group('thought')}"
                            f"\nAction:{tool_name}"
                            f"\nAction Input: {match.group('action_input')}"
                        ),
                    )
                )

                # check if tool name is valid
                tool = self.tools.objects_map.get(tool_name)
                if tool is None:
                    request = Message(
                        role=self.observation_role,
                        content=self.INVALID_TOOL_NAME_MESSAGE.format(
                            name=tool_name, tool_names=self.tool_names
                        ),
                    )
                    continue

                # check if parameters is valid & execute tool
                try:
                    tool_response = await tool.forward(
                        match.group("action_input").strip("`'\" \n")
                    )
                except (ValidationError, json.JSONDecodeError) as e:
                    # invalid params case
                    request = [
                        Message(
                            role=self.observation_role,
                            content=str(e),
                        )
                    ]
                    continue

                # valid tool output
                request = Message(
                    role=self.observation_role,
                    content=self.TOOL_OUTPUT_MESSAGE.format(output=tool_response),
                )
                continue

            # USER response path
            elif (match := self.ANSWER_PATT.match(response.content)) is not None:
                # cut extra output from memory
                self.predictor.replace_last(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=(
                            f"Thought:{match.group('thought')}"
                            f"\nAnswer:{match.group('answer')}"
                        ),
                    )
                )
                # valid user response
                return Message(
                    role=MessageRole.ASSISTANT, content=match.group("answer").strip()
                )
            else:
                # invalid user response
                request = Message(
                    role=self.observation_role,
                    content=self.INVALID_FORMAT_MESSAGE,
                )

        raise MaxIterExceeded()


class OpenaiToolCallingAgent(ToolCallingAgent):
    def __init__(
        self,
        prompt: str,
        tools: list[Tool | Callable],
        max_steps: int = 5,
        memory_size: int = 50,
    ):
        super().__init__(tools, max_steps, memory_size, prompt, MessageRole.TOOL)

    async def forward(
        self, request: list[Message] | Message | str, **kwargs
    ) -> Message:
        for step in range(self.max_steps):
            response = await self.predictor.forward(
                request,
                provider_kwargs={
                    "tools": [
                        tool.to_openai_function()
                        for tool in self.tools.objects_map.values()
                    ]
                },
                **kwargs,
            )

            # TOOL path
            if tool_requests := response.tool_calls:
                futures = []

                for tool_request in tool_requests:
                    # check if tool name is valid
                    if (
                        tool := self.tools.objects_map.get(tool_request.function.name)
                    ) is None:
                        raise errors.InvalidToolName(
                            f"{tool_request.function.name}, expected one of {self.tool_names}"
                        )

                    tool_future = tool.forward(tool_request.function.arguments)
                    futures.append(tool_future)

                # valid tool output
                tool_results = await asyncio.gather(*futures, return_exceptions=True)

                request = [
                    Message(
                        role=self.observation_role,
                        content=str(tool_result),
                        tool_call_id=tool_request.id,
                    )
                    for tool_request, tool_result in zip(tool_requests, tool_results)
                ]

                continue

            # USER response path
            else:
                # valid user response
                return Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                )
        raise MaxIterExceeded()
