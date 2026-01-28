import asyncio
import json
import re
from collections.abc import Awaitable
from inspect import Parameter as InspectParameter
from inspect import iscoroutinefunction, signature
from typing import Any, Callable, Generic, Protocol, Self, TypeVar, runtime_checkable

from pydantic import BaseModel, ValidationError, create_model, validate_call

from promptimus import errors
from promptimus.core import Module, Parameter
from promptimus.core.module import ModuleDict
from promptimus.dto import Message, MessageRole
from promptimus.errors import InvalidToolConfig, MaxIterExceeded
from promptimus.modules.memory import Memory, MemoryModule

T = TypeVar("T")


@runtime_checkable
class SupportsHandoff(Protocol):
    """Protocol for modules that accept conversation history during handoff."""

    async def forward(
        self, history: list[Message] | Message | str, **kwargs
    ) -> Message:
        """
        Forward method that accepts conversation history.

        Args:
            history: Conversation History (list of Messages, single Message, or string)
            **kwargs: Additional parameters specific to the module.

        Returns:
            Message response from the sub-agent.
        """
        ...


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
        module: Module | None = None,
        handoff_history: bool = False,
        handoff_include_tool_loop: bool = False,
    ):
        super().__init__()

        # Validate configuration 1: can't include tool loop without enabling history
        if handoff_include_tool_loop and not handoff_history:
            raise InvalidToolConfig(
                f"Tool '{name}' configured with handoff_include_tool_loop=True, but "
                f"handoff_history=False. To include tool loop messages, you must enable "
                f"history passing by setting handoff_history=True."
            )

        # Validate configuration 2: can't enable history if module doesn't support protocol
        if (
            handoff_history
            and module is not None
            and not isinstance(module, SupportsHandoff)
        ):
            raise InvalidToolConfig(
                f"Tool '{name}' configured with handoff_history=True, but its module "
                f"does not implement SupportsHandoff protocol. Either set handoff_history=False "
                f"or implement the protocol in the module's forward() method."
            )

        self.fn = validate_call(fn)
        self.name = name
        self.description = Parameter(description)
        self.module = module
        self.handoff_history = handoff_history
        self.handoff_include_tool_loop = handoff_include_tool_loop

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    @property
    def supports_handoff(self) -> bool:
        """
        Check if this tool supports history handoff.

        Returns:
            True if handoff_history is enabled, False otherwise
        """
        return self.handoff_history

    async def forward(
        self, json_data: str, history: list[Message] | Message | str | None = None
    ) -> T:
        """
        Execute tool with optional conversation history.

        Args:
            json_data: JSON string with tool parameters
            history: Optional conversation history to pass to sub-agents

        Returns:
            Result from the tool function
        """
        input_data = json.loads(json_data)

        # Pass history as first positional argument if handoff is enabled
        if iscoroutinefunction(self.fn):
            if self.handoff_history and history is not None:
                result = await self.fn(history, **input_data)
            else:
                result = await self.fn(**input_data)
        else:
            if self.handoff_history and history is not None:
                result: T = await asyncio.to_thread(self.fn, history, **input_data)  # type: ignore
            else:
                result: T = await asyncio.to_thread(self.fn, **input_data)  # type: ignore
        return result

    def build_param_block(
        fn: Callable[..., T | Awaitable[T]], handoff_history: bool = False
    ) -> str:
        sig = signature(fn)

        params_desc = []
        for pname, pvalue in sig.parameters.items():
            if pname in {"self", "cls"}:
                continue

            if pname == "history" and handoff_history:
                continue

            params_desc.append(
                PARAM_TEMPLATE.format(
                    name=pname,
                    p_type=str(pvalue.annotation),
                ).strip()
            )

        return "\n".join(params_desc)

    # TODO add ags to decorator
    @classmethod
    def decorate(cls, fn: Callable[..., T | Awaitable[T]]) -> Self:
        description = DESCRIPTION_TEMPLATE.format(
            name=fn.__name__,  # ty:ignore[unresolved-attribute]
            description=fn.__doc__ if fn.__doc__ is not None else "",
            param_block=cls.build_param_block(fn, False),
        ).strip()

        return cls(fn, fn.__name__, description)  # ty:ignore[unresolved-attribute]

    @classmethod
    def from_module(
        cls,
        module: Module,
        name: str | None = None,
        description: str | None = None,
        handoff_history: bool = False,
        handoff_include_tool_loop: bool = False,
    ) -> Self:
        name = name or Module.__name__.lower()

        description = DESCRIPTION_TEMPLATE.format(
            name=name,
            description=description or module.__doc__ or "",
            param_block=cls.build_param_block(
                module.forward, handoff_history=handoff_history
            ),
        ).strip()

        return cls(
            module.forward,
            name,
            description,
            module=module,
            handoff_history=handoff_history,
            handoff_include_tool_loop=handoff_include_tool_loop,
        )

    def build_model(self) -> type[BaseModel]:
        sig = signature(self.fn)

        fields = {}
        for pname, pvalue in sig.parameters.items():
            # Skip history parameter when handoff is enabled
            if pname == "history" and self.handoff_history:
                continue

            # Handle annotation: convert empty to Any
            annotation = (
                pvalue.annotation
                if pvalue.annotation != InspectParameter.empty
                else Any
            )

            # Handle default: convert empty to ... (required field)
            default = (
                pvalue.default if pvalue.default != InspectParameter.empty else ...
            )

            fields[pname] = (annotation, default)

        return create_model(self.fn.__name__, **fields)  # ty:ignore[possibly-missing-attribute]

    def to_openai_function(self) -> dict:
        schema = self.build_model().model_json_schema()
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
        tools: list[Tool | Module | Callable],
        max_steps: int = 5,
        memory_size: int = 20,
        prompt: str | None = None,
        observation_role: MessageRole = MessageRole.TOOL,
        shared_memory: Memory | None = None,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.observation_role = observation_role

        self.tools = ModuleDict()

        for obj in tools:
            match obj:
                case Tool() as tool:
                    pass
                case Module():
                    tool = Tool.from_module(obj)
                case _:
                    tool = Tool.decorate(obj)

            self.tools[tool.name] = tool

        self.predictor = MemoryModule(
            system_prompt=prompt if prompt is not None else DEFAULT_PROMPT,
            memory_size=memory_size,
            shared_memory=shared_memory,
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
        self, history: list[Message] | Message | str, **kwargs
    ) -> Message:
        """
        Execute tool-calling agent loop with conversation history.

        Args:
            history: Conversation History (list of Messages, single Message, or string)
            **kwargs: Additional parameters for predictor

        Returns:
            Final response message
        """
        for step in range(self.max_steps):
            response = await self.predictor.forward(
                history,
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
                    history = Message(
                        role=self.observation_role,
                        content=self.INVALID_TOOL_NAME_MESSAGE.format(
                            name=tool_name, tool_names=self.tool_names
                        ),
                    )
                    continue

                # check if parameters is valid & execute tool
                try:
                    # Prepare history for handoff if tool supports it
                    tool_history = None
                    if tool.supports_handoff:
                        # Get current conversation history from internal memory
                        current_history = self.predictor.memory.as_list()

                        # Filter history based on handoff_include_tool_loop
                        if tool.handoff_include_tool_loop:
                            tool_history = current_history
                        else:
                            # Filter out tool loop messages (observations and assistant tool calls)
                            tool_history = [
                                msg
                                for msg in current_history
                                if msg.role != self.observation_role
                                and not (
                                    msg.role == MessageRole.ASSISTANT and msg.tool_calls
                                )
                            ]

                    tool_response = await tool.forward(
                        match.group("action_input").strip("`'\" \n"),
                        history=tool_history,
                    )
                except (ValidationError, json.JSONDecodeError) as e:
                    # invalid params case
                    history = [
                        Message(
                            role=self.observation_role,
                            content=str(e),
                        )
                    ]
                    continue

                # valid tool output
                history = Message(
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
                history = Message(
                    role=self.observation_role,
                    content=self.INVALID_FORMAT_MESSAGE,
                )

        raise MaxIterExceeded()


M = TypeVar("M", bound=BaseModel)


class OpenaiToolCallingAgent(ToolCallingAgent, Generic[M]):
    def __init__(
        self,
        prompt: str,
        tools: list[Tool | Module | Callable],
        max_steps: int = 5,
        memory_size: int = 50,
        structural_output: type[M] | None = None,
        shared_memory: Memory | None = None,
    ):
        super().__init__(
            tools,
            max_steps,
            memory_size,
            prompt,
            MessageRole.TOOL,
            shared_memory,
        )
        self.structural_output_model = structural_output

    async def forward(
        self, history: list[Message] | Message | str, **kwargs
    ) -> Message | M:  # ty:ignore[invalid-method-override]
        """
        Execute OpenAI tool-calling agent loop with conversation history.

        Args:
            history: Conversation History (list of Messages, single Message, or string)
            **kwargs: Additional parameters for predictor

        Returns:
            Final response message
        """

        provider_kwargs = {
            "tools": [
                tool.to_openai_function() for tool in self.tools.objects_map.values()
            ]
        }

        if self.structural_output_model is not None:
            provider_kwargs.update(
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": self.structural_output_model.__name__,
                            "schema": self.structural_output_model.model_json_schema(),
                        },
                    }
                }
            )

        for step in range(self.max_steps):
            response = await self.predictor.forward(
                history,
                provider_kwargs=provider_kwargs,
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

                    # Prepare history for handoff if tool supports it
                    tool_history = None
                    if tool.supports_handoff:
                        # Get current conversation history from internal memory
                        current_history = self.predictor.memory.as_list()

                        # Filter history based on handoff_include_tool_loop
                        if tool.handoff_include_tool_loop:
                            tool_history = current_history
                        else:
                            # Filter out tool loop messages (observations and assistant tool calls)
                            tool_history = [
                                msg
                                for msg in current_history
                                if msg.role != self.observation_role
                                and not (
                                    msg.role == MessageRole.ASSISTANT and msg.tool_calls
                                )
                            ]

                    tool_future = tool.forward(
                        tool_request.function.arguments, history=tool_history
                    )
                    futures.append(tool_future)

                # valid tool output
                tool_results = await asyncio.gather(*futures, return_exceptions=True)

                history = [
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
                if self.structural_output_model is not None:
                    try:
                        structured_response = (
                            self.structural_output_model.model_validate_json(
                                response.content.strip("\n `").removeprefix("json")
                            )
                        )
                        return structured_response

                    except ValidationError as e:
                        history = Message(
                            role=MessageRole.USER,
                            content=str(e),
                        )

                # valid user response
                return Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                )
        raise MaxIterExceeded()
