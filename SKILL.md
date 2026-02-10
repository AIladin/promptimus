---
name: promptimus
description: promptimus python library for building AI agents
---
# Promptimus Skill Reference

This is the sole reference for AI coding agents building agents with Promptimus.

---

## 1. Import Map

```python
import promptimus as pm
```

### Top-level exports (`pm.*`)

| Symbol          | Type     |
|-----------------|----------|
| `pm.Module`     | ABC base class for all agents |
| `pm.Parameter`  | Serializable config container |
| `pm.Prompt`     | Single-shot LLM call with template |
| `pm.Message`    | Pydantic model for chat messages |
| `pm.MessageRole`| StrEnum: `USER`, `SYSTEM`, `ASSISTANT`, `TOOL` |
| `pm.ImageContent`| Pydantic model for image payloads |

### Modules (`pm.modules.*`)

| Symbol | Description |
|--------|-------------|
| `pm.modules.MemoryModule` | Multi-turn conversation with rolling history |
| `pm.modules.StructuralOutput` | Pydantic model extraction from LLM |
| `pm.modules.Tool` | Function/module wrapper for agent tools |
| `pm.modules.ToolCallingAgent` | ReACT text-based tool loop |
| `pm.modules.OpenaiToolCallingAgent` | Native OpenAI function calling loop |
| `pm.modules.RetrievalModule` | Embed query + search vector store |
| `pm.modules.RAGModule` | Retrieval + memory composition |
| `pm.modules.ResetMemoryContext` | Context manager to clear all memories |
| `pm.modules.SupportsHandoff` | Protocol for handoff-capable modules |

### LLMs (`pm.llms.*`)

| Symbol | Description |
|--------|-------------|
| `pm.llms.OpenAILike` | OpenAI-compatible LLM client |
| `pm.llms.OllamaProvider` | Convenience wrapper for local Ollama |

### Embedders (`pm.embedders.*`)

| Symbol | Description |
|--------|-------------|
| `pm.embedders.OpenAILikeEmbedder` | OpenAI-compatible embedding client |

### Not re-exported (direct import required)

```python
from promptimus.llms.dummy import DummyLLm        # Testing stub
from promptimus.modules.memory import Memory       # Raw shared state buffer
```

---

## 2. Core Rules

1. **Every agent is a `pm.Module` with `async def forward()`.** Always call `super().__init__()` in your `__init__`.
2. **Assign submodules and Parameters as instance attributes in `__init__`.** They are auto-registered via `__setattr__` for hierarchy tracking, serialization, and digest computation.
3. **Configure providers once on the root module.** Call `.with_llm()`, `.with_embedder()`, `.with_vector_store()` on the root -- they propagate recursively to all submodules.

---

## 3. Data Types

### `pm.Message`

Pydantic model. Fields:
- `role: MessageRole | str` -- the message role
- `content: str` -- text content
- `images: list[ImageContent]` -- image attachments (default `[]`)
- `tool_calls: list[ToolRequest] | None` -- OpenAI-style tool calls (default `None`)
- `tool_call_id: str | None` -- for TOOL role responses
- `reasoning: str | None` -- reasoning content from supported models

### `pm.MessageRole`

StrEnum with values: `USER = "user"`, `SYSTEM = "system"`, `ASSISTANT = "assistant"`, `TOOL = "tool"`.

### `pm.ImageContent`

- `ImageContent(url="https://...")` -- direct URL
- `ImageContent.from_buffer(buffer: BytesIO, mimetype: str)` -- base64-encodes a buffer

### Forward convention

Most modules accept `list[Message] | Message | str` and return `Message`. Exceptions noted per block.

---

## 4. Provider Setup

### OpenAILike

```python
llm = pm.llms.OpenAILike(
    model_name="gpt-4.1-nano",
    # All extra kwargs pass to AsyncOpenAI():
    # api_key="...", base_url="...", etc.
)
```

API key is auto loaded from env for OpenAI, no need to load it explicitly.

Constructor: `OpenAILike(model_name: str, call_kwargs: dict | None = None, max_concurrency: int = 10, n_retries: int = 5, base_wait: float = 3.0, **client_kwargs)`

`client_kwargs` are forwarded to `AsyncOpenAI(...)`.

### OllamaProvider

```python
llm = pm.llms.OllamaProvider(
    model_name="gemma3:4b",
    base_url="http://localhost:11434/v1",
)
```

Constructor: `OllamaProvider(model_name: str, base_url: str)`. Wraps `OpenAILike` internally with `api_key="DUMMY"`.

### OpenAILikeEmbedder

```python
embedder = pm.embedders.OpenAILikeEmbedder(
    model_name="text-embedding-3-small",
    # api_key="...", base_url="...", etc.
)
```

Constructor: `OpenAILikeEmbedder(model_name: str, embed_kwargs: dict | None = None, max_concurrency: int = 10, n_retries: int = 5, base_wait: float = 3.0, **client_kwargs)`

Methods: `await embedder.aembed(text) -> list[float]`, `await embedder.aembed_batch(texts) -> list[list[float]]`

### DummyLLm (testing)

```python
from promptimus.llms.dummy import DummyLLm

dummy = DummyLLm(message="fixed response", delay=0)
```

Constructor: `DummyLLm(message: str = "DUMMY ASSITANT", delay: float = 3)`. Returns a fixed `Message` from `achat()`.

### Rate limiting

Built into `OpenAILike` and `OpenAILikeEmbedder`. Exponential backoff: `base_wait` seconds (default 3.0), doubling each retry, up to `n_retries` (default 5). Concurrency capped by `max_concurrency` (default 10).

---

## 5. Building Blocks

### 5.1 Prompt

Single-shot LLM call with `str.format_map` template variables.

**Constructor:**
```python
pm.Prompt(prompt: str | None, role: MessageRole | str = MessageRole.SYSTEM)
```

**Forward:**
```python
async def forward(self, history: list[Message] | None = None, provider_kwargs: dict | None = None, **kwargs) -> Message
```

`**kwargs` become template variables substituted into the prompt text via `str.format_map`.

**Example:**
```python
prompt = pm.Prompt("You are {name}, a helpful assistant.").with_llm(llm)
response = await prompt.forward(
    [pm.Message(role=pm.MessageRole.USER, content="Who are you?")],
    name="Henry",
)
```

**Key behaviors:**
- The prompt is prepended as a system message before `history`
- Prompt text and role are stored as `Parameter` instances (serializable)

### 5.2 Parameter

Serializable config container.

**Constructor:**
```python
pm.Parameter(value: T | None = None)
```

**Access:** `.value` property (raises `ParamNotSet` if `None`).

**Key behaviors:**
- Auto-registered when assigned as a Module attribute
- MD5 digest tracking via `.digest`
- Survives `save()` / `load()` round-trips as TOML

### 5.3 MemoryModule

Multi-turn conversation with rolling history.

**Constructor:**
```python
pm.modules.MemoryModule(
    memory_size: int = 10,
    shared_memory: Memory | None = None,
    system_prompt: str | None = None,
    new_message_role: MessageRole | str = MessageRole.USER,
)
```

**Forward:**
```python
async def forward(self, history: list[Message] | Message | str, **kwargs) -> Message
```

**Example:**
```python
assistant = pm.modules.MemoryModule(
    memory_size=3, system_prompt="You are an assistant."
).with_llm(llm)

await assistant.forward("Hi my name is Ailadin!")
await assistant.forward("What is my name?")

# Inspect memory contents
assistant.memory.as_list()
```

**Key behaviors:**
- Deque-based: oldest messages are evicted when `memory_size` is exceeded
- Strings are auto-wrapped as `Message(role=new_message_role, content=...)`
- Orphaned TOOL messages at the start of memory are stripped automatically
- `memory.reset()` to clear manually
- `memory.as_list()` returns current conversation as `list[Message]`

### 5.4 Memory (raw)

Shared state deque buffer. NOT a Module -- has no `forward()`.

```python
from promptimus.modules.memory import Memory

shared = Memory(size=50)
```

**Constructor:** `Memory(size: int)`

**Usage:** Pass as `shared_memory` parameter to `MemoryModule`, `ToolCallingAgent`, or `OpenaiToolCallingAgent` to share conversation state across multiple agents.

**Context manager:** Clears on enter and exit:
```python
with shared:
    # memory is empty here
    ...
# memory is cleared again here
```

### 5.5 StructuralOutput

Pydantic model extraction from LLM responses.

**Constructor:**
```python
pm.modules.StructuralOutput(
    output_model: type[T],       # Pydantic BaseModel subclass
    n_retries: int = 5,
    system_prompt: str | None = None,
    retry_template: str | None = None,
    retry_message_role: MessageRole = MessageRole.TOOL,
    native: bool = True,
)
```

**Forward:**
```python
async def forward(self, history: list[Message] | Message | str, **kwargs) -> T
```

Returns a **Pydantic model instance**, not a `Message`.

**Example:**
```python
from enum import StrEnum, auto
from pydantic import BaseModel, Field

class Operation(StrEnum):
    SUM = auto()
    SUB = auto()
    DIV = auto()
    MUL = auto()

class CalculatorSchema(BaseModel):
    reasoning: str
    a: float = Field(description="The left operand.")
    b: float = Field(description="The right operand.")
    op: Operation = Field(description="The operation to execute.")

module = pm.modules.StructuralOutput(
    CalculatorSchema,
    retry_message_role=pm.MessageRole.USER,  # for providers rejecting consecutive TOOL messages
).with_llm(llm)

result = await module.forward("I have 10 cows, I need twice the amount")
# result is a CalculatorSchema instance
```

**Key behaviors:**
- `native=True` (default): uses OpenAI `response_format` with `json_schema`. Best for OpenAI-compatible providers.
- `native=False`: injects JSON schema into the system prompt. Use for providers without native structured output.
- `retry_message_role=MessageRole.USER` recommended for providers that reject consecutive TOOL messages (e.g., some Ollama models).
- Each `forward()` call gets a clean retry context (`with self.predictor.memory:` clears memory).
- Raises `FailedToParseOutput` after exhausting retries.

### 5.6 Tool

Function/module wrapper for agent tools.

**Decorator (for plain functions):**
```python
@pm.modules.Tool.decorate
def power(a: float, b: float) -> float:
    """Calculates `a` to the power of `b`"""
    return a ** b
```

**Module wrap:**
```python
tool = pm.modules.Tool.from_module(
    module,
    name=None,            # defaults to class name lowercased
    description=None,
    handoff_history=False,
    handoff_include_tool_loop=False,
)
```

**Key behaviors:**
- `.describe()` shows the auto-generated TOML with tool description
- Direct call: `power(2, 8)` -- calls the original function
- Async forward: `await power.forward('{"a": 2, "b": 8}')` -- parses JSON, validates with Pydantic
- Invalid inputs raise `ValidationError` with clear messages
- `handoff_history=True` requires the module to implement `SupportsHandoff` protocol
- `handoff_include_tool_loop=True` requires `handoff_history=True`

### 5.7 ToolCallingAgent

ReACT text-based tool loop. Parses Thought/Action/Action Input/Answer from LLM output.

**Constructor:**
```python
pm.modules.ToolCallingAgent(
    tools: list[Tool | Module | Callable],
    max_steps: int = 5,
    memory_size: int = 20,
    prompt: str | None = None,
    observation_role: MessageRole = MessageRole.TOOL,
    shared_memory: Memory | None = None,
)
```

**Forward:**
```python
async def forward(self, history: list[Message] | Message | str, **kwargs) -> Message
```

**Example:**
```python
import math

@pm.modules.Tool.decorate
def power(a: float, b: float) -> float:
    """Calculates `a` to the power of `b`"""
    return a ** b

def factorial(a: int) -> int:
    """Calculates the factorial of `a`"""
    return math.factorial(a)

agent = pm.modules.ToolCallingAgent(
    [power, factorial],
    observation_role=pm.MessageRole.USER,  # for providers without TOOL role
).with_llm(llm)

result = await agent.forward("What is the factorial of (2 to the power of 3)?")
# Agent chains: power(2,3) -> 8 -> factorial(8) -> 40320
```

**Key behaviors:**
- Auto-wraps plain functions into `Tool` instances (via `Tool.decorate`) and `Module` instances (via `Tool.from_module`)
- `observation_role=MessageRole.USER` for providers that don't support the TOOL role
- Raises `MaxIterExceeded` after `max_steps` iterations
- Memory accessible at `agent.predictor.memory`
- `**kwargs` in `forward()` become format variables in the system prompt

### 5.8 OpenaiToolCallingAgent

Native OpenAI function calling. Concurrent tool execution.

**Constructor:**
```python
pm.modules.OpenaiToolCallingAgent(
    prompt: str,
    tools: list[Tool | Module | Callable],
    max_steps: int = 5,
    memory_size: int = 50,
    structural_output: type[BaseModel] | None = None,
    shared_memory: Memory | None = None,
)
```

**Forward:**
```python
async def forward(self, history: list[Message] | Message | str, **kwargs) -> Message | T
```

Returns a Pydantic model instance if `structural_output` is set, otherwise `Message`.

**Example:**
```python
agent = pm.modules.OpenaiToolCallingAgent(
    "Utilize your tools step by step, to act as a calculator.",
    [power, factorial, multiply],
).with_llm(llm)

result = await agent.forward("What is twice the factorial of 3?")
# Calls factorial(3) -> 6, then multiply(2, 6) -> 12
```

**Key behaviors:**
- Multiple tool calls execute concurrently via `asyncio.gather()`
- Optional `structural_output` for typed final response (uses OpenAI `response_format`)
- Uses `to_openai_function()` to generate tool schemas
- Raises `MaxIterExceeded` after `max_steps`
- Subclass of `ToolCallingAgent` -- shares tool registration logic

### 5.9 RetrievalModule

Embed query + search vector store.

**Constructor:**
```python
pm.modules.RetrievalModule(n_results: int = 10)
```

**Methods:**
- `await retrieval.insert(documents: list[str]) -> list[Hashable]` -- embed and store
- `await retrieval.forward(query: str) -> list[str]` -- embed query and search

**Key behaviors:**
- Requires both embedder AND vector_store configured on the module
- Returns `list[str]`, not `Message`

### 5.10 RAGModule

Retrieval + memory composition.

**Constructor:**
```python
pm.modules.RAGModule(n_results: int = 5, memory_size: int = 10)
```

**Forward:**
```python
async def forward(self, query: str, **kwargs) -> Message
```

**Example:**
```python
rag = (
    pm.modules.RAGModule()
    .with_llm(llm)
    .with_embedder(embedder)
    .with_vector_store(vector_store)
)
await rag.retrieval.insert(["doc1...", "doc2...", "doc3..."])
response = await rag.forward("What is the capital of Nandor?")
```

**Key behaviors:**
- Requires LLM + embedder + vector_store
- Internal structure: `self.retrieval` (RetrievalModule) + `self.memory_module` (MemoryModule)
- Customizable `query_template` Parameter (default: `"Context:\n{context}\n\nQuestion: {query}"`)
- Insert documents via `rag.retrieval.insert(docs)`

### 5.11 ResetMemoryContext

Context manager that clears all MemoryModules in a module hierarchy.

**Constructor:**
```python
pm.modules.ResetMemoryContext(*modules: Module)
```

**Usage:**
```python
with pm.modules.ResetMemoryContext(agent) as ctx:
    response = await agent.forward("Hello")
    # ctx.reset()  # manual mid-execution clear if needed
# All MemoryModules cleared on both enter and exit
```

**Key behaviors:**
- BFS traversal finds ALL MemoryModule instances at any depth
- Handles diamond patterns (shared submodules visited once)
- `ctx.reset()` for manual mid-execution clear
- Exception-safe (clears on exit even if exception is raised)

---

## 6. Decision Tree

```
Single LLM call with template?                  -> Prompt
Multi-turn conversation?                         -> MemoryModule
Structured data extraction?                      -> StructuralOutput
Agent that calls functions?
  Provider supports OpenAI function calling?     -> OpenaiToolCallingAgent
  Otherwise                                      -> ToolCallingAgent
Document search only?                            -> RetrievalModule
Search + conversation?                           -> RAGModule
Multiple agents sharing conversation?            -> shared Memory instance
Sub-agent delegation from tool loop?             -> Tool.from_module(module, handoff_history=True)
Clean session boundaries?                        -> ResetMemoryContext
Typed final output from tool agent?              -> OpenaiToolCallingAgent(structural_output=Model)
```

---

## 7. Prompt Writing Guide

### 7a. Using XML Tags in Prompts

XML tags create unambiguous boundaries for LLMs. Use them to separate instructions, context, examples, and output format.

**System prompt with XML sections:**
```python
SYSTEM_PROMPT = """
<instructions>
You are a customer support agent. Answer questions using the provided context.
If the answer is not in the context, say "I don't have that information."
</instructions>

<output_format>
Respond in 1-3 sentences. Be concise and helpful.
</output_format>

<examples>
User: What are your hours?
Assistant: We are open Monday through Friday, 9 AM to 5 PM EST.
</examples>
"""

agent = pm.modules.MemoryModule(system_prompt=SYSTEM_PROMPT).with_llm(llm)
```

**RAG context injection with XML tags:**
```python
QUERY_TEMPLATE = """
<context>
{context}
</context>

<question>
{query}
</question>
"""

rag = pm.modules.RAGModule()
rag.query_template.value = QUERY_TEMPLATE
```

**Multi-step reasoning prompt for tool agents:**
```python
AGENT_PROMPT = """
<instructions>
You solve math problems step by step using the provided tools.
Always verify intermediate results before proceeding to the next step.
</instructions>

<tools>
{tool_desc}
</tools>

<rules>
- Use one tool per step
- Never fabricate Observation: lines
- Show your reasoning in Thought: before each action
</rules>
"""

agent = pm.modules.ToolCallingAgent(
    tools=[...],
    prompt=AGENT_PROMPT,
    observation_role=pm.MessageRole.USER,
).with_llm(llm)
```

### 7b. Generalizing Prompts for TOML Serialization

Prompts stored in `pm.Parameter` are serialized to TOML and loaded back via `save()`/`load()`.

**Use `{placeholder}` for runtime values** passed as `**kwargs` to `forward()`:

```python
# Hardcoded (bad -- not reusable)
prompt = pm.Prompt("You are Henry, a helpful assistant.")

# Generalized (good -- TOML-friendly)
prompt = pm.Prompt("You are {name}, a helpful assistant.")
response = await prompt.forward(history, name="Henry")
```

**Escape literal braces** as `{{` and `}}` because `str.format_map` is used internally:

```python
# This prompt contains literal JSON braces
prompt = pm.Prompt(
    "Return JSON like {{'key': 'value'}}. Your name is {name}."
)
response = await prompt.forward(history, name="Henry")
```

**TOML round-trip:**
```python
agent.save("agent.toml")    # Parameters serialized
agent.load("agent.toml")    # Parameters restored
agent.describe()             # View TOML representation
```

Example TOML output:
```toml
[chat]
prompt = """
You are {name}, a helpful assistant.
"""

role = """
system
"""
```

**Rule:** If a value should change at runtime, use `{placeholder}` in the prompt. If it should change between deployments, put it in a separate `Parameter`.

---

## 8. Composition Patterns

### 8.1 Hello World Custom Module

```python
import promptimus as pm


class HelloAgent(pm.Module):
    def __init__(self):
        super().__init__()
        self.chat = pm.Prompt("You are a friendly assistant named {name}.")

    async def forward(self, question: str, name: str = "Bot") -> str:
        response = await self.chat.forward(
            [pm.Message(role=pm.MessageRole.USER, content=question)],
            name=name,
        )
        return response.content


agent = HelloAgent().with_llm(llm)
answer = await agent.forward("Hello!", name="Henry")
```

### 8.2 Conversational Agent with Tools

```python
import math
import promptimus as pm


def factorial(a: int) -> int:
    """Calculates the factorial of `a`"""
    return math.factorial(a)

def multiply(a: float, b: float) -> float:
    """Multiplies `a` and `b`"""
    return a * b


agent = pm.modules.OpenaiToolCallingAgent(
    prompt="You are a calculator assistant. Use tools step by step.",
    tools=[factorial, multiply],
).with_llm(pm.llms.OpenAILike(model_name="gpt-4.1-nano"))

response = await agent.forward("What is twice the factorial of 5?")
```

### 8.3 Agent-as-Tool Handoff

```python
import promptimus as pm


class ResearchAgent(pm.Module):
    """Performs deep research on a topic."""

    def __init__(self):
        super().__init__()
        self.memory = pm.modules.MemoryModule(
            memory_size=20,
            system_prompt="You are a research assistant. Analyze the conversation and provide detailed answers.",
        )

    async def forward(self, history: list[pm.Message] | pm.Message | str, **kwargs) -> pm.Message:
        return await self.memory.forward(history, **kwargs)


research = ResearchAgent()
research_tool = pm.modules.Tool.from_module(
    research,
    name="research",
    description="Delegate complex research questions to a specialist.",
    handoff_history=True,
)

orchestrator = pm.modules.OpenaiToolCallingAgent(
    prompt="Route research questions to the research tool.",
    tools=[research_tool],
).with_llm(llm)

# Use ResetMemoryContext for session cleanup
with pm.modules.ResetMemoryContext(orchestrator):
    response = await orchestrator.forward("Research the history of Python.")
```

### 8.4 Router + Planner + Quick/Slow Agents

```python
import promptimus as pm
from promptimus.modules.memory import Memory
from pydantic import BaseModel


class RouteModel(BaseModel):
    route: str  # "quick" or "slow"


class PlanItem(BaseModel):
    title: str
    status: str = "todo"
    what_was_done: str | None = None


class PlanModel(BaseModel):
    todos: list[PlanItem]


class Agent(pm.Module):
    def __init__(self, tools: list, memory_size: int = 50):
        super().__init__()
        self.shared_memory = Memory(memory_size)
        self.todos: list[PlanItem] = []

        self.router = pm.modules.StructuralOutput(
            RouteModel,
            retry_message_role=pm.MessageRole.USER,
        )
        self.planner = pm.modules.StructuralOutput(
            PlanModel,
            retry_message_role=pm.MessageRole.USER,
        )

        self.quick_agent = pm.modules.OpenaiToolCallingAgent(
            prompt="Handle simple requests quickly with minimal tool calls.",
            tools=tools,
            shared_memory=self.shared_memory,
            max_steps=5,
        )
        self.slow_agent = pm.modules.OpenaiToolCallingAgent(
            prompt="Handle complex requests step-by-step and update todo progress.",
            tools=[*tools, self.mark_completed, self.mark_skipped],
            shared_memory=self.shared_memory,
            max_steps=15,
        )

        self.todo_item_format = pm.Parameter(
            "{idx}. [{status}] {title} | done={what_was_done}"
        )

    def _format_todos(self, todos: list[PlanItem]) -> str:
        return "\n".join(
            self.todo_item_format.value.format(idx=i + 1, **todo.model_dump())
            for i, todo in enumerate(todos)
        )

    def mark_completed(self, todo_idx: int, description: str) -> str:
        assert 0 < todo_idx <= len(self.todos), "invalid todo_idx"
        todo = self.todos[todo_idx - 1]
        todo.status = "completed"
        todo.what_was_done = description
        return f"Todo {todo_idx} completed."

    def mark_skipped(self, todo_idx: int, description: str) -> str:
        assert 0 < todo_idx <= len(self.todos), "invalid todo_idx"
        todo = self.todos[todo_idx - 1]
        todo.status = "skipped"
        todo.what_was_done = description
        return f"Todo {todo_idx} skipped."

    async def forward(self, history: list[pm.Message], today: str) -> pm.Message:
        context = self.shared_memory.as_list() + history

        route = await self.router.forward(
            context,
            current_todos=self._format_todos(self.todos),
        )

        if route.route == "quick":
            response = await self.quick_agent.forward(history, today=today)
        else:
            if not self.todos:
                plan = await self.planner.forward(context)
                self.todos = plan.todos

            response = await self.slow_agent.forward(
                history,
                current_todos=self._format_todos(self.todos),
                today=today,
            )

        if self.todos and all(todo.status != "todo" for todo in self.todos):
            self.todos = []

        return response
```

### 8.5 Testing Pattern

```python
import pytest
import promptimus as pm
from promptimus.llms.dummy import DummyLLm


class MyAgent(pm.Module):
    def __init__(self):
        super().__init__()
        self.chat = pm.modules.MemoryModule(
            system_prompt="You are a test assistant.",
        )

    async def forward(self, history: list[pm.Message] | pm.Message | str, **kwargs) -> pm.Message:
        return await self.chat.forward(history, **kwargs)


@pytest.mark.asyncio
async def test_agent_responds():
    agent = MyAgent().with_llm(DummyLLm(message="Hello!", delay=0))

    with pm.modules.ResetMemoryContext(agent):
        response = await agent.forward("Hi")
        assert response.content == "Hello!"
        assert response.role == pm.MessageRole.ASSISTANT
```

---

## 9. Serialization

```python
agent.save("agent.toml")     # Store all Parameters + submodule state
agent.load("agent.toml")     # Restore Parameters + submodule state (modifies in-place, returns self)
agent.describe()              # Returns TOML string representation
agent.digest()                # Returns MD5 hex digest of all Parameters
```

Example TOML output:
```toml
[retriver]
top_k = 1
similarity_thr = 0.5

[query_generator]
prompt = """
Generate a query for RAG based on this question: `{question}`.
"""

role = """
user
"""
```

**Rule:** Only `Parameter` values and registered submodule state survive `save()`/`load()`. Plain Python attributes (lists, dicts, etc.) are NOT serialized.

---

## 10. Error Reference

All exceptions are in `promptimus.errors` and inherit from `PromptimusError`.

| Exception | Cause | Fix |
|-----------|-------|-----|
| `LLMNotSet` | Accessing `self.llm` without calling `.with_llm()` | Call `.with_llm(provider)` on root module |
| `EmbedderNotSet` | Accessing `self.embedder` without calling `.with_embedder()` | Call `.with_embedder(embedder)` on root module |
| `VectorStoreNotSet` | Accessing `self.vector_store` without calling `.with_vector_store()` | Call `.with_vector_store(store)` on root module |
| `ParamNotSet` | Accessing `Parameter.value` when value is `None` | Set the parameter value before access |
| `FailedToParseOutput` | `StructuralOutput` exhausted all retries | Strengthen prompt, check schema, increase `n_retries` |
| `MaxIterExceeded` | Tool agent exceeded `max_steps` | Increase `max_steps`, narrow tool descriptions |
| `InvalidToolName` | OpenAI agent received unknown tool name from LLM | Check tool registration, improve prompt |
| `ProviderRetryExceded` | LLM rate limit retries exhausted | Reduce concurrency, increase `n_retries`/`base_wait` |
| `RecursiveModule` | Circular dependency in module tree | Do not add an ancestor as a submodule |
| `InvalidToolConfig` | `handoff_include_tool_loop=True` without `handoff_history=True`, or `handoff_history=True` on a module not implementing `SupportsHandoff` | Fix the tool configuration flags |

---

## 11. Hard Rules

1. Do not mutate hidden global state to track conversation.
2. Do not bypass the module tree for critical runtime dependencies.
3. Do not keep complex business logic inside giant prompts when code/state can enforce it.
4. Do not leave tool outputs unvalidated when they drive state transitions.
5. Always call `super().__init__()` in `Module` subclasses.
6. Do not use `handoff_history=True` on modules that don't implement `SupportsHandoff`.
7. Always run `forward()` in an async context (`await`).
