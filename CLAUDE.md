# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Promptimus is a PyTorch-inspired framework for building composable LLM agents. The architecture follows PyTorch's modular design patterns, with `Module` as the base class, `Parameter` for prompts, and hierarchical composition of components.

## Development Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run single test
uv run pytest tests/test_filename.py::test_function_name

# Type checking
uv run ty check

# Lint and format
uv run ruff check --fix && uv run ruff format

# Pre-commit checks
pre-commit run --all-files

# Build package
uv build
```

## Architecture Overview

### Core Design Pattern: PyTorch-like Module System

The framework is built around three interconnected concepts:

1. **Module** (`src/promptimus/core/module.py`): Base abstract class for all components
   - Automatically tracks `Parameter` objects assigned as attributes (via `__setattr__`)
   - Automatically tracks submodules in `_submodules` dict
   - Provides hierarchical path tracking (`self.path`) through parent references
   - Implements fluent interface: `with_llm()`, `with_embedder()`, `with_vector_store()` propagate to all submodules
   - Enforces recursion checking to prevent circular module references
   - Requires `async def forward()` implementation (like PyTorch's `forward()`)

2. **Parameter** (`src/promptimus/core/parameters.py`): Generic container for module parameters
   - Tracks parent module and hierarchical path
   - Computes MD5 digest for change detection (used in `module.digest()`)
   - Primarily used for system prompts via `Prompt` module
   - Raises `ParamNotSet` if accessed before initialization

3. **Serialization** (`src/promptimus/core/checkpointing.py`): TOML-based persistence
   - `serialize()` → dict → TOML via `module_dict_to_toml_str()`
   - `load()` / `load_dict()` → deserialize from TOML
   - Uses BFS traversal with deque for nested module structures
   - Multiline strings for prompts, flat key-value pairs for other parameters

### Module Hierarchy Registration

When you assign an object to a Module instance:
```python
class MyAgent(Module):
    def __init__(self):
        super().__init__()
        self.prompt = Parameter("system prompt")  # Auto-registers in _parameters
        self.memory = MemoryModule(5)             # Auto-registers in _submodules
```

The `__setattr__` override automatically:
- Detects `Parameter` instances and adds to `_parameters` dict
- Detects `Module` instances and adds to `_submodules` dict
- Sets parent reference and name for path tracking

### Pre-built Modules (`src/promptimus/modules/`)

**Prompt** (`prompt.py`): Simplest module - formats a prompt string and calls LLM
- Uses `Parameter` for prompt text and role
- Implements `forward(history, provider_kwargs, **kwargs)` where `**kwargs` are format variables
- Calls `self.llm.achat()` with formatted messages

**MemoryModule** (`memory.py`): Conversation history with fixed-size deque
- Wraps `Memory` class (deque-based circular buffer)
- Extends history with new messages, calls `Prompt.forward()`, stores response
- Provides `add_message()`, `extend()`, `drop_last()`, `replace_last()` for manipulation
- Includes `ResetMemoryContext` class for automatic memory clearing (see Memory Context Manager section below)

**Tool** (`tool.py`): Function-to-tool converter
- `@Tool.decorate` decorator auto-generates descriptions from docstrings and signatures
- `Tool.from_module()` wraps any Module's `forward()` method as a tool
- Uses Pydantic's `validate_call` for parameter validation
- Handles both sync and async functions (converts sync to async via `asyncio.to_thread`)

**ToolCallingAgent** (`tool.py`): ReACT-style agent loop
- Takes list of `Tool` objects, implements thought→action→observation loop
- Parses LLM output for `<thought>`, `<action>`, `<observation>` XML tags
- `max_iter` prevents infinite loops (raises `MaxIterExceeded`)
- Two variants: `ToolCallingAgent` (ReACT) and `OpenaiToolCallingAgent` (native OpenAI function calling)

**RetrievalModule** (`retrieval.py`): Vector store operations
- Requires embedder and vector store via fluent interface
- `insert()` / `insert_batch()` for adding documents
- `forward()` embeds query and searches vector store

**RAGModule** (`rag.py`): Combines retrieval + memory + prompting
- Composes `RetrievalModule` + `MemoryModule`
- `forward()` retrieves context, injects into prompt, maintains conversation history

**StructuralOutput** (`structural.py`): Pydantic schema-based JSON extraction
- Takes Pydantic model as input, generates JSON schema
- Uses OpenAI's structured output mode or parses from markdown code blocks

### Provider Protocols

**LLMProtocol** (`src/promptimus/llms/base.py`):
- `async achat(messages, **kwargs) -> Message`
- Implementations: `OpenAILike`, `OllamaLLM`, `DummyLLM`

**EmbedderProtocol** (`src/promptimus/embedders/base.py`):
- `async aembed(text) -> list[float]`
- `async aembed_batch(texts) -> list[list[float]]`
- Implementation: `OpenAILikeEmbedder`

**VectorStoreProtocol** (`src/promptimus/vectore_store/base.py`):
- `async insert()`, `async search()`, `async delete()`, `async update()`
- External implementation: `ChromaVectorStore` in `libs/chromadb-store/`

### Memory Context Manager

The `ResetMemoryContext` class provides automatic memory clearing for MemoryModule instances. This is particularly useful for agent handoff scenarios where sub-agents are used as tools with history passthrough.

**Basic usage:**

```python
from promptimus.modules import ResetMemoryContext

# Single module
with ResetMemoryContext(agent):
    response = await agent.forward(query)
# All memories cleared on entry and exit

# Multiple modules
with ResetMemoryContext(agent1, agent2):
    await agent1.forward(query1)
    await agent2.forward(query2)
# All memories in both hierarchies cleared
```

**How it works:**
- Performs BFS traversal on instantiation to find ALL MemoryModule instances in the hierarchy
- Clears all found memories on context entry (clean slate)
- Clears all found memories on context exit (cleanup)
- Handles nested agents, diamond patterns, and ModuleDict collections
- Raises `ValueError` if no modules are provided

**Key feature: Eager Registration**

`ResetMemoryContext` uses eager registration via BFS traversal to find ALL MemoryModule instances in the module hierarchy upfront:

```python
class ParentModule(Module):
    def __init__(self):
        super().__init__()
        self.agent1 = MemoryModule(...)
        self.agent2 = MemoryModule(...)

parent = ParentModule()

with ResetMemoryContext(parent):
    # Both agent1 and agent2 memories cleared, even if only agent1 is used
    await parent.agent1.forward(query)
```

This ensures complete cleanup of all conversation state in the hierarchy.

**Manual control:**

```python
with ResetMemoryContext(agent) as ctx:
    await agent.forward(query1)
    ctx.reset()  # Manual clear mid-execution
    await agent.forward(query2)
```

The context manager provides a `reset()` method for manual control during execution.

**Multiple module support:**

```python
# Clear multiple independent module hierarchies
with ResetMemoryContext(agent1, agent2, agent3):
    await agent1.forward(query1)
    await agent2.forward(query2)
```

**Nested agents with tools:**

```python
main_agent = ToolCallingAgent([
    Tool.from_module(sub_agent, handoff_history=True)
])
with ResetMemoryContext(main_agent):
    await main_agent.forward(query)
# Both main_agent and sub_agent memories cleared
```

**Use cases:**
- Fresh conversation contexts (isolate user sessions)
- Testing with isolated state (prevent test interference)
- Multi-session handling (reset between sessions)
- Preventing memory leaks in long-running agents
- Nested agent handoff (clear sub-agent memories after tool calls)
- Clearing complex agent hierarchies (RAG, ToolCalling, multi-agent systems)

**Implementation details:**
- Uses BFS traversal to find all MemoryModule instances at any depth
- Handles diamond patterns (same module in multiple branches) without duplicates
- Clears on both entry and exit for complete state management
- Exception-safe: clears memory even if agent raises exception
- Supports multiple root modules via varargs (`*modules`)
- Implementation in `src/promptimus/modules/memory.py:91-155`
- Tests in `tests/test_memory_reset.py`

### Workspace Structure

This is a UV workspace with multiple packages:
- **Main package**: `src/promptimus/` (core framework)
- **Plugin: phoenix-tracer**: `libs/phoenix-tracer/` (Arize Phoenix tracing integration)
- **Plugin: chromadb-store**: `libs/chromadb-store/` (ChromaDB vector store implementation)

`pyproject.toml` declares workspace members and sources. Plugins are optional dependencies.

## Code Style Conventions

### Imports
- Use relative imports within package: `from . import module`, `from .core import Module`
- Group: stdlib → third-party → local (with blank lines between)

### Type Hints
- Comprehensive type hints required for all functions
- Use `Self` from `typing` for fluent method returns
- Union types with `|` syntax (Python 3.10+)
- Avoid `Any` - prefer specific types

### Naming
- Classes: `PascalCase` (e.g., `MemoryModule`, `ToolCallingAgent`)
- Functions/methods: `snake_case` (e.g., `with_llm`, `aembed_batch`)
- Private attributes: `_leading_underscore` (e.g., `_parameters`, `_llm`)

### Patterns
- Double quotes for strings
- f-strings for formatting
- async/await for all I/O operations
- ABC for interfaces
- Pydantic for data validation
- Fluent interfaces for configuration (return `Self`)

### Error Handling
- Custom exceptions in `src/promptimus/errors.py`
- Examples: `LLMNotSet`, `EmbedderNotSet`, `RecursiveModule`, `MaxIterExceeded`
- Descriptive error messages with context

## Key Implementation Details

### Why Fluent Interface Pattern
The `with_llm()`, `with_embedder()`, `with_vector_store()` methods propagate configuration down the entire module tree. This allows you to configure a complex agent hierarchy in one call:
```python
agent = RAGModule(...).with_llm(llm).with_embedder(embedder).with_vector_store(store)
# All submodules (RetrievalModule, MemoryModule, Prompt) automatically get these dependencies
```

### Digest System
Each `Parameter` and `Module` computes an MD5 digest used for change detection. When parameters change, digests change, allowing downstream caching/memoization systems to detect modifications.

### TOML Serialization Format
Parameters are stored as key-value pairs. Submodules become TOML tables. Multiline strings use triple-quote syntax:
```toml
prompt = """
You are a helpful assistant.
"""

[memory]
memory_size = 5
```

### Tool Calling Architecture
Two approaches:
1. **ReACT**: LLM outputs XML tags, agent parses and executes in loop
2. **Native OpenAI**: Uses OpenAI's function calling API directly

Both use the same `Tool` wrapper but different execution strategies in `ToolCallingAgent` vs `OpenaiToolCallingAgent`.

## Testing

Tests focus on serialization/checkpointing (`tests/test_chekpointing.py`). When adding features:
- Test TOML round-trip: `module.save()` → `load()` → verify equality
- Test module composition and hierarchy
- Test async operations with `pytest-asyncio`

## Tutorials

Five Jupyter notebooks in `notebooks/` demonstrate framework usage:
1. LLM providers and embedders
2. Core Prompt and Module concepts
3. Pre-built modules (Memory, RAG, Retrieval)
4. Tool calling and custom agents
5. Phoenix tracing integration

These serve as integration tests and documentation.
