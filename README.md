# Promptimus 🧠

A PyTorch-like API for building composable LLM agents with advanced tool calling, memory management, and observability.

[![PyPI version](https://badge.fury.io/py/promptimus.svg)](https://pypi.org/project/promptimus/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ Key Features

- 🧠 **PyTorch-like Modules**: Composable agent architecture with hierarchical module system
- 🔧 **Tool Calling**: ReACT-style and native OpenAI function calling with automatic schema generation
- 📝 **Structured Output**: Pydantic schema-based JSON generation with validation
- 💾 **Memory Management**: Conversation context with configurable memory limits
- 🔍 **Embeddings**: Text embedding generation with batch processing support
- 📊 **Tracing**: Arize Phoenix integration for comprehensive observability
- 💾 **Serialization**: TOML-based save/load for prompts and module configurations
- ⚡ **Async First**: Built for high-performance asynchronous operations

## 🚀 Quick Start

### Installation

```bash
pip install promptimus
```

### Basic Example

```python
import promptimus as pm

# Create an LLM provider
llm = pm.llms.OpenAILike(
    model_name="gpt-4",
    api_key="your-api-key"
)

# Create a simple agent with memory
agent = pm.modules.MemoryModule(
    memory_size=5,
    system_prompt="You are a helpful assistant."
).with_llm(llm)

# Have a conversation
response1 = await agent.forward("Hi, I'm Alice!")
response2 = await agent.forward("What's my name?")
print(response2.content)  # "Your name is Alice!"
```

## 🏗️ Architecture

### Core Concepts

**Modules**: Container system for organizing prompts, submodules, and logic
```python
class MyAgent(pm.Module):
    def __init__(self):
        super().__init__()
        self.chat = pm.Prompt("You are a helpful assistant")
        self.memory = []

    async def forward(self, message: str) -> str:
        # Custom logic here
        pass
```

**Prompts**: Parameter-like system for system prompts (similar to PyTorch parameters)
```python
prompt = pm.Prompt("You are a {role} assistant").with_llm(llm)
response = await prompt.forward(role="helpful")
```

**Tools**: Function decoration for external capabilities
```python
@pm.modules.Tool.decorate
def calculate(a: float, b: float, operation: str) -> float:
    """Calculate result of operation on two numbers."""
    if operation == "add":
        return a + b
    # ... more operations
```

### Pre-built Modules

**Memory Module**: Conversation memory with configurable limits
```python
agent = pm.modules.MemoryModule(
    memory_size=10,
    system_prompt="You are a helpful assistant."
).with_llm(llm)
```

**Structural Output**: Pydantic schema-based JSON generation
```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

module = pm.modules.StructuralOutput(Person).with_llm(llm)
result = await module.forward("Extract info about John, a 30-year-old engineer")
```

**Tool Calling Agents**: Agents that can use tools autonomously
```python
agent = pm.modules.ToolCallingAgent([
    calculate,
    # ... more tools
]).with_llm(llm)

result = await agent.forward("What is 15 + 27?")
```

## 📚 Example Use Cases

### Conversational AI with Memory
```python
agent = pm.modules.MemoryModule(
    memory_size=5,
    system_prompt="You are a friendly assistant."
).with_llm(llm)

await agent.forward("My favorite color is blue")
await agent.forward("What color do I like?")  # Remembers "blue"
```

### Tool-Augmented Calculator
```python
@pm.modules.Tool.decorate
def power(base: float, exponent: float) -> float:
    """Calculate base raised to exponent."""
    return base ** exponent

calculator = pm.modules.ToolCallingAgent([
    power,
    # ... more math tools
]).with_llm(llm)

result = await calculator.forward("What is 2 to the power of 8?")
```

## 🔧 Advanced Features

### Serialization
Save and load module configurations:
```python
agent.save("my_agent.toml")
loaded_agent = pm.modules.MemoryModule().load("my_agent.toml")
```

### Tracing with Phoenix
```python
import phoenix as px
from phoenix_tracer import trace

px.launch_app()
trace(agent, "my_agent", project_name="my_project")
```

### Custom Embedders
```python
embedder = pm.embedders.OpenAILikeEmbedder(
    model_name="text-embedding-3-small"
)

embeddings = await embedder.aembed_batch([
    "Hello world",
    "How are you?"
])
```

## 📖 Documentation

### Tutorials
Explore our comprehensive notebook tutorials:

1. **[LLM Providers & Embedders](notebooks/step_1_llm_provider.ipynb)** - Getting started with providers
2. **[Prompts & Modules](notebooks/step_2_prompts_and_modules.ipynb)** - Core architecture concepts
3. **[Pre-built Modules](notebooks/step_3_prebuit_modules.ipynb)** - Ready-to-use components
4. **[Custom Agents](notebooks/step_4_custom_agent.ipynb)** - Tool calling and advanced agents
5. **[Tracing](notebooks/step_5_tracing.ipynb)** - Observability with Phoenix

### API Reference
- `pm.Module`: Base class for all modules
- `pm.Prompt`: System prompt management
- `pm.llms.*`: LLM provider implementations
- `pm.embedders.*`: Embedding provider implementations
- `pm.modules.*`: Pre-built module components

## 🛠️ Installation Options

### Basic Installation
```bash
pip install promptimus
```

### With Optional Dependencies
```bash
# Phoenix tracing support
pip install promptimus[phoenix]

# ChromaDB vector store
pip install promptimus[chromadb]

# All optional dependencies
pip install promptimus[all]
```

### Development Setup
```bash
git clone https://github.com/AIladin/promptimus.git
cd promptimus
pip install -e .[dev]
```

## 🙏 Acknowledgments

- Inspired by PyTorch's modular architecture
- Built on top of modern Python async patterns
- Integrated with [Arize Phoenix](https://phoenix.arize.com/) for tracing
- Compatible with OpenAI and OpenAI-compatible APIs

---

**Ready to build your next LLM agent?** Check out our [tutorials](notebooks/) to get started! 🚀
