# Project Index: Promptimus

**Generated**: 2026-01-14
**Version**: 0.1.11
**Description**: A PyTorch-like API for building composable LLM agents

---

## 📁 Project Structure

```
promptimus/
├── src/promptimus/           # Main package source
│   ├── core/                 # Core module system (Module, Parameter, checkpointing)
│   ├── dto/                  # Data transfer objects (Message, Tool, Image)
│   ├── llms/                 # LLM provider implementations
│   ├── embedders/            # Embedding providers
│   ├── vectore_store/        # Vector store protocols
│   ├── modules/              # Pre-built modules (Memory, RAG, Tools, etc.)
│   ├── common/               # Shared utilities (rate limiting)
│   └── errors.py             # Custom exceptions
├── libs/                     # Optional plugin packages
│   ├── phoenix-tracer/       # Arize Phoenix tracing integration
│   └── chromadb-store/       # ChromaDB vector store implementation
├── notebooks/                # Tutorial notebooks (5 comprehensive guides)
├── tests/                    # Test suite
└── dist/                     # Build artifacts

```

---

## 🚀 Entry Points

- **Main Package**: `src/promptimus/__init__.py` - Core API exports
- **CLI/Scripts**: None (library package)
- **Tests**: `tests/test_chekpointing.py` - Serialization tests (188 lines)
- **Notebooks**: `notebooks/` - 5 interactive tutorials

---

## 📦 Core Modules

### Module: core
- **Path**: `src/promptimus/core/`
- **Exports**: `Module`, `Parameter`
- **Key Files**:
  - `module.py` - Base Module class with PyTorch-like architecture
  - `parameters.py` - Parameter system for prompts
  - `checkpointing.py` - TOML-based serialization
- **Purpose**: Foundation for composable agent architecture

### Module: dto
- **Path**: `src/promptimus/dto/`
- **Exports**: `Message`, `MessageRole`, `ImageContent`, `Tool`
- **Purpose**: Standardized data structures for LLM interactions

### Module: llms
- **Path**: `src/promptimus/llms/`
- **Exports**: `OpenAILike`, `OllamaLLM`, `DummyLLM`
- **Key Files**:
  - `base.py` - Abstract LLM interface
  - `openai.py` - OpenAI/compatible API provider
  - `ollama.py` - Local Ollama provider
  - `dummy.py` - Testing mock
- **Purpose**: LLM provider abstraction layer

### Module: embedders
- **Path**: `src/promptimus/embedders/`
- **Exports**: `OpenAILikeEmbedder`, `BaseEmbedder`
- **Key Files**:
  - `base.py` - Abstract embedder interface
  - `openai.py` - OpenAI embeddings provider
  - `ops.py` - Embedding operations
- **Purpose**: Text embedding generation with batch processing

### Module: modules (Pre-built Components)
- **Path**: `src/promptimus/modules/`
- **Exports**:
  - `Prompt` - System prompt management
  - `MemoryModule` - Conversation memory
  - `RetrievalModule` - Vector database operations
  - `RAGModule` - Retrieval-Augmented Generation
  - `StructuralOutput` - Pydantic schema-based JSON
  - `Tool` - Tool decorator
  - `ToolCallingAgent` - ReACT-style agents
  - `OpenaiToolCallingAgent` - Native OpenAI function calling
- **Purpose**: Ready-to-use building blocks for agent construction

### Module: vectore_store
- **Path**: `src/promptimus/vectore_store/`
- **Exports**: `VectorStore` protocol
- **Purpose**: Vector database interface abstraction

### Module: common
- **Path**: `src/promptimus/common/`
- **Exports**: Rate limiting utilities
- **Purpose**: Shared helper functionality

---

## 🔌 Optional Plugins

### Plugin: phoenix-tracer
- **Package**: `promptimus-phoenix-tracer` (v0.1.4)
- **Path**: `libs/phoenix-tracer/`
- **Dependencies**: `arize-phoenix>=8.20.0`, OpenInference conventions
- **Purpose**: Comprehensive LLM observability and tracing
- **Install**: `pip install promptimus[phoenix]`

### Plugin: chromadb-store
- **Package**: `promptimus-chromadb-store` (v0.1.4)
- **Path**: `libs/chromadb-store/`
- **Dependencies**: `chromadb>=1.0.15`
- **Purpose**: ChromaDB vector store implementation for RAG
- **Install**: `pip install promptimus[chromadb]`

---

## 🔧 Configuration Files

- **pyproject.toml** - Main project metadata, dependencies, build config, UV workspace
- **libs/phoenix-tracer/pyproject.toml** - Phoenix plugin metadata
- **libs/chromadb-store/pyproject.toml** - ChromaDB plugin metadata
- **.pre-commit-config.yaml** - Pre-commit hooks (linting, formatting)
- **uv.lock** - Locked dependency tree
- **AGENTS.md** - Agent developer guidelines (build commands, code style)
- **LICENSE** - MIT License

---

## 📚 Documentation

### Root Documentation
- **README.md** - Main project documentation with quickstart, features, examples
- **AGENTS.md** - Code style guidelines and build commands for contributors

### Tutorial Notebooks (notebooks/)
1. **step_1_llm_provider.ipynb** - LLM providers and embedders
2. **step_2_prompts_and_modules.ipynb** - Core architecture concepts
3. **step_3_prebuit_modules.ipynb** - Pre-built modules (Memory, RAG, Retrieval)
4. **step_4_custom_agent.ipynb** - Tool calling and custom agents
5. **step_5_tracing.ipynb** - Phoenix observability integration

### Configuration Assets
- **notebooks/assets/** - Example TOML configs (structural, memory, QA modules)

---

## 🧪 Test Coverage

- **Test files**: 1 (test_chekpointing.py)
- **Test lines**: 188
- **Focus**: Serialization and checkpointing functionality
- **Framework**: pytest with async support

---

## 🔗 Key Dependencies

### Core Dependencies
- **openai** (>=1.58.1) - LLM API client
- **pydantic** (>=2.10.4) - Data validation and serialization
- **loguru** (>=0.7.3) - Logging
- **tomlkit** (>=0.13.2) - TOML config parsing

### Optional Dependencies
- **arize-phoenix** (>=8.20.0) - Tracing and observability
- **chromadb** (>=1.0.15) - Vector database for RAG
- **pytest** (>=8.3.4) - Testing framework
- **pre-commit** (>=4.2.0) - Code quality hooks
- **ipykernel** (>=6.29.5) - Jupyter notebook support

---

## 📝 Quick Start

### Installation
```bash
# Basic installation
pip install promptimus

# With Phoenix tracing
pip install promptimus[phoenix]

# With ChromaDB for RAG
pip install promptimus[chromadb]
```

### Development Setup
```bash
# Clone and install
git clone https://github.com/AIladin/promptimus.git
cd promptimus
uv sync

# Run tests
uv run pytest

# Type checking
uv run ty check

# Linting
uv run ruff check --fix && uv run ruff format
```

### Basic Usage
```python
import promptimus as pm

# Create LLM and agent
llm = pm.llms.OpenAILike(model_name="gpt-4", api_key="...")
agent = pm.modules.MemoryModule(memory_size=5).with_llm(llm)

# Use the agent
response = await agent.forward("Hello!")
```

---

## 🏗️ Architecture Highlights

### PyTorch-Inspired Design
- **Module System**: Composable hierarchical architecture
- **Parameter Objects**: Prompts as first-class parameters
- **Checkpointing**: Save/load module configurations to TOML
- **Fluent Interfaces**: Method chaining (`with_llm()`, `with_embedder()`)

### Key Features
- ✅ Async-first API design
- ✅ Tool calling with automatic schema generation
- ✅ Structured output via Pydantic schemas
- ✅ Conversation memory management
- ✅ RAG with vector store integration
- ✅ Full observability via Phoenix tracing
- ✅ OpenAI and Ollama provider support

### Design Patterns
- Abstract base classes for extensibility
- Protocol-based interfaces for vector stores
- Decorator pattern for tool registration
- Builder pattern via fluent interfaces
- Serialization via TOML for human-readable configs

---

## 📊 Project Stats

- **Python Version**: >=3.12
- **Package Version**: 0.1.11
- **Core Modules**: 8 (core, dto, llms, embedders, vectore_store, modules, common, errors)
- **Pre-built Components**: 8 (Prompt, Memory, Retrieval, RAG, Structural, Tool, ToolCallingAgent, OpenaiToolCallingAgent)
- **Plugin Packages**: 2 (phoenix-tracer, chromadb-store)
- **Tutorial Notebooks**: 5
- **License**: MIT
- **Repository**: https://github.com/AIladin/promptimus

---

## 🎯 Use Cases

1. **Conversational Agents** - Memory-based chatbots
2. **RAG Applications** - Document Q&A with vector search
3. **Tool-Using Agents** - ReACT-style autonomous agents
4. **Structured Extraction** - Schema-based data extraction
5. **Multi-Step Workflows** - Composable agent pipelines
6. **Research & Experimentation** - Observable LLM development

---

**Index Size**: ~5KB
**Last Updated**: 2026-01-14
**Maintained By**: ailadin (korzhukandrew@gmail.com)
