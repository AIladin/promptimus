# Agent Guidelines for Promptimus

## Build/Lint/Test Commands
- **Install dependencies**: `uv sync`
- **Run all tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/test_filename.py::test_function_name`
- **Type check**: `uv run pyright`
- **Lint and format**: `uv run ruff check --fix && uv run ruff format`
- **Pre-commit checks**: `pre-commit run --all-files`
- **Build package**: `uv build`

## Code Style Guidelines

### Imports
- Use relative imports within the package: `from . import module`
- Standard library first, then third-party, then local imports
- Group imports with blank lines between groups

### Types and Annotations
- Use comprehensive type hints for all function parameters and return values
- Use `Self` from `typing` for fluent method chaining
- Use union types with `|` syntax (Python 3.10+)
- Use `Any` sparingly, prefer specific types

### Naming Conventions
- Classes: PascalCase (e.g., `Module`, `Parameter`)
- Functions/methods: snake_case (e.g., `with_llm`, `serialize`)
- Variables: snake_case (e.g., `module_dict`, `checkpoint`)
- Constants: UPPER_CASE (e.g., `TOML_STRING`)
- Private attributes: leading underscore (e.g., `_parameters`)

### Error Handling
- Use custom error classes from `promptimus.errors`
- Raise specific exceptions rather than generic ones
- Use descriptive error messages

### Code Patterns
- Use double quotes for strings
- Use f-strings for string formatting
- Use async/await for asynchronous operations
- Use ABC (Abstract Base Class) for interfaces
- Use Pydantic for data validation and serialization
- Use loguru for logging
- Prefer fluent interfaces with method chaining

### File Structure
- Keep related functionality in modules under `src/promptimus/`
- Use `__init__.py` files for package initialization
- Include `py.typed` marker for type checking support