from collections import deque
from typing import Self

from promptimus.core import Module
from promptimus.dto import Message, MessageRole

from .prompt import Prompt


class Memory:
    def __init__(self, size: int):
        self.data: deque[Message] = deque(maxlen=size)

    def add_message(self, message: Message) -> Self:
        self.data.append(message)
        return self

    def extend(self, history: list[Message]) -> Self:
        self.data.extend(history)
        return self

    def replace_last(self, message: Message):
        self.data[-1] = message

    def drop_last(self, n: int = 1):
        for _ in range(n):
            self.data.pop()

    def reset(self):
        self.data.clear()

    def __enter__(self):
        self.reset()
        return

    def __exit__(self, *args, **kwargs):
        self.reset()

    def as_list(self) -> list[Message]:
        return list(self.data)

    def __repr__(self) -> str:
        return f"Memory[{self.data}]"


class MemoryModule(Module):
    def __init__(
        self,
        memory_size: int = 10,
        shared_memory: Memory | None = None,
        system_prompt: str | None = None,
        new_message_role: MessageRole | str = MessageRole.USER,
    ):
        super().__init__()

        self.new_message_role = new_message_role
        self.prompt = Prompt(system_prompt)
        if shared_memory is not None:
            self.memory = shared_memory
        else:
            self.memory = Memory(memory_size)

    async def forward(
        self, history: list[Message] | Message | str, **kwargs
    ) -> Message:
        if isinstance(history, Message):
            history = [history]
        elif isinstance(history, str):
            history = [Message(role=self.new_message_role, content=history)]

        self.memory.extend(history)

        while self.memory.data[0].role == MessageRole.TOOL:
            self.memory.data.popleft()

        response = await self.prompt.forward(self.memory.as_list(), **kwargs)
        self.memory.add_message(response)

        return response

    def add_message(self, message: Message) -> Self:
        self.memory.add_message(message)
        return self

    def extend(self, history: list[Message]) -> Self:
        self.memory.extend(history)
        return self

    def drop_last(self, n: int = 1):
        self.memory.drop_last(n)
        return self

    def replace_last(self, message: Message):
        self.memory.replace_last(message)
        return self


class ResetMemoryContext:
    """
    Context manager that eagerly finds and resets all MemoryModule instances
    in provided module hierarchies.

    Performs BFS traversal to find ALL MemoryModule instances at any nesting
    depth. Clears memories on both context entry and exit.

    This class is typically not instantiated directly. Use the reset_memory()
    function instead.
    """

    def __init__(self, *modules: Module):
        if not modules:
            raise ValueError("At least one module must be provided to reset_memory()")
        self._root_modules = modules
        self._memory_modules = self._collect_memory_modules()

    def _collect_memory_modules(self) -> list["MemoryModule"]:
        """
        Collect all MemoryModule instances from module hierarchies via BFS traversal.

        Args:
            root_modules: One or more root Module instances to traverse

        Returns:
            List of all MemoryModule instances found in the hierarchies
        """
        memory_modules = []
        visited = set()
        queue = deque(self._root_modules)

        while queue:
            module = queue.popleft()

            # Skip if already visited (handles diamond patterns)
            module_id = id(module)
            if module_id in visited:
                continue
            visited.add(module_id)

            # Register if MemoryModule
            if isinstance(module, MemoryModule):
                memory_modules.append(module)

            # Add all submodules to queue
            queue.extend(module._submodules.values())

        return memory_modules

    def __enter__(self) -> Self:
        """Traverse hierarchy, register memories, clear on entry."""
        self.reset()  # Clear on entry
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Clear all registered memories on exit."""
        self.reset()  # Clear on exit
        return None

    def reset(self) -> None:
        """Clear all registered MemoryModule instances."""
        for memory_module in self._memory_modules:
            memory_module.memory.reset()
