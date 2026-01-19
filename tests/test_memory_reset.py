import pytest

from promptimus import Message, MessageRole, Module
from promptimus.llms.dummy import DummyLLm
from promptimus.modules import MemoryModule, ResetMemoryContext


@pytest.mark.asyncio
async def test_basic_reset():
    """Test that memory is cleared on entry and exit."""
    agent = MemoryModule(memory_size=10, system_prompt="").with_llm(DummyLLm(delay=0))

    # Add initial message
    await agent.forward("initial")
    assert len(agent.memory.as_list()) == 2

    with ResetMemoryContext(agent):
        # Memory cleared on entry
        assert len(agent.memory.as_list()) == 0

        await agent.forward("test message")
        assert len(agent.memory.as_list()) == 2

    # Memory cleared on exit
    assert len(agent.memory.as_list()) == 0


@pytest.mark.asyncio
async def test_nested_modules():
    """Test that BFS traversal finds and clears nested MemoryModule instances."""

    class ParentModule(Module):
        def __init__(self):
            super().__init__()
            self.child1 = MemoryModule(memory_size=10, system_prompt="").with_llm(
                DummyLLm(delay=0)
            )
            self.child2 = MemoryModule(memory_size=10, system_prompt="").with_llm(
                DummyLLm(delay=0)
            )

        async def forward(self, query):
            return await self.child1.forward(query)

    parent = ParentModule()

    # Add messages to both children
    await parent.child1.forward("msg1")
    parent.child2.add_message(Message(role=MessageRole.USER, content="msg2"))

    assert len(parent.child1.memory.as_list()) == 2
    assert len(parent.child2.memory.as_list()) == 1

    with ResetMemoryContext(parent):
        # Both cleared on entry (eager BFS finds both)
        assert len(parent.child1.memory.as_list()) == 0
        assert len(parent.child2.memory.as_list()) == 0

        await parent.forward("test")

    # Both cleared on exit
    assert len(parent.child1.memory.as_list()) == 0
    assert len(parent.child2.memory.as_list()) == 0


@pytest.mark.asyncio
async def test_multiple_modules():
    """Test clearing multiple independent module hierarchies with varargs."""
    agent1 = MemoryModule(memory_size=10, system_prompt="").with_llm(DummyLLm(delay=0))
    agent2 = MemoryModule(memory_size=10, system_prompt="").with_llm(DummyLLm(delay=0))
    agent3 = MemoryModule(memory_size=10, system_prompt="").with_llm(DummyLLm(delay=0))

    # Add messages
    await agent1.forward("msg1")
    await agent2.forward("msg2")
    await agent3.forward("msg3")

    assert len(agent1.memory.as_list()) == 2
    assert len(agent2.memory.as_list()) == 2
    assert len(agent3.memory.as_list()) == 2

    # Clear all three
    with ResetMemoryContext(agent1, agent2, agent3):
        assert len(agent1.memory.as_list()) == 0
        assert len(agent2.memory.as_list()) == 0
        assert len(agent3.memory.as_list()) == 0

    # All still cleared
    assert len(agent1.memory.as_list()) == 0
    assert len(agent2.memory.as_list()) == 0
    assert len(agent3.memory.as_list()) == 0
