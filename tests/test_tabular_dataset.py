from aapo.data.dataset import TabularDataset
from aapo.dto import Message, MessageRole
import polars as pl

data = pl.DataFrame(
    data=[
        [0, 0, MessageRole.USER, "0"],
        [0, 1, MessageRole.ASSISTANT, "1"],
        [0, 2, MessageRole.USER, "2"],
        [0, 3, MessageRole.ASSISTANT, "3"],
        [0, 4, MessageRole.USER, "4"],
        [0, 5, MessageRole.ASSISTANT, "5"],
        [1, 0, MessageRole.USER, "6"],
        [1, 1, MessageRole.ASSISTANT, "7"],
        [1, 2, MessageRole.USER, "8"],
        [1, 3, MessageRole.ASSISTANT, "9"],
        [1, 4, MessageRole.USER, "10"],
        [1, 5, MessageRole.ASSISTANT, "11"],
    ],
    schema=[
        ("session_id", pl.Int32),
        ("order", pl.Int32),
        ("role", pl.Enum(MessageRole)),
        ("content", pl.String),
    ],
    orient="row",
)

dataset = TabularDataset(data)


def test_len():
    assert len(dataset) == 6


def test_getitem_0():
    x, y = dataset[0]

    assert x == [Message(role=MessageRole.USER, content="0")]
    assert y == Message(role=MessageRole.ASSISTANT, content="1")


def test_getitem_second_session():
    x, y = dataset[3]

    assert x == [Message(role=MessageRole.USER, content="6")]
    assert y == Message(role=MessageRole.ASSISTANT, content="7")


def test_getitem_mid_session():
    x, y = dataset[4]

    assert x == [
        Message(role=MessageRole.USER, content="6"),
        Message(role=MessageRole.ASSISTANT, content="7"),
        Message(role=MessageRole.USER, content="8"),
    ]
    assert y == Message(role=MessageRole.ASSISTANT, content="9")


def test_iter():
    for _ in dataset:
        pass
