from aapo.dto import Sample, MessageRole, Message, History
import polars as pl
from functools import cached_property


class TabularDataset:
    """

    Table structure

    Cols:
    - session_id: Any
    - order: int
    - role: Enum
    - content: str

    """

    def __init__(self, df: pl.DataFrame):
        self.df = df.select("session_id", "order", "role", "content").sort(
            "session_id", "order"
        )
        self._len = 0

    @cached_property
    def len(self) -> int:
        _, count = (
            self.df.select(pl.col("role").value_counts(name="count"))
            .unnest("role")
            .row(by_predicate=pl.col("role") == MessageRole.ASSISTANT)
        )

        assert isinstance(count, int)

        return count

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Sample:
        if idx >= self.len or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.len}")

        response_session_id, response_order, response_content = (
            self.df.filter(pl.col("role") == MessageRole.ASSISTANT)
            .select("session_id", "order", "content")
            .row(idx)
        )

        history = History.validate_python(
            self.df.filter(
                pl.col("session_id") == response_session_id,
                pl.col("order") < response_order,
            )
            .select(
                "role",
                "content",
            )
            .to_dicts()
        )
        return Sample(
            x=history, y=Message(role=MessageRole.ASSISTANT, content=response_content)
        )
