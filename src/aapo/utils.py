from aapo.dto import Message

DEFAULT_SEPARATOR = "-" * 20 + "\n\n"


def prettify_history(history: list[Message], sep: str = DEFAULT_SEPARATOR) -> str:
    return sep.join((message.prettify() for message in history))
