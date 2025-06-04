from collections import deque

from tomlkit import document, nl, parse, string, table
from tomlkit.items import Table
from tomlkit.toml_document import TOMLDocument


def module_dict_to_toml_str(root_module: dict) -> str:
    """Converts dict representation of a module to a toml sting.

    - submodules: is stored as toml tables.
    - prompts: is stored as toml key value pairs in repsective table.
    """
    doc = document()

    q: deque[tuple[TOMLDocument | Table, dict[str, dict]]] = deque()

    q.append((doc, root_module))

    while q:
        container, module = q.pop()

        for param_name, param_value in module["params"].items():
            if isinstance(param_value, str):
                container.add(
                    param_name,
                    string(
                        f"\n{param_value.strip('\n')}\n",
                        multiline=True,
                    ),
                )
                container.add(nl())
            else:
                container.add(param_name, param_value)

        for submodule_name, submodule in module["submodules"].items():
            submodule_container = table()

            container.add(nl())
            container.add(submodule_name, submodule_container)

            q.append((submodule_container, submodule))

    return doc.as_string()


def module_dict_from_toml_str(toml_str: str) -> dict:
    """Converts toml representation of a module to a dict."""

    doc = parse(toml_str)
    q: deque[tuple[TOMLDocument | Table, dict[str, dict]]] = deque()
    root_module = {"params": {}, "submodules": {}}
    q.append((doc, root_module))

    while q:
        container, module = q.pop()

        for name, value in container.items():
            if isinstance(value, Table):
                submodue = {"params": {}, "submodules": {}}
                module["submodules"][name] = submodue
                q.append((value, submodue))

            elif isinstance(value, str):
                module["params"][name] = value.strip("\n")
            else:
                module["params"][name] = value

    return root_module
