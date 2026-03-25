import re
from collections import deque
from importlib.resources import files as pkg_files
from pathlib import Path

from tomlkit import document, nl, parse, string, table
from tomlkit.items import Table
from tomlkit.toml_document import TOMLDocument

from promptimus.errors import RefResolutionError

_PKG_REF = re.compile(r"^\$\{pkg:(.+\.toml)\}$")
_FILE_REF = re.compile(r"^\$\{file:(.+\.toml)\}$")


def _resolve_pkg(path: str) -> str:
    """Resolve ${pkg:dotted.package.file.toml} via importlib.resources."""
    parts = path.rsplit(".", 2)
    if len(parts) < 3:
        raise RefResolutionError(
            f"Invalid pkg reference '{path}': "
            "expected format 'package.path.filename.toml'"
        )
    package = ".".join(parts[:-2])
    filename = f"{parts[-2]}.{parts[-1]}"
    try:
        resource = pkg_files(package) / filename
        return resource.read_text(encoding="utf-8")
    except ModuleNotFoundError:
        raise RefResolutionError(
            f"Cannot resolve package '{package}' in ${{pkg:{path}}}"
        )
    except FileNotFoundError:
        raise RefResolutionError(
            f"Resource '{filename}' not found in package '{package}'"
        )


def _resolve_file(path: str, base_path: Path | None) -> str:
    """Resolve ${file:./relative.toml} relative to base_path."""
    if base_path is None:
        raise RefResolutionError(
            f"Cannot resolve ${{file:{path}}} without a base path. "
            "Use Module.load() with a file path."
        )
    resolved = (base_path / path).resolve()
    if not resolved.is_file():
        raise RefResolutionError(f"File '{path}' not found (resolved to '{resolved}')")
    return resolved.read_text(encoding="utf-8")


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


def module_dict_from_toml_str(
    toml_str: str,
    base_path: Path | None = None,
) -> dict:
    """Converts toml representation of a module to a dict.

    Supports ${pkg:...} and ${file:...} references that resolve external
    .toml files as submodule definitions (no nesting — referenced files
    must not contain refs).
    """

    doc = parse(toml_str)
    q: deque[tuple[TOMLDocument | Table, dict[str, dict]]] = deque()
    root_module = {"params": {}, "submodules": {}}
    q.append((doc, root_module))

    while q:
        container, module = q.pop()

        for name, value in container.items():
            match value:
                case str() if m := _PKG_REF.match(value.strip("\n")):
                    content = _resolve_pkg(m.group(1))
                    module["submodules"][name] = module_dict_from_toml_str(content)

                case str() if m := _FILE_REF.match(value.strip("\n")):
                    content = _resolve_file(m.group(1), base_path)
                    module["submodules"][name] = module_dict_from_toml_str(content)

                case Table():
                    submodule = {"params": {}, "submodules": {}}
                    module["submodules"][name] = submodule
                    q.append((value, submodule))

                case str():
                    module["params"][name] = value.strip("\n")

                case _:
                    module["params"][name] = value

    return root_module
