from pydantic import BaseModel

TEMPLATE = """
## {model_name} JSON model.
{model_description}

Fields:
{fields}
"""

FIELD_TEMPLATE = """
- {name}: {field_type}; {description}
"""


def format_pydantic_schema(schema: type[BaseModel]) -> str:
    field_desc = []
    for fname, finfo in schema.model_fields.items():
        field_desc.append(
            FIELD_TEMPLATE.format(
                name=fname,
                field_type=finfo.annotation,
                description=finfo.description,
            )
        )

    return TEMPLATE.format(
        model_name=schema.__repr_name__,
        description=schema,
        fields="\n".join(field_desc),
    )
