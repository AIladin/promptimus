from promptimus.llms.claude import (
    Claude,
    _force_no_additional_properties,
    _image_source,
)


def make_client() -> Claude:
    return Claude(model_name="claude-opus-4-8", api_key="dummy")


def test_image_source_data_url():
    assert _image_source("data:image/png;base64,QUJD") == {
        "type": "base64",
        "media_type": "image/png",
        "data": "QUJD",
    }


def test_build_kwargs_translates_tools():
    client = make_client()
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "adds numbers",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    result = client._build_kwargs({"tools": openai_tools})
    assert result["tools"] == [
        {
            "name": "add",
            "description": "adds numbers",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_build_kwargs_translates_response_format():
    client = make_client()
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "Answer",
            "schema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        },
    }
    result = client._build_kwargs({"response_format": response_format})
    assert "response_format" not in result
    assert result["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
        }
    }


def test_force_no_additional_properties_nested():
    schema = {
        "type": "object",
        "properties": {
            "nested": {"type": "object", "properties": {"x": {"type": "integer"}}}
        },
        "$defs": {"Item": {"type": "object", "properties": {"y": {"type": "string"}}}},
    }
    result = _force_no_additional_properties(schema)
    assert result["additionalProperties"] is False
    assert result["properties"]["nested"]["additionalProperties"] is False
    assert result["$defs"]["Item"]["additionalProperties"] is False
