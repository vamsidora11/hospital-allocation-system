"""Shared chat-LLM access point for the router and agents.

Provider priority (see app/config.py): Google AI Studio > Azure OpenAI > `None`. Callers
(router, agents) must have a deterministic rule-based fallback for the `None` case; there is
no local chat-model substitute here (unlike embeddings, a small offline model can't
meaningfully replace structured LLM reasoning, so the fallback lives in each caller's domain
logic instead).

Google calls go through a thin direct-REST wrapper (`GoogleAIChatClient`, below) rather than
`langchain_google_genai` — that SDK's client hung indefinitely in this sandbox (see
app/rag/embeddings.py and PROGRESS.md Phase 3) even though plain HTTPS calls succeed
in under a second. `with_structured_output(...).invoke(prompt)` mirrors the langchain
interface so callers (app/graph/nodes/router.py) don't need to know which provider is active.
"""

import json
from typing import Any

from pydantic import BaseModel

from app.config import get_settings

_TYPE_MAP = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def _resolve_ref(schema: dict, defs: dict) -> dict:
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return defs[name]
    return schema


def _to_gemini_schema(schema: dict, defs: dict) -> dict:
    schema = _resolve_ref(schema, defs)

    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if _resolve_ref(s, defs).get("type") != "null"]
        return _to_gemini_schema(non_null[0], defs) if non_null else {"type": "STRING", "nullable": True}

    json_type = schema.get("type", "string")
    result: dict[str, Any] = {"type": _TYPE_MAP.get(json_type, "STRING")}

    if "enum" in schema:
        result["enum"] = schema["enum"]
    if "description" in schema:
        result["description"] = schema["description"]
    if json_type == "object":
        properties = schema.get("properties", {})
        result["properties"] = {k: _to_gemini_schema(v, defs) for k, v in properties.items()}
        if "required" in schema:
            result["required"] = schema["required"]
    if json_type == "array":
        result["items"] = _to_gemini_schema(schema.get("items", {}), defs)

    return result


class _StructuredCaller:
    def __init__(self, chat_client: "GoogleAIChatClient", output_model: type[BaseModel]):
        self._chat_client = chat_client
        self._output_model = output_model

    def invoke(self, prompt: str) -> BaseModel:
        import httpx

        json_schema = self._output_model.model_json_schema()
        response_schema = _to_gemini_schema(json_schema, json_schema.get("$defs", {}))

        model = self._chat_client.model
        model = model if model.startswith("models/") else f"models/{model}"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            },
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, params={"key": self._chat_client.api_key}, json=body)
            response.raise_for_status()

        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return self._output_model(**json.loads(text))


class GoogleAIChatClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def with_structured_output(self, output_model: type[BaseModel]) -> _StructuredCaller:
        return _StructuredCaller(self, output_model)


def get_chat_llm() -> Any | None:
    settings = get_settings()

    if settings.google_api_key:
        return GoogleAIChatClient(api_key=settings.google_api_key, model=settings.google_chat_model)

    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        from langchain_openai import AzureChatOpenAI  # optional dep, imported lazily

        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            azure_deployment="gpt-4o",
            temperature=0,
        )

    return None
