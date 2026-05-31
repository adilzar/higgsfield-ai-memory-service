import pytest

from src.ingestion.extraction import (
    ExtractedMemory,
    ExtractionError,
    extract_memories,
    parse_extraction_response,
)


def test_parse_response_accepts_json_markdown_fence():
    memories = parse_extraction_response("""```json
[{"type":"fact","key":"location","value":"Lives in Berlin","confidence":0.9}]
```""")

    assert memories == [
        ExtractedMemory(type="fact", key="location", value="Lives in Berlin", confidence=0.9)
    ]


def test_parse_response_filters_malformed_memory_items():
    memories = parse_extraction_response("""[
            {"type":"fact","key":"employment","value":"Works at Notion","confidence":0.9},
            {"type":"fact","key":"location"},
            {"type":"fact","key":"age","value":42},
            "not an object"
        ]""")

    assert memories == [
        ExtractedMemory(type="fact", key="employment", value="Works at Notion", confidence=0.9)
    ]


def test_parse_response_rejects_non_array_json():
    with pytest.raises(ExtractionError, match="Expected JSON array"):
        parse_extraction_response('{"type":"fact","value":"Works at Notion"}')


def test_parse_response_normalizes_optional_fields():
    memories = parse_extraction_response("""[
            {
                "type":"profile",
                "key":17,
                "value":"  Lives in Berlin  ",
                "confidence":1.7,
                "supersedes_key":"  location  "
            }
        ]""")

    assert memories == [
        ExtractedMemory(
            type="fact",
            key="unknown",
            value="Lives in Berlin",
            confidence=1.0,
            supersedes_key="location",
        )
    ]


def test_extract_memories_returns_raw_response(monkeypatch):
    monkeypatch.setattr(
        "src.ingestion.extraction._call_llm",
        lambda prompt: '[{"type":"preference","key":"style","value":"Prefers concise replies"}]',
    )

    result = extract_memories("user: Keep it short.", [])

    assert result.raw_response == (
        '[{"type":"preference","key":"style","value":"Prefers concise replies"}]'
    )
    assert result.memories == [
        ExtractedMemory(
            type="preference",
            key="style",
            value="Prefers concise replies",
            confidence=1.0,
        )
    ]
