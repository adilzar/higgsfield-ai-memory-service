from src.ingestion.intake import (
    TurnMessage,
    format_turn_messages,
    message_to_dict,
    parse_turn_timestamp,
)


def test_format_turn_messages_preserves_roles_and_order():
    messages = [
        TurnMessage(role="user", content="I work at Stripe."),
        TurnMessage(role="assistant", content="How long have you been there?"),
        TurnMessage(role="tool", content='{"company":"Stripe"}', name="lookup"),
    ]

    assert format_turn_messages(messages) == (
        "user: I work at Stripe.\n"
        "assistant: How long have you been there?\n"
        'tool: {"company":"Stripe"}'
    )


def test_message_to_dict_preserves_name_key():
    assert message_to_dict(TurnMessage("tool", "{}", "lookup")) == {
        "role": "tool",
        "content": "{}",
        "name": "lookup",
    }


def test_parse_turn_timestamp_accepts_z_suffix():
    parsed = parse_turn_timestamp("2025-03-15T10:30:00Z")

    assert parsed.isoformat() == "2025-03-15T10:30:00+00:00"
