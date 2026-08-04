from typing import Optional
import re


def default_fallback_message(
    business_name: str,
    escalation_contact: Optional[str] = None,
) -> str:
    """
    Generic, business-agnostic fallback message used when a client didn't
    provide their own fallback_message. Built from the profile's own
    fields, never hardcoded per-business.
    """

    message = (
        f"I'm sorry, I don't have that information right now. "
    )

    if escalation_contact:
        message += (
            f"For further help, please reach out to {business_name}'s "
            f"team directly at {escalation_contact}."
        )
    else:
        message += (
            f"Please reach out to {business_name}'s team directly for "
            f"further help."
        )

    return message


def truncate_text(text: str, max_chars: int = 4000) -> str:
    """
    Truncates text to a maximum number of characters, appending a marker
    if truncation happened. Used to keep retrieved chunks from blowing up
    the context window.
    """

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + " ... [truncated]"


def safe_get(dictionary: dict, key: str, default=None):
    """
    Defensive dict getter that never raises.
    """

    if not isinstance(dictionary, dict):
        return default

    return dictionary.get(key, default)


def strip_markdown_emphasis(text: str) -> str:
    """
    Removes common markdown emphasis/list markers from LLM output as a
    safety net, in case the model ignores the plain-text instruction in
    the prompt. Turns "**bold**" -> "bold", "*italic*" -> "italic",
    and "* item" / "- item" bullet lines -> "item" lines, without
    touching the actual sentence content.
    """

    if not text:
        return text

    # **bold** / __bold__ -> bold
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)

    # *italic* / _italic_ -> italic (only when wrapping a word/phrase,
    # not a stray asterisk used as e.g. a multiplication sign)
    text = re.sub(r"(?<!\w)\*(\S(?:[^*\n]*\S)?)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_(\S(?:[^_\n]*\S)?)_(?!\w)", r"\1", text)

    # Leading "* " / "- " bullet markers at the start of a line
    text = re.sub(r"(?m)^[ \t]*[*-][ \t]+", "", text)

    # Markdown headers "# Heading" -> "Heading"
    text = re.sub(r"(?m)^#{1,6}[ \t]+", "", text)

    return text.strip()