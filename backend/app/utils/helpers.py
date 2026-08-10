from typing import Optional
import re


def extract_html_document(text: str) -> str:
    """
    Defensive cleanup for LLM-generated HTML (website codegen/edit
    agents). Model output should already be a raw HTML document per the
    prompt, but this strips a wrapping ```html fence if one slipped
    through, and trims anything before <!DOCTYPE/<html or after the
    final </html>, so a stray preamble/sign-off never ends up embedded
    in the live preview.
    """

    if not text:
        return text

    cleaned = text.strip()

    # Strip a wrapping ```html ... ``` or ``` ... ``` fence.
    fence_match = re.match(
        r"^```(?:html)?\s*\n(.*)\n```$", cleaned, flags=re.DOTALL
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()

    lower = cleaned.lower()

    start = lower.find("<!doctype")
    if start == -1:
        start = lower.find("<html")

    end = lower.rfind("</html>")

    if start != -1 and end != -1:
        cleaned = cleaned[start : end + len("</html>")]

    return cleaned.strip()


def slugify_business_name(name: str, suffix: str) -> str:
    """
    Turns a business name into a URL-safe slug for the public site
    address (GET /sites/{slug}), e.g. "Nimbus Analytics!" + a project_id
    -> "nimbus-analytics-829a32". The short suffix (first 6 chars of the
    project_id) keeps two businesses with the same name from colliding
    without making the URL any less readable.
    """

    base = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    base = base or "site"
    return f"{base}-{suffix[:6]}"


    """
    Raised when generated HTML is missing its closing </html> — almost
    always means the model hit its output-token limit mid-document
    rather than that it returned genuinely malformed HTML. Callers
    should surface this distinctly (e.g. "try fewer pages/sections, or
    raise WEBSITE_LLM_MAX_TOKENS") instead of silently rendering a
    half-written page, which shows up in a browser as a blank/broken
    preview with no indication of what went wrong.
    """


def assert_complete_html_document(html: str) -> None:
    """
    Validates that `html` (already run through extract_html_document)
    is a plausibly complete document, distinguishing "the model didn't
    return HTML at all" from "the model's HTML got cut off partway
    through" so each can be reported with an actionable message.
    """

    lower = html.lower()

    if "<html" not in lower:
        raise ValueError(
            "Generation did not return an HTML document at all — this "
            "is usually a model/prompt-format issue, not a size issue."
        )

    if "</html>" not in lower:
        raise TruncatedHTMLError(
            "The generated page was cut off before it finished (missing "
            "closing </html>) — this almost always means the site was "
            "too large for the current WEBSITE_LLM_MAX_TOKENS budget. "
            "Try asking for fewer pages/sections, or raise "
            "WEBSITE_LLM_MAX_TOKENS in backend/.env (Gemini 2.5 Flash "
            "supports up to 65536 output tokens)."
        )


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