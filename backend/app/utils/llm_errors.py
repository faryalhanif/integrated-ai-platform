import re

# Matches the telltale signatures of a provider-side rate-limit / quota
# error across the SDKs this platform might use (Gemini's
# RESOURCE_EXHAUSTED, generic HTTP 429s, "rate limit" wording from other
# providers) without hardcoding any one provider's exact error shape.
_RATE_LIMIT_PATTERN = re.compile(
    r"RESOURCE_EXHAUSTED|rate.?limit|\b429\b|quota", re.IGNORECASE
)


def is_llm_rate_limited(exc: BaseException) -> bool:
    """
    True if this exception looks like a provider-side rate-limit/quota
    error (e.g. Gemini free-tier's 20-requests/day cap) rather than a
    genuine application bug.
    """

    return bool(_RATE_LIMIT_PATTERN.search(str(exc)))


def friendly_llm_error_message(exc: BaseException) -> str:
    """
    Client-facing message for an LLM call failure. Never echoes the raw
    exception text back to the caller -- provider errors can include
    internal account/project details (quota IDs, billing links, etc.)
    that shouldn't be shown to end users. Full detail should still be
    logged server-side by the caller (e.g. logger.exception(...)) before
    this is used.
    """

    exc_text = str(exc)

    if is_llm_rate_limited(exc):
        return (
            "The AI service is temporarily rate-limited (the connected "
            "account has hit its request quota). This isn't an error in "
            "your data or prompt -- please wait a few minutes and try "
            "again."
        )

    if (
        "PERMISSION_DENIED" in exc_text
        or "denied access" in exc_text.lower()
        or "not available to new users" in exc_text.lower()
        or "NOT_FOUND" in exc_text
    ):
        return (
            "The configured AI model or API key does not have access to the "
            "selected provider. Please check your .env settings and use a "
            "supported model or valid API key."
        )

    return "Something went wrong while processing this request. Please try again."