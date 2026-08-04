from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict, List, Tuple

MAX_TURNS_PER_SESSION = 6  # keep last N (user, assistant) turns


class SessionStore:
    """
    In-memory conversation history, keyed by session_id.

    This is intentionally simple for the MVP (single-process, no
    persistence across restarts). Swap for a Redis/DB-backed store later
    behind the same three methods if sessions need to survive restarts
    or be shared across multiple backend instances.
    """

    def __init__(self):
        self._sessions: Dict[str, Deque[Tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=MAX_TURNS_PER_SESSION)
        )
        self._lock = Lock()

    def get_history_text(self, session_id: str) -> str:

        with self._lock:
            turns = list(self._sessions.get(session_id, []))

        if not turns:
            return "(no previous conversation)"

        lines: List[str] = []

        for user_msg, assistant_msg in turns:
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {assistant_msg}")

        return "\n".join(lines)

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:

        with self._lock:
            self._sessions[session_id].append((user_msg, assistant_msg))


_session_store: SessionStore = None
_lock = Lock()


def get_session_store() -> SessionStore:

    global _session_store

    if _session_store is None:
        with _lock:
            if _session_store is None:
                _session_store = SessionStore()

    return _session_store
