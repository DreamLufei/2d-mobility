from __future__ import annotations

from typing import Protocol


class NotificationBackend(Protocol):
    def send(self, *, subject: str, body: str) -> dict[str, object]:
        ...

