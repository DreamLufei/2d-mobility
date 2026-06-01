from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

from .templates import build_escalation_body, build_escalation_subject


class EmailNotificationBackend:
    def __init__(self):
        self.smtp_host = os.environ.get("SMTP_HOST", "").strip()
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587") or "587")
        self.smtp_username = os.environ.get("SMTP_USERNAME", "").strip()
        self.smtp_password = os.environ.get("SMTP_PASSWORD", "").strip()
        self.smtp_from = os.environ.get("SMTP_FROM", "").strip() or self.smtp_username or "mobility-agent@localhost"
        self.smtp_use_tls = os.environ.get("SMTP_USE_TLS", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.recipients = [item.strip() for item in os.environ.get("EMAIL_NOTIFY_TO", "").split(",") if item.strip()]
        self.dry_run = os.environ.get("EMAIL_DRY_RUN", "true").strip().lower() in {"1", "true", "yes", "on"}

    def send_payload(self, payload: dict[str, object]) -> dict[str, object]:
        subject = build_escalation_subject(payload)
        body = build_escalation_body(payload)
        return self.send(subject=subject, body=body)

    def send(self, *, subject: str, body: str) -> dict[str, object]:
        if not self.smtp_host:
            return {"enabled": False, "sent": False, "reason": "smtp_host_missing"}
        if not self.recipients:
            return {"enabled": False, "sent": False, "reason": "email_recipient_missing"}
        if self.dry_run:
            return {"enabled": True, "sent": False, "reason": "dry_run", "recipients": self.recipients}
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.smtp_from
        message["To"] = ", ".join(self.recipients)
        message.set_content(body)
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20) as server:
                if self.smtp_use_tls:
                    server.starttls()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                server.send_message(message)
            return {"enabled": True, "sent": True, "recipients": self.recipients}
        except Exception as exc:
            return {"enabled": True, "sent": False, "reason": f"email_send_failed:{type(exc).__name__}"}

