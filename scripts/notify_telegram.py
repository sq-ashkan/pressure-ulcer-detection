"""
Send progress messages and files to Ashkan's phone through the Telegram bot.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from .env (never commit them).

  python scripts/notify_telegram.py --setup            # wait for /start, store chat id
  python scripts/notify_telegram.py --text "Training finished: Dice 0.71"
  python scripts/notify_telegram.py --file results/report.pdf --text "New report"

Also importable: from notify_telegram import send_message, send_document
"""
import argparse
import json
import mimetypes
import os
import re
import sys
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ENV = Path(__file__).resolve().parents[1] / ".env"
API = "https://api.telegram.org/bot{token}/{method}"


def _env():
    vals = {}
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip()
    token = os.getenv("TELEGRAM_BOT_TOKEN") or vals.get("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID") or vals.get("TELEGRAM_CHAT_ID")
    if not token:
        sys.exit("TELEGRAM_BOT_TOKEN missing in .env")
    return token, chat


def _call(method, params=None, timeout=60):
    token, _ = _env()
    data = urllib.parse.urlencode(params or {}).encode()
    with urllib.request.urlopen(API.format(token=token, method=method), data=data, timeout=timeout) as r:
        out = json.load(r)
    if not out.get("ok"):
        raise RuntimeError(out)
    return out["result"]


def send_message(text: str, silent: bool = True) -> None:
    """Progress updates are silent by default (no sound on the phone)."""
    _, chat = _env()
    if not chat:
        sys.exit("TELEGRAM_CHAT_ID missing -- run with --setup first")
    for i in range(0, len(text), 4000):  # Telegram limit is 4096 chars
        _call("sendMessage", {"chat_id": chat, "text": text[i:i + 4000],
                              "disable_notification": "true" if silent else "false"})


def send_document(path: str, caption: str = "", silent: bool = True) -> None:
    token, chat = _env()
    if not chat:
        sys.exit("TELEGRAM_CHAT_ID missing -- run with --setup first")
    p = Path(path)
    boundary = uuid.uuid4().hex
    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    parts = []
    for k, v in (("chat_id", chat), ("caption", caption[:1000]),
                 ("disable_notification", "true" if silent else "false")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="document"; filename="{p.name}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n".encode() + p.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(API.format(token=token, method="sendDocument"), data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        if not json.load(r).get("ok"):
            raise RuntimeError("sendDocument failed")


def setup(wait_seconds: int = 900) -> None:
    """Long-poll until someone sends the bot a private message, then store that chat id."""
    import time
    deadline, offset = time.time() + wait_seconds, None
    print("waiting for a private message to the bot ...", flush=True)
    while time.time() < deadline:
        params = {"timeout": 50}
        if offset:
            params["offset"] = offset
        for u in _call("getUpdates", params, timeout=70):
            offset = u["update_id"] + 1
            chat = (u.get("message") or {}).get("chat") or {}
            if chat.get("type") == "private":
                cid = str(chat["id"])
                text = ENV.read_text()
                if re.search(r"^TELEGRAM_CHAT_ID=", text, re.M):
                    text = re.sub(r"^TELEGRAM_CHAT_ID=.*$", f"TELEGRAM_CHAT_ID={cid}", text, flags=re.M)
                else:
                    text += f"\nTELEGRAM_CHAT_ID={cid}\n"
                ENV.write_text(text)
                who = chat.get("username") or chat.get("first_name")
                print(f"chat id stored for @{who}", flush=True)
                return
    sys.exit("no message received in time")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--text", default="")
    ap.add_argument("--file")
    ap.add_argument("--loud", action="store_true", help="ring the phone (default: silent)")
    a = ap.parse_args()
    if a.setup:
        setup()
    if a.file:
        send_document(a.file, a.text, silent=not a.loud)
    elif a.text:
        send_message(a.text, silent=not a.loud)
