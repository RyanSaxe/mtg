"""HUD window bridge (H3): stdlib HTTP server exposing panel.html, /state JSON,
and disk-cached Scryfall card art at /art/<card_id>.

The follower/advisor loop runs in the caller's thread and pushes snapshots via
HudServer.update(); the panel polls /state every 500ms.

H5 adds the deck-builder section: `update_deck()` merges a `deck` block into
the same /state payload (one poll loop, per docs/DECKBUILDER_HANDOFF.md), and
POST /lock, /rebuild, /clear_locks drive lock-and-rebuild through an action
handler installed by the app. Handlers run on the HTTP thread and return the
refreshed deck payload, so the panel sees the result on its next poll.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

from draftbot.data.cards import HEADERS, SCRYFALL_CACHE_DIR

ART_DIR = Path("data/raw/scryfall/images")
PANEL = Path(__file__).parent / "panel.html"

IDLE_STATE = {"status": "idle", "status_line": "log connected", "version": "v0.1"}


def art_url_map(set_code: str) -> dict[int, str]:
    """our card id → scryfall art_crop URL, via the cached set JSONs."""
    import pandas as pd

    from draftbot.data.cards import PROCESSED_DIR, name_to_id, norm_name, source_sets
    cards = pd.read_parquet(PROCESSED_DIR / set_code / "cards.parquet")
    nti = name_to_id(cards)
    out: dict[int, str] = {}
    for code in source_sets(set_code):
        cache = SCRYFALL_CACHE_DIR / f"{code.lower()}.json"
        if not cache.exists():
            continue
        for c in json.loads(cache.read_text()):
            vid = nti.get(norm_name(c["name"]))
            uris = c.get("image_uris") or (c.get("card_faces") or [{}])[0].get("image_uris") or {}
            url = uris.get("art_crop") or uris.get("small")
            if vid is not None and url:
                out.setdefault(vid, url)
    return out


class HudServer:
    def __init__(self, port: int = 8787):
        self.port = port
        self._state = dict(IDLE_STATE)
        self._deck: dict | None = None
        self._lock = threading.Lock()
        self._art_urls: dict[int, str] = {}
        self._action = None      # set_action_handler(fn(action, payload))
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence request logging
                pass

            def _send(self, code, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(200, PANEL.read_bytes(), "text/html")
                elif self.path == "/state":
                    self._send(200, json.dumps(outer.state()).encode(),
                               "application/json")
                elif self.path.startswith("/art/"):
                    try:
                        data = outer._art(int(self.path.split("/")[-1]))
                    except (ValueError, KeyError):
                        data = None
                    if data is None:
                        self._send(404, b"", "image/jpeg")
                    else:
                        self._send(200, data, "image/jpeg")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                action = self.path.lstrip("/")
                if action not in ("lock", "rebuild", "clear_locks"):
                    self._send(404, b"not found", "text/plain")
                    return
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    payload = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    payload = {}
                if outer._action is None:
                    self._send(503, b'{"error":"no deck builder"}',
                               "application/json")
                    return
                try:
                    deck = outer._action(action, payload)
                except Exception as exc:   # never take the panel down with us
                    self._send(500, json.dumps({"error": str(exc)}).encode(),
                               "application/json")
                    return
                if deck is not None:
                    outer.update_deck(deck)
                self._send(200, json.dumps(outer.state()).encode(),
                           "application/json")

        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    def set_art_urls(self, urls: dict[int, str]):
        self._art_urls = urls

    def set_action_handler(self, fn):
        """fn(action, payload) -> refreshed deck payload | None."""
        self._action = fn

    def _art(self, card_id: int) -> bytes | None:
        cached = ART_DIR / f"{card_id}.jpg"
        if cached.exists():
            return cached.read_bytes()
        url = self._art_urls.get(card_id)
        if not url:
            return None
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return None
        ART_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(resp.content)
        return resp.content

    def state(self) -> dict:
        """The merged snapshot the panel polls — pick view plus deck section."""
        with self._lock:
            out = dict(self._state)
            if self._deck is not None:
                out["deck"] = self._deck
            return out

    def update(self, state: dict):
        """Replace the pick-view snapshot; the deck section survives."""
        with self._lock:
            self._state = state

    def update_deck(self, deck: dict | None):
        with self._lock:
            self._deck = deck

    def start(self):
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.port}/"

    def stop(self):
        self._httpd.shutdown()
