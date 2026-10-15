# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/release_check.py - whether a newer XeStrata release is out: GitHub's latest release of MistVVK/XeStrata
against this copy's version.  Only says so; updating stays the user's (a new deb or rpm, or ./update.sh).

setup.py (--check-update; the system's python3 in a package) and server.py (at the start, every six hours, the web
app's About) both use it, so it imports nothing beyond the standard library.  XESTRATA_UPDATE_CHECK=0 turns the
automatic checks off; an explicit one (--check-update, the web app's button) still runs.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

REPO = "MistVVK/XeStrata"
RELEASES = f"https://github.com/{REPO}/releases"
API = f"https://api.github.com/repos/{REPO}/releases/latest"   # drafts and pre-releases are not "latest"
EVERY_S = 6 * 3600
# GitHub allows 60 unauthenticated API requests an hour per address: a button pressed again soon gets the last answer
FORCE_GAP_S = 60
ROOT = Path(__file__).resolve().parents[1]


def current_version(root: Path = ROOT) -> str:
    """This copy's version without the "xe": a package's (its PACKAGED record), else the one the source tree builds
    (CMakeLists.txt's XESTRATA_VERSION: the upstream version, and the release on it after the first)."""
    try:
        return str(json.loads((root / "PACKAGED").read_text(encoding="utf-8")).get("version", "0"))
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        return "0"
    try:
        text = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    except OSError:
        return "0"
    m = re.search(r'set\(STRATA_UPSTREAM_VERSION "([\d.]+)"\)', text)
    r = re.search(r"set\(XESTRATA_REVISION (\d+)\)", text)
    if not m:
        return "0"
    return m.group(1) + (f".{int(r.group(1)) - 1}" if r and int(r.group(1)) > 1 else "")


def packaged(root: Path = ROOT) -> bool:
    return (root / "PACKAGED").is_file()


def vtuple(v) -> tuple | None:
    """xe0.1.40.2.2 (a tag) or 0.1.40.2.2 -> (0, 1, 40, 2, 2); None for anything else."""
    m = re.fullmatch(r"(?:xe|v)?(\d+(?:\.\d+)*)", str(v or "").strip())
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def how_to_update(root: Path = ROOT) -> str:
    if packaged(root):
        return (f"download the new package from {RELEASES} and install it as before (sudo apt install ./<file>.deb "
                "or sudo dnf install ./<file>.rpm); the models and settings stay")
    return "run ./update.sh in the XeStrata folder"


def fetch_latest(current: str, timeout: float = 10.0) -> dict:
    """The latest release: {"tag", "version", "url", "published"}.  Raises OSError or ValueError."""
    req = urllib.request.Request(API, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": f"xestrata/{current}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        rel = json.loads(r.read())
    tag = str(rel.get("tag_name") or "")
    if vtuple(tag) is None:
        raise ValueError(f"the latest release's tag is not a version ({tag[:40]!r})")
    url = str(rel.get("html_url") or "")
    if not url.startswith("https://github.com/"):        # the web app links it
        url = f"{RELEASES}/tag/{tag}"
    return {"tag": tag, "version": re.sub(r"^(xe|v)", "", tag), "url": url, "published": rel.get("published_at")}


def is_newer(latest: dict | None, current: str) -> bool:
    new, cur = vtuple(latest and latest.get("version")), vtuple(current)
    return bool(new and cur and new > cur)


def enabled() -> bool:
    return os.environ.get("XESTRATA_UPDATE_CHECK", "1").strip() != "0"


def error_text(e: Exception) -> str:
    reason = getattr(e, "reason", None) or e
    return f"GitHub could not be asked ({reason})"


class Checker:
    """The server's answer, kept between checks: asked at most every six hours unless forced."""

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.current = current_version(root)
        self.latest: dict | None = None
        self.checked_at: float | None = None
        self.error: str | None = None
        self.lock = threading.Lock()

    def check(self, force: bool = False) -> dict:
        with self.lock:                                # one request to GitHub at a time; the others wait for it
            age = time.time() - self.checked_at if self.checked_at else None
            if age is None or age >= (FORCE_GAP_S if force else EVERY_S):
                try:
                    self.latest, self.error = fetch_latest(self.current), None
                except (OSError, ValueError) as e:
                    self.error = error_text(e)
                self.checked_at = time.time()
        return self.info()

    def newer(self) -> bool:
        return is_newer(self.latest, self.current)

    def info(self) -> dict:
        newer = self.newer()
        return {"current": self.current, "latest": self.latest and self.latest["version"],
                "tag": self.latest and self.latest["tag"], "url": self.latest and self.latest["url"],
                "published": self.latest and self.latest["published"], "newer": newer,
                "checked_at": int(self.checked_at) if self.checked_at else None, "error": self.error,
                "automatic": enabled(), "how": how_to_update(self.root) if newer else None, "releases": RELEASES}

    def watch(self, say) -> None:
        """At the start and every six hours on a daemon thread: `say` a line once for each newer release found."""
        told = set()

        def loop():
            while True:
                self.check()
                if self.newer() and self.latest["tag"] not in told:
                    told.add(self.latest["tag"])
                    say(f"XeStrata {self.latest['tag']} is out (this is xe{self.current}): {self.latest['url']} - "
                        f"{how_to_update(self.root)}")
                time.sleep(EVERY_S)

        threading.Thread(target=loop, name="release-check", daemon=True).start()
