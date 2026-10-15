# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/test_keepwarm_builtin.py - the in-server keepwarm thread (fork, win): it pings an idle
engine through the normal HTTP path and leaves a busy one alone."""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "serve/test_chat_template.jinja"


class KeepwarmBuiltin(unittest.TestCase):
    def test_idle_ping(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(TEMPLATE))
        httpd = serve(svc, port=0)
        try:
            svc.port = httpd.server_address[1]
            svc.keepwarm_interval_s = 0.3
            svc.keepwarm_idle_grace = 0.0
            before = svc.totals["requests"]
            svc.start_keepwarm()
            time.sleep(1.5)
            self.assertGreater(svc.totals["requests"], before)   # the idle engine got pinged
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_busy_engine_is_left_alone(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(TEMPLATE))
        httpd = serve(svc, port=0)
        try:
            svc.port = httpd.server_address[1]
            svc.keepwarm_interval_s = 0.3
            svc.keepwarm_idle_grace = 600.0        # "busy within 10 min": every ping is skipped
            before = svc.totals["requests"]
            svc.start_keepwarm()
            time.sleep(1.2)
            self.assertEqual(svc.totals["requests"], before)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
