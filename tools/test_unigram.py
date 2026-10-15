# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The SentencePiece Unigram tokenizer (strata_unigram, llm-jp-4.1) against llama.cpp's on the same GGUF.

Needs the model and llama.cpp's tokenizer program, so it runs only with both given and is skipped otherwise:
XESTRATA_LLMJP_GGUF (any shard of llm-jp-4.1) and XESTRATA_LLAMA_TOKENIZE (llama.cpp's llama-tokenize).
    .venv/bin/python -m unittest discover -s tools -p test_unigram.py
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

import strata_tokenizer  # noqa: E402

GGUF = os.environ.get("XESTRATA_LLMJP_GGUF")
LLAMA_TOKENIZE = os.environ.get("XESTRATA_LLAMA_TOKENIZE")

# Japanese and English prose, the normalizer's cases (runs of spaces, line breaks, full-width forms), digits, code,
# characters outside the vocabulary
TEXTS = [
    "こんにちは。今日はいい天気ですね。",
    "The quick brown fox jumps over the lazy dog.",
    "日本語と English が混ざった文。",
    "  leading and   inner   spaces  ",
    "line one\nline two\n\n\tindented",
    "ＡＢＣ１２３　全角の英数字と空白",
    "円周率は3.14159、2026年10月11日。",
    "def f(x):\n    return x ** 2  # 二乗",
    "絵文字 🍣🍜 と記号 ∀x∈ℝ",
    "ｶﾀｶﾅ半角 と ㈱ ① ㍻",
]


def llama_ids(text: str) -> list[int]:
    assert GGUF and LLAMA_TOKENIZE
    out = subprocess.run([LLAMA_TOKENIZE, "-m", GGUF, "-p", text, "--ids", "--no-bos", "--no-escape",
                          "--no-parse-special", "--log-disable"],
                         check=True, capture_output=True, text=True).stdout
    line = [ln for ln in out.splitlines() if ln.startswith("[")][-1]
    return [int(v) for v in line.strip("[]").split(",") if v.strip()]


@unittest.skipUnless(GGUF and LLAMA_TOKENIZE, "set XESTRATA_LLMJP_GGUF and XESTRATA_LLAMA_TOKENIZE")
class UnigramMatchesLlamaCpp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        strata_tokenizer.extract(GGUF, cls.tmp.name)
        cls.tok = strata_tokenizer.Tokenizer.from_dir(Path(cls.tmp.name) / "tokenizer")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_token_ids(self):
        for text in TEXTS:
            with self.subTest(text=text):
                self.assertEqual(self.tok.encode(text), llama_ids(text))


if __name__ == "__main__":
    unittest.main()
