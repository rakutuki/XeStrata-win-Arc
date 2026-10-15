# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The Harmony output parser (llm-jp-4.1's chat format), fed one character at a time as a stream arrives."""
import unittest

from serve.frontend import HarmonyParser


def run(text: str, reason: str = "stop", **kw) -> list[tuple]:
    """Every event of `text` fed character by character, then finish(reason); consecutive texts of one kind joined."""
    p = HarmonyParser(**kw)
    events = []
    for ch in text:
        events += p.feed(ch)
    events += p.finish(reason)
    out: list[tuple] = []
    for e in events:
        if e.kind == "tool_call":
            assert e.call is not None
            out.append(("tool_call", e.call.name, e.call.arguments))
        elif out and out[-1][0] == e.kind:
            out[-1] = (e.kind, out[-1][1] + e.text)
        elif e.text:
            out.append((e.kind, e.text))
    return out


class HarmonyOutput(unittest.TestCase):
    def test_reasoning_then_answer(self):
        text = ("<|channel|>analysis<|message|> 2 と 2 を足す。<|end|>"
                "<|start|>assistant<|channel|>final<|message|> 答えは 4 です。<|return|>")
        self.assertEqual(run(text), [("reasoning", "2 と 2 を足す。"), ("content", "答えは 4 です。")])

    def test_tool_call(self):
        text = ("<|channel|>analysis<|message|> 天気を調べる。<|end|>"
                "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>"
                ' {"city": "東京"}<|call|>')
        self.assertEqual(run(text, tools=[{"name": "get_weather"}]),
                         [("reasoning", "天気を調べる。"), ("tool_call", "get_weather", {"city": "東京"})])

    def test_call_ended_by_the_stop_token(self):
        # the server stops at <|call|> without passing it to the parser
        text = '<|channel|>commentary to=functions.f<|message|> {"a": 1}'
        self.assertEqual(run(text, tools=[{"name": "f"}]), [("tool_call", "f", {"a": 1})])

    def test_cut_inside_a_call_is_text(self):
        text = '<|channel|>commentary to=functions.f<|message|> {"a": 1'
        self.assertEqual(run(text, reason="length", tools=[{"name": "f"}]), [("content", '{"a": 1')])


if __name__ == "__main__":
    unittest.main()
