# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""tools/setup_choices.py - Strata-main's setup Step 2 ("your choices") for this Windows port.

SETUP.bat asks this before anything is downloaded: which model family, which size, which context,
which KV precision - the same four questions in the same order as Strata-main's setup.py.  The
recommendations come from this PC's VRAM and RAM, measured the same way as the engine sees them
(sysman), and the same numbers this port measured on the B70: the KV is ~23 KiB/token at int8 and
~12.5 KiB/token at q4_0, the MTP draft layer costs ~1.1 GiB of VRAM, the head + verify windows +
driver about 1.5 GiB, and an expert-cache slot ~1.9 MiB.  A context only fits when it leaves at
least 8 GiB for the expert cache (the B70 runs 22 GiB; below that the hit rate and decode fall
off).  The choice goes to settings.json, which setup_models.py and setup_config.py both read.
Images and the experimental speed projection are NOT offered: this port has neither.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from setup_models import CHOICES  # the five sizes this port offers, with GB and RAM numbers

FAMILIES = {
    "qwen": {"title": "Qwen3.8-Flash-Next", "about": "the general model"},
    "coder": {"title": "Qwen3.8-Flash-Next Coder",
              "about": "half the experts (code, tools, images kept); needs ~32 GB of RAM, faster"},
}
CONTEXTS = [32768, 65536, 131072, 262144]   # 262144 is the model's trained length: no RoPE beyond it
KIB_PER_TOK = {"int8": 23.0, "q4_0": 12.5}
MTP_VRAM_GIB = 1.1
FIXED_VRAM_GIB = 1.5
MIN_EXPERT_CACHE_GIB = 8.0
SLOT_BYTES = 2_000_000


def say(msg=""): print(msg, flush=True)
def ok(msg): say(f"  [ok] {msg}")


def vram_gib() -> float:
    """Total dedicated VRAM via the engine's own sysman view (WMI AdapterRAM caps at 4 GB);
    999 = unreadable (no sysman yet): the table then shows no VRAM cap."""
    try:
        sys.path.insert(0, str(ROOT / "serve"))
        import telemetry
        g = telemetry._gpu()
        _, total = g._mem()
        if total:
            return total / 2**30
    except Exception:
        pass
    return 999.0


def ram_gib() -> float:
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX(); st.dwLength = ctypes.sizeof(st)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullTotalPhys / 2**30
    except Exception:
        return 0.0


def cache_left(vram, ctx, kv, mtp):
    kv_gib = ctx * KIB_PER_TOK[kv] / 2**20
    return max(0.0, vram - FIXED_VRAM_GIB - (MTP_VRAM_GIB if mtp else 0.0) - kv_gib)


def ask(question, default, yes):
    if yes:
        say(f"  {question} [{default}] (with --yes)")
        return str(default)
    return (input(f"  {question} [{default}]: ").strip() or str(default))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yes", action="store_true", help="no questions: take every recommendation")
    ap.add_argument("--show", action="store_true", help="print the four tables, write nothing")
    a = ap.parse_args()

    vram, ram = vram_gib(), ram_gib()
    say("=== your choices ===")
    say(f"  this PC: VRAM {vram:.0f} GiB, RAM {ram:.0f} GB")

    # ---- which model? (the family) -----------------------------------------------
    say()
    fams = list(FAMILIES)
    for i, f in enumerate(fams, 1):
        d = FAMILIES[f]
        say(f"  {i}) {d['title']:28s} {d['about']}")
    fam_pick = ask("Which model?", 1, a.yes)
    family = fams[int(fam_pick) - 1]

    # ---- which size? (the quant of that family) ----------------------------------
    say()
    sizes = [c for c in CHOICES if c["family"] == family]
    for i, c in enumerate(sizes, 1):
        fit = "" if c["ram_gb"] <= ram else f"   <- needs ~{c['ram_gb']} GB RAM, you have {ram:.0f}"
        v = "   (verified on this port)" if c["verified"] else "   (untested on this port)"
        say(f"  {i}) {c['model']:8s} {c['about']}; download {c['gb']:.0f} GB, uses ~{c['ram_gb']} GB of RAM{v}{fit}")
    rec_size = next((i for i, c in enumerate(sizes, 1) if c["verified"] and c["ram_gb"] <= ram), 1)
    size_pick = ask("Which size?", rec_size, a.yes)
    size = sizes[int(size_pick) - 1]
    if not size["verified"]:
        say(f"  [!]  {size['model']} is UNTESTED on this Windows port: the kernels are built, but nobody has run "
            "them here. IQ3_S is the verified choice.")
    ok(f"model: {family}/{size['model']}")

    # ---- context? -----------------------------------------------------------------
    say()
    say("  Context length = how much text the model can see at once (your chat, files, tool output).")
    say("  Longer needs more VRAM for it, so fewer experts fit on the GPU:")
    # the fit line is 8 GiB of expert cache; walk contexts biggest-first and try 8-bit KV before
    # 4-bit, so the recommendation is the biggest context that fits at either precision (a 12 GB
    # card lands on 64K with 4-bit KV, not 32K with 8-bit).
    rec_ctx, rec_kv = None, "int8"
    for c in reversed(CONTEXTS):
        for kv_try in ("int8", "q4_0"):
            if cache_left(vram, c, kv_try, True) >= MIN_EXPERT_CACHE_GIB:
                rec_ctx, rec_kv = c, kv_try
                break
        if rec_ctx is not None:
            break
    if rec_ctx is None:
        rec_ctx, rec_kv = CONTEXTS[0], "int8"
    for i, c in enumerate(CONTEXTS, 1):
        left = cache_left(vram, c, "int8", True)
        note = "   (recommended for your GPU)" if c == rec_ctx else ""
        if rec_kv == "q4_0" and c == rec_ctx:
            note = "   (recommended for your GPU, with 4-bit KV)"
        if left < MIN_EXPERT_CACHE_GIB:
            note += f"   (only ~{left:.1f} GiB left for the expert cache at 8-bit KV)"
        say(f"  {i}) {c // 1024}K tokens{note}")
    ctx_pick = ask("Context?", CONTEXTS.index(rec_ctx) + 1, a.yes)
    ctx = CONTEXTS[int(ctx_pick) - 1]

    # ---- KV cache? ----------------------------------------------------------------
    say()
    say("  KV cache precision (the model's memory of the conversation):")
    say("  1) 8-bit   (recommended: what every published number was measured with)")
    say("  2) 4-bit   half the memory (about 4% faster at 128K), but measurably less precise on long")
    say("             documents; long-context lookups (needle tests) still pass")
    kv_pick = ask("KV cache?", 2 if rec_kv == "q4_0" else 1, a.yes)
    kv = {1: "int8", 2: "q4_0"}[int(kv_pick)]
    ok(f"context: {ctx} tokens, KV {'8-bit' if kv == 'int8' else '4-bit (Hadamard-rotated)'}")

    # ---- tight VRAM: offer to free the MTP draft layer ----------------------------
    mtp = True
    left = cache_left(vram, ctx, kv, mtp)
    if left < 4.0:
        say(f"  [!]  only ~{left:.1f} GiB left for the expert cache at this size - the hit rate drops and decode slows.")
        if vram < 999 and ask("Free the MTP draft layer's 1.1 GiB instead? (draft off: slower decode)", "n", a.yes) == "y":
            mtp = False
            left = cache_left(vram, ctx, kv, mtp)
            ok(f"MTP draft off: expert cache left ~{left:.1f} GiB")

    if a.show:
        return 0
    settings = {"family": family, "model": size["model"], "max_context": ctx, "kv": kv, "mtp": mtp}
    (ROOT / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    say(f"  choices -> {ROOT / 'settings.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
