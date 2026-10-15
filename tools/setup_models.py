# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""tools/setup_models.py - fetch and prepare the model files for XeStrata-win (Hugging Face only).

SETUP.bat offers this when the data folder is empty.  It downloads the GGUF shards of the chosen
model from Hugging Face at the pinned revision (resumable, .part + .done markers), checks each shard
is whole (tools/gguf_reader.py, the same test as upstream setup.py's check_shards), builds the pack
(tools/iq_pack.py + strata_tokenizer.py) and fetches the MTP draft layer (tools/mtp_fetch.py,
mtp_pack.py, mtp_rt.py).  The choice is recorded in <data>/install.json for tools/setup_config.py.

Windows-port note: only IQ3_S (qwen) is verified on this port; the other quants below have the
kernels in the engine (src/kernels/xe/iq_*.cpp) but were never run here, so they are offered with a
warning.  ModelScope mirrors are NOT used by this tool (Hugging Face only, by the port's choice).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

# pinned revisions (upstream setup.py HF_REVISIONS, 2026-10-14)
REV_QWEN = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
REV_CODER = "5348543e0147355ac9cbcb031184a3546350988e"

# the five choices this port offers: family, quant, repo, subfolder, shard name, shards, GB, RAM GB, verified
CHOICES = [
    {"model": "IQ3_S", "family": "qwen", "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
     "rev": REV_QWEN, "sub": "IQ3_S/", "shard": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{i}-of-00002.gguf",
     "shards": 2, "gb": 83.6, "ram_gb": 62, "verified": True,
     "about": "3.5-bit, the best quality, the slowest; needs a 64 GB PC with little else running"},
    {"model": "IQ3_XXS", "family": "qwen", "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
     "rev": REV_QWEN, "sub": "IQ3_XXS/", "shard": "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{i}-of-00002.gguf",
     "shards": 2, "gb": 75.8, "ram_gb": 60, "verified": False,
     "about": "3-bit, better quality, slower (more CPU work per token)"},
    {"model": "IQ2_XS", "family": "qwen", "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
     "rev": REV_QWEN, "sub": "IQ2_XS/", "shard": "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{i}-of-00002.gguf",
     "shards": 2, "gb": 68.0, "ram_gb": 48, "verified": False,
     "about": "2-bit i-quant, a little better quality, close in speed"},
    {"model": "Q2_0", "family": "qwen", "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
     "rev": REV_QWEN, "sub": "Q2_0/", "shard": "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{i}-of-00002.gguf",
     "shards": 2, "gb": 66.4, "ram_gb": 48, "verified": False,
     "about": "2-bit, the fastest"},
    {"model": "IQ1_M", "family": "coder", "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
     "rev": REV_CODER, "sub": "IQ1_M/", "shard": "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-0000{i}-of-00002.gguf",
     "shards": 2, "gb": 58.4, "ram_gb": 32, "verified": False,
     "about": "the Coder: half the experts (code, tools, images kept); needs ~32 GB of RAM, faster"},
]
PACK_TAG = {"qwen": "", "coder": "coder_"}
# the pack folder names this port uses (IQ3_S keeps the name the shipped template already points at)
PACK_NAME = {("qwen", "IQ3_S"): "iq3_s43"}


def pack_name(c) -> str:
    return PACK_NAME.get((c["family"], c["model"]), PACK_TAG[c["family"]] + c["model"].lower())


def say(msg): print(msg, flush=True)
def ok(msg): print(f"  [ok] {msg}", flush=True)


def ram_gb_total() -> float:
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


def pick_choice(args) -> dict:
    ram = ram_gb_total()
    fits = [c for c in CHOICES if c["ram_gb"] <= ram]
    # setup_choices.py asked the family/size already (settings.json): no second question.
    settings = ROOT / "settings.json"
    if settings.exists():
        s = json.loads(settings.read_text(encoding="utf-8"))
        for c in CHOICES:
            if c["model"] == s.get("model") and c["family"] == s.get("family"):
                if c["ram_gb"] > ram:
                    say(f"  WARNING: {c['model']} wants ~{c['ram_gb']} GB of RAM, this PC has {ram:.0f} GB.")
                say(f"  chosen (from settings.json): {c['family']}/{c['model']}")
                return c
    if args.model:
        for c in CHOICES:
            if c["model"] == args.model and (not args.family or c["family"] == args.family):
                if c["ram_gb"] > ram:
                    say(f"  WARNING: {c['model']} wants ~{c['ram_gb']} GB of RAM, this PC has {ram:.0f} GB.")
                return c
        sys.exit(f"unknown model {args.model} (choose: {', '.join(c['model'] for c in CHOICES)})")
    say(f"  this PC: {ram:.0f} GB of RAM")
    say("  models this Windows port offers (IQ3_S is the only one verified here; the rest have the")
    say("  kernels but are UNTESTED on this port):")
    for i, c in enumerate(CHOICES, 1):
        fit = "fits" if c["ram_gb"] <= ram else f"NEEDS ~{c['ram_gb']} GB RAM"
        v = "verified" if c["verified"] else "untested"
        say(f"   {i}) {c['model']:8s} ({c['family']}): {c['about']}")
        say(f"      {c['gb']} GB download, {v}, {fit}")
    rec = next((c for c in reversed(fits) if c["verified"]), fits[0] if fits else CHOICES[0])
    dflt = CHOICES.index(rec) + 1
    if args.yes:
        say(f"  --yes: choosing {rec['model']}")
        return rec
    q = input(f"  which model? [Enter = {rec['model']}]: ").strip()
    if not q:
        return rec
    try:
        return CHOICES[int(q) - 1]
    except ValueError:
        sys.exit("not a number in the list")


def download(url: str, dst: Path, what: str) -> None:
    """Resumable download: .part with Range, a .done marker with the size, retried on stalls."""
    done = dst.with_name(dst.name + ".done")
    if dst.exists() and done.exists():
        ok(f"{what}: already downloaded")
        return
    part = dst.with_name(dst.name + ".part")
    while True:
        pos = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url)
        if pos:
            req.add_header("Range", f"bytes={pos}-")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                if pos and r.status != 206:      # server restarted the body: start over
                    pos = 0
                total = (r.headers.get("Content-Length") and int(r.headers["Content-Length"]) + pos) or None
                mode = "ab" if pos else "wb"
                with open(part, mode) as f:
                    got, t0 = pos, time.time()
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        got += len(chunk)
                        if got % (1 << 28) < (1 << 20) or got == pos + len(chunk):
                            rate = (got - pos) / max(1e-9, time.time() - t0) / 2**20
                            pct = f" {100 * got / total:.1f}%" if total else ""
                            say(f"  {what}: {got / 2**30:.1f} GB{pct} ({rate:.0f} MB/s)")
            if total and part.stat().st_size != total:
                raise urllib.error.URLError(f"server says {total}, got {part.stat().st_size}")
            break
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            say(f"  {what}: interrupted ({e}); retrying in 10 s (resumes at {part.stat().st_size if part.exists() else 0:,} bytes) ...")
            time.sleep(10)
    part.rename(dst)
    done.write_text(str(dst.stat().st_size), encoding="ascii")
    ok(f"{what}: downloaded ({dst.stat().st_size / 2**30:.1f} GB)")


def check_whole(shards: list[Path]) -> None:
    """Every shard as long as its own tensor directory says (upstream check_shards' test)."""
    sys.path.insert(0, str(TOOLS))
    from gguf_reader import GGUFFile
    for s in shards:
        try:
            g = GGUFFile(s)
        except (ValueError, OSError) as e:
            sys.exit(f"{s.name} is not a whole GGUF shard ({e}) - delete it and run SETUP.bat again")
        need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
        if s.stat().st_size < need:
            sys.exit(f"{s.name} is short: {s.stat().st_size:,} of {need:,} bytes - delete it and run SETUP.bat again")
    ok("all shards whole")


def run(tool: str, *a: str) -> None:
    say(f"  running tools/{tool} ...")
    r = subprocess.run([sys.executable, str(TOOLS / tool), *a])
    if r.returncode:
        sys.exit(f"tools/{tool} failed (exit {r.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="the folder that will hold packs/, models/, mtp/")
    ap.add_argument("--model", help="IQ3_S | IQ3_XXS | IQ2_XS | Q2_0 | IQ1_M")
    ap.add_argument("--family", help="qwen | coder (with --model)")
    ap.add_argument("--yes", action="store_true", help="no questions: the RAM-fitting recommendation")
    ap.add_argument("--check-only", action="store_true", help="only show the choice and sizes, download nothing")
    a = ap.parse_args()

    data = Path(a.data_dir).resolve()
    c = pick_choice(a)
    say(f"  chosen: {c['family']}/{c['model']} ({c['gb']} GB of shards + ~5 GB MTP)")
    if a.check_only:
        return 0
    if not c["verified"]:
        say("  NOTE: this quant is UNTESTED on this Windows port; if the engine refuses it, delete")
        say(f"  models/packs for it and run SETUP.bat again with IQ3_S.")
        if not a.yes:
            if input("  continue? [y/N]: ").strip().lower() != "y":
                sys.exit("cancelled - nothing was written")

    (data / "models").mkdir(parents=True, exist_ok=True)
    mdir = data / "models" / c["model"]      # models\IQ3_S etc. (the shipped template's layout)
    shards = [mdir / c["shard"].format(i=i) for i in range(1, c["shards"] + 1)]
    for s in shards:
        url = f"https://huggingface.co/{c['repo']}/resolve/{c['rev']}/{c['sub']}{s.name}"
        download(url, s, s.name)
    check_whole(shards)

    # the pack (native experts: seconds-to-minutes, reads the shards beside shard 1)
    pack = data / "packs" / pack_name(c)
    if not (pack / "native_experts.txt").exists():
        run("iq_pack.py", "--gguf", str(shards[0]), "--out", str(pack))
    if not (pack / "tokenizer" / "vocab.json").exists():
        run("strata_tokenizer.py", "--gguf", str(shards[0]), "--out", str(pack))
    ok(f"pack ready: {pack}")

    # the MTP draft layer (~5 GB from the original Qwen checkpoint, then packed)
    mtp = data / "mtp"
    if not (mtp / "rt" / "experts.bin").exists():
        run("mtp_fetch.py", "fetch", "--out", str(mtp))
        run("mtp_pack.py", "--src", str(mtp), "--experts", "q2_0", "--out", str(mtp / "mtp-q2_0.gguf"))
        run("mtp_rt.py", "--gguf", str(mtp / "mtp-q2_0.gguf"), "--out", str(mtp / "rt"))
    ok(f"MTP draft layer ready: {mtp / 'rt'}")

    (data / "install.json").write_text(json.dumps({
        "family": c["family"], "model": c["model"], "pack": pack_name(c),
        "shard1": shards[0].name, "shard2": shards[1].name,
    }, indent=2) + "\n", encoding="utf-8")
    say(f"model files ready. setup_config will use {data / 'install.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
