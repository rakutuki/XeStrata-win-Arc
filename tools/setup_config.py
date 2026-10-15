# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""tools/setup_config.py - make the serve config from strata-xe-template.json.

SETUP.bat calls this after checking the PC.  %TREE% becomes the XeStrata folder, %DATA% the folder holding
packs/, models/ and mtp/ (the model files themselves are not distributed).  The engine's verify path needs a
profile-filled VRAM expert tier: without --expert-profile it exits right after "session is up".  The tree
ships a generic data\expert-profile.bin and this tool adds it to args when present; the server keeps
learning into data\expert-profile-learned.bin (expert_profile_save) and takes over from that once it exists.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="the folder with packs/, models/, mtp/ (~70 GB of model files)")
    ap.add_argument("--out", default=str(ROOT / "strata-xe-iq3_s.json"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing config")
    a = ap.parse_args()

    data = Path(a.data_dir).resolve()
    out = Path(a.out).resolve()
    if out.exists() and not a.force:
        print(f"config exists: {out} (use --force to overwrite)")
        return 1

    tpl = json.loads((ROOT / "strata-xe-template.json").read_text(encoding="utf-8"))
    tree = str(ROOT)

    # setup_models.py records the chosen model in <data>/install.json; the template's IQ3_S paths
    # become that model's files (the pack folder, the two GGUF shards, the model name).
    inst = data / "install.json"
    if inst.exists():
        info = json.loads(inst.read_text(encoding="utf-8"))
        q = info["model"]
        args = tpl["args"]
        def set_arg(flag, value):
            args[args.index(flag) + 1] = value
        pack = data / "packs" / info.get("pack", q.lower())
        set_arg("--pack", str(pack))
        set_arg("--native", str(data / "models" / q / info["shard1"]))
        set_arg("--ple-gguf", str(data / "models" / q / info["shard2"]))
        tpl["tokenizer"] = str(pack / "tokenizer")
        tpl["model_name"] = "xe-strata-" + q.lower().replace("_", "-")

    # the verify path needs a profile-filled expert tier: use the profile that ships in the tree
    # (a fresh install has none of its own yet; the learned one takes over once it exists).
    prof = ROOT / "data" / "expert-profile.bin"
    if prof.is_file() and "--expert-profile" not in tpl["args"]:
        a = tpl["args"]
        i = a.index("--expert-cache")
        a[i:i] = ["--expert-profile", str(prof)]

    def fill(v):
        if isinstance(v, str):
            return v.replace("%TREE%", tree).replace("%DATA%", str(data))
        if isinstance(v, list):
            return [fill(x) for x in v]
        if isinstance(v, dict):
            return {k: fill(x) for k, x in v.items()}
        return v

    # setup_sizing.py records the context/KV choice for this PC in settings.json (next to this
    # tool); apply it to the template before filling paths.  Without it the template's own
    # 262k/int8 stands.
    settings = ROOT / "settings.json"
    if settings.exists():
        s = json.loads(settings.read_text(encoding="utf-8"))
        a = tpl["args"]
        if "max_context" in s:
            a[a.index("--max-context") + 1] = str(s["max_context"])
        if "kv" in s:
            a[a.index("--kv") + 1] = s["kv"]
        if s.get("mtp") is False and "--mtp" in a:
            i = a.index("--mtp")
            del a[i:i + 2]

    cfg = fill(tpl)

    missing = []
    pack = cfg["args"][cfg["args"].index("--pack") + 1]
    for key in ("--pack", "--mtp"):
        p = Path(cfg["args"][cfg["args"].index(key) + 1])
        if not p.is_dir():
            missing.append(f"{key}: {p}")
    for key in ("--native", "--ple-gguf"):
        p = Path(cfg["args"][cfg["args"].index(key) + 1])
        if not p.is_file():
            missing.append(f"{key}: {p}")
    tok = Path(cfg["tokenizer"])
    if not tok.is_dir():
        missing.append(f"tokenizer: {tok}")
    if missing:
        print("the model files are not in place yet:")
        for m in missing:
            print("  " + m)
        print("put packs/, models/ and mtp/ under the data folder (README: 'Model files'), then run SETUP.bat again.")
        return 1

    (ROOT / "data").mkdir(exist_ok=True)
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
    print(f"config written: {out}")
    print(f"  pack: {pack}")
    print(f"  ctx:  {cfg['args'][cfg['args'].index('--max-context') + 1]} tokens, kv {cfg['args'][cfg['args'].index('--kv') + 1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
