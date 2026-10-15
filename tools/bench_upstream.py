# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Setup's --bench --original-strata TAG: upstream Strata's release TAG beside XeStrata, on the same model files.

The source comes with git into <data>/strata-upstream (one clone; another tag is fetched into it and checked out), and
the engine the way that tag's own setup.py gets it, with that setup.py's own functions where it has them:

- NVIDIA: the release's ready-made engine (download and verify_engine_archive: the SHA-256 of GitHub's API), and
  NVIDIA's cuBLAS and CUDA runtime wheels its setup.py names, installed with pip into a folder of their own (not into
  the Python that runs setup).
- AMD: compiled with HIP by its build_engine_hip, with the system's ROCm (its setup.py would otherwise install ROCm into
  the Python that runs it).
- Intel: its sycl/ port compiled and run in its Docker image (docs/INTEL.md), JIT code (an AOT target would have to be
  chosen by the device's ID); the data folder is mounted at /work.  The arguments get what its sycl/setup_intel.py
  writes for the card (sycl_setup), as its setup does.

Run as a script, `bench_upstream.py --sycl-setup TREE SELECTOR -- ARGS...` prints what sycl_setup makes of ARGS for
upstream's sycl/ port in TREE on the Level Zero device SELECTOR, for a wrapper that runs that engine without the
port's own sycl/serve/strata-sycl.sh (the release's up-sycl.sh): the environment (K=V, the wrapper's defaults
included), then `--`, then the arguments, each ended by a NUL byte.

Upstream's prebuilt CUDA engine and its oneAPI image are not free software: a free build of XeStrata does not fetch
them (AGENTS.md); the AMD comparison is free.  Nothing here is bundled: it is fetched when the user asks for it.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_core import Engine  # noqa: E402

REPO_URL = "https://github.com/Niko1221/Strata.git"
CLONE = "strata-upstream"            # a folder of the data folder: the Intel image mounts the data folder at /work
DOCKER_IMAGE = "strata-sycl-dev"


class UpstreamError(Exception):
    pass


def git(*a, cwd: Path | None = None) -> str:
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise UpstreamError(f"git {' '.join(a)}: {(r.stderr or r.stdout).strip()[-400:]}")
    return r.stdout


def fetch(tag: str, data: Path, say: Callable) -> Path:
    """The tag's source in <data>/strata-upstream, cloned the first time and fetched into the same clone after."""
    if shutil.which("git") is None:
        raise UpstreamError("git is not installed: install it (Ubuntu/Debian: sudo apt install git; Fedora: sudo dnf "
                            "install git) and run this again")
    src = data / CLONE
    if not (src / ".git").is_dir():
        say(f"  Cloning upstream Strata {tag} into {src} ...")
        git("clone", "--quiet", "--depth", "1", "--branch", tag, REPO_URL, str(src))
    else:
        say(f"  Fetching upstream Strata {tag} into {src} ...")
        git("fetch", "--quiet", "--depth", "1", "origin", "tag", tag, cwd=src)
        git("checkout", "--quiet", "--force", tag, cwd=src)
    return src


def load_setup(src: Path):
    """The tag's setup.py as a module (its ROOT is the clone; it only defines things when imported)."""
    spec = importlib.util.spec_from_file_location("strata_upstream_setup", src / "setup.py")
    if spec is None or spec.loader is None:
        raise UpstreamError(f"{src}/setup.py cannot be read")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except (Exception, SystemExit) as e:
        raise UpstreamError(f"the tag's setup.py does not load here ({e})") from e
    return mod


def call(mod, name: str, *a, **k):
    """One of the tag's setup.py functions; its fail() (SystemExit) and a missing function become an UpstreamError."""
    fn = getattr(mod, name, None)
    if fn is None:
        raise UpstreamError(f"the tag's setup.py has no {name}(): this tag is not supported by --original-strata")
    try:
        return fn(*a, **k)
    except SystemExit as e:
        raise UpstreamError(f"the tag's setup.py stopped in {name}() ({e})") from e


def constant(src: Path, name: str):
    """A constant of the tag's setup.py, the Linux branch of `X if WIN else Y`."""
    tree = ast.parse((src / "setup.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            v = node.value.orelse if isinstance(node.value, ast.IfExp) else node.value
            return ast.literal_eval(v)
    raise UpstreamError(f"the tag's setup.py has no {name}")


class SyclSetup:
    """What the tag's sycl/setup_intel.py writes for an Intel card, as a change to the arguments the bench gives:
    its to_sycl() (the experts streamed with the pinned mirror, the VRAM reserve, KV streaming from 64K when the RAM
    holds the KV, 4,096-token prompt chunks, the xe cards' expert profile in place of the shipped one), with the
    options its install() adds for a small card, and the environment its config carries.  The card is the one
    `selector` (level_zero:N) takes; its install() counts the Level Zero devices in the sysfs order of its
    intel_gpus().  The paths stay as they are given (the caller's are already the ones the engine sees)."""

    def __init__(self, src: Path, selector: str | None):
        path = src / "sycl" / "setup_intel.py"
        if not path.is_file():
            raise UpstreamError(f"{path} is missing: this tag has no setup for its sycl/ port")
        saved_path, saved_setup = list(sys.path), sys.modules.get("setup")
        sys.modules["setup"] = load_setup(src)          # setup_intel.py imports its tree's setup.py as `setup`
        try:
            spec = importlib.util.spec_from_file_location("strata_upstream_setup_intel", path)
            if spec is None or spec.loader is None:
                raise UpstreamError(f"{path} cannot be read")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except (Exception, SystemExit) as e:
            raise UpstreamError(f"the tag's sycl/setup_intel.py does not load here ({e})") from e
        finally:
            sys.path[:] = saved_path
            if saved_setup is None:
                sys.modules.pop("setup", None)
            else:
                sys.modules["setup"] = saved_setup
        for name in ("intel_gpus", "to_sycl", "small_card"):
            if not callable(getattr(mod, name, None)):
                raise UpstreamError(f"the tag's sycl/setup_intel.py has no {name}(): this tag is not supported")
        cards = mod.intel_gpus()
        if not cards:
            raise UpstreamError("the tag's sycl/setup_intel.py finds no Intel Arc (an xe or i915 card in "
                                "/sys/class/drm)")
        m = re.fullmatch(r"level_zero:(\d+)", selector or "")
        self.card = cards[int(m.group(1))] if m and int(m.group(1)) < len(cards) else cards[0]
        setattr(mod, "sycl_path", str)                  # the paths stay the caller's
        self.mod, self.ram = mod, mod.S.ram_gb()
        self.wrapper = src / "sycl" / "serve" / "strata-sycl.sh"
        self.env = {k: v for k, v in (self.config([])[1]).items() if not k.startswith("STRATA_SYCL_")}

    def config(self, args: list[str]) -> tuple[list[str], dict]:
        args = list(args)
        card = self.card
        if self.mod.small_card(card["vram_gb"], card["driver"]) and "--draft-vocab" not in args:
            args += ["--draft-vocab", "en"]             # its install()'s option for a small card
        try:                                            # the AOT binary's name: its config then names no other
            cfg = self.mod.to_sycl({"args": args}, self.mod.ROOT / "build-sycl-aot" / "strata", self.ram, {},
                                   card["vram_gb"], card["driver"], card.get("devid", ""))
        except SystemExit as e:
            raise UpstreamError(f"the tag's sycl/setup_intel.py stopped in to_sycl() ({e})") from e
        return cfg["args"], cfg.get("env") or {}

    def args(self, args: list[str]) -> list[str]:
        return self.config(args)[0]

    def wrapper_env(self) -> dict:
        """The run-time switches its sycl/serve/strata-sycl.sh sets (`declare -A setting=([K]=V ...)`) under the
        config's environment, without the ones at 0 or empty, which that wrapper does not pass on."""
        m = re.search(r"declare -A setting=\(([^)]*)\)", self.wrapper.read_text(encoding="utf-8")
                      if self.wrapper.is_file() else "")
        if m is None:
            raise UpstreamError(f"{self.wrapper} sets no run-time switches this reads: this tag is not supported")
        env = {**dict(re.findall(r"\[(\w+)\]=(\S*)", m.group(1))), **self.env}
        return {k: v for k, v in env.items() if v not in ("0", "")}


def nvidia_index(pci: str) -> tuple[int, int]:
    """The GPU's number in PCI order among NVIDIA's (CUDA_DEVICE_ORDER=PCI_BUS_ID) and its compute capability."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id,compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=60).stdout
    rows = sorted((r.split(",")[0].strip().lower()[-12:], r.split(",")[1].strip())
                  for r in out.splitlines() if "," in r)
    for i, (bus, cc) in enumerate(rows):
        if bus == pci.lower()[-12:]:
            return i, int(cc.replace(".", ""))
    raise UpstreamError(f"nvidia-smi does not list the GPU at {pci}")


def nvidia(tag: str, src: Path, data: Path, gpu: dict, say: Callable) -> dict:
    mod = load_setup(src)
    index, cc = nvidia_index(gpu["pci"])
    old = cc < constant(src, "CUDA13_MIN_ARCH")
    asset = constant(src, "CUDA12_ASSET" if old else "PREBUILT_ASSET")
    wheels = constant(src, "CUDA12_WHEELS" if old else "CUDA_WHEELS")
    eng = data / "strata-upstream-engine" / tag
    if not (eng / "strata").exists():
        base = constant(src, "PREBUILT_TAG_URL").format(version=tag.lstrip("v"))
        eng.mkdir(parents=True, exist_ok=True)
        z = eng / asset
        say(f"  Downloading upstream's ready-made engine {asset} ...")
        call(mod, "download", base + asset, z, f"Strata {tag} engine")
        call(mod, "verify_engine_archive", z, asset, base)     # the SHA-256 GitHub's API publishes for the asset
        with zipfile.ZipFile(z) as f:
            f.extractall(eng)
        z.unlink()
        for x in ("strata", "strata-vision"):
            if (eng / x).exists():
                (eng / x).chmod(0o755)
    if not (eng / "strata").exists():
        raise UpstreamError(f"upstream's {asset} has no strata")
    libs = eng / "cuda-libs"
    if not (libs / ".done").exists():
        say(f"  Installing NVIDIA's CUDA libraries for it ({', '.join(wheels)}) into {libs} ...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                            "--target", str(libs), *wheels], capture_output=True, text=True)
        if r.returncode != 0:
            raise UpstreamError(f"pip could not install {', '.join(wheels)}: {r.stderr.strip()[-400:]}")
        (libs / ".done").write_text("")
    dirs = sorted({str(p.parent) for p in (libs / "nvidia").rglob("*.so*")})
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(index),
           "LD_LIBRARY_PATH": os.pathsep.join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)}
    return {"exe": str(eng / "strata"), "env": env, "how": f"the release's ready-made engine {asset}",
            "cfg_extra": {"lib_dirs": dirs}}


def amd(tag: str, src: Path, data: Path, gpu: dict, say: Callable) -> dict:
    if not Path("/opt/rocm/bin/hipcc").exists():
        raise UpstreamError("upstream's AMD engine is compiled with HIP: it needs the system's ROCm 7 (/opt/rocm with "
                            "hipcc and hipBLAS); xestrata --setup lists what to install")
    if not gpu.get("gfx"):
        raise UpstreamError("the AMD GPU's architecture (gfx...) is not known")
    mod = load_setup(src)
    llama = call(mod, "get_llama_cpp")
    say(f"  Compiling upstream's engine for {gpu['gfx']} with HIP (its setup.py's build_engine_hip; 10-20 minutes "
        "the first time) ...")
    eng = call(mod, "build_engine_hip", {"arch": gpu["gfx"], "name": gpu.get("name")}, llama)
    meta = json.loads((Path(eng) / "BUILD.json").read_text(encoding="utf-8"))
    dirs = meta.get("lib_dirs") or []
    amd_cards = [x for x in gpu.get("_all", []) if x.get("vendor") == "amd"]
    index = next((i for i, x in enumerate(amd_cards) if x.get("pci") == gpu.get("pci")), 0)
    env = {**os.environ, "HIP_VISIBLE_DEVICES": str(index),
           "LD_LIBRARY_PATH": os.pathsep.join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)}
    return {"exe": str(Path(eng) / "strata"), "env": env, "how": f"compiled here with HIP for {gpu['gfx']}",
            "cfg_extra": {"lib_dirs": dirs}}


def intel(tag: str, src: Path, data: Path, gpu: dict, say: Callable) -> dict:
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise UpstreamError("upstream's Intel engine (its sycl/ port) is built and run in Docker, and docker does not "
                            "run here without sudo: install docker.io and add yourself to the docker group (sudo "
                            "usermod -aG docker $USER, then log in again), as upstream's docs/INTEL.md says")
    stamp = src / "build-sycl" / "XESTRATA-TAG"
    if not (src / "build-sycl" / "strata").exists() or (stamp.read_text().strip() if stamp.exists() else "") != tag:
        say("  Building upstream's Docker image strata-sycl-dev (about 6 minutes and 13 GB the first time) ...")
        r = subprocess.run(["docker", "build", "-q", "-t", DOCKER_IMAGE, str(src / "sycl" / "tools")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise UpstreamError(f"docker build: {r.stderr.strip()[-400:]}")
        say("  Compiling upstream's SYCL engine in it (about 3 minutes) ...")
        r = subprocess.run(["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}", "-v", f"{data}:/work",
                            "-e", f"REPO=/work/{CLONE}", "-e", f"BUILD_DIR=/work/{CLONE}/build-sycl", DOCKER_IMAGE,
                            f"cd /work/{CLONE} && bash sycl/tools/build.sh strata"], capture_output=True, text=True)
        if r.returncode != 0 or not (src / "build-sycl" / "strata").exists():
            raise UpstreamError(f"upstream's SYCL build failed: {(r.stdout + r.stderr).strip()[-400:]}")
        stamp.write_text(tag + "\n")
    intel_cards = [x for x in gpu.get("_all", []) if x.get("vendor") == "intel"]
    index = next((i for i, x in enumerate(intel_cards) if x.get("pci") == gpu.get("pci")), 0)
    setup = SyclSetup(src, f"level_zero:{index}")
    env = {**os.environ, **setup.env, "STRATA_SYCL_ROOT": str(data), "STRATA_SYCL_BIN": "build-sycl/strata",
           "STRATA_SYCL_NAME": "xestrata-bench-upstream", "ONEAPI_DEVICE_SELECTOR": f"level_zero:{index}"}
    return {"exe": str(src / "sycl" / "serve" / "strata-sycl.sh"), "env": env, "map": (str(data), "/work"),
            "server": "sycl/serve/server_intel.py", "adapt": setup.args,
            "how": "its sycl/ port, compiled in its Docker image (JIT), with what its sycl/setup_intel.py writes"}


def prepare(tag: str, data: Path, gpus: list[dict], license: str, profile: str, say: Callable) -> Engine:
    """The upstream engine for bench.run (the "up" arm): its command, environment and tree, and how it was had.
    `profile`: XeStrata's expert profile, whose name upstream's tree has its own of."""
    if not gpus:
        raise UpstreamError("no GPU found")
    gpu = {**gpus[0], "_all": gpus}
    vendor = str(gpu.get("vendor"))
    if license == "free" and vendor in ("intel", "nvidia"):
        what = "its oneAPI image" if vendor == "intel" else "the CUDA runtime"
        raise UpstreamError(f"this XeStrata is the free build: upstream's engine for this GPU is not free software "
                            f"({what}), so it is not fetched; a contrib build of XeStrata can compare them")
    src = fetch(tag, data, say)
    how = {"nvidia": nvidia, "amd": amd, "intel": intel}.get(vendor)
    if how is None:
        raise UpstreamError(f"no upstream engine for a {vendor} GPU")
    e = how(tag, src, data, gpu, say)
    return Engine(f"upstream {tag}", "up", [e["exe"]], str(src / "data" / Path(profile).name), env=e["env"],
                  cwd=str(src), root=src, py=sys.executable, server=e.get("server"), map=e.get("map"),
                  cfg_extra=e.get("cfg_extra"), how=e["how"], adapt=e.get("adapt"))


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] != "--sycl-setup" or "--" not in argv or argv.index("--") != 3:
        print("usage: bench_upstream.py --sycl-setup TREE SELECTOR -- ARGS...", file=sys.stderr)
        return 2
    try:
        setup = SyclSetup(Path(argv[1]), argv[2])
        words = [f"{k}={v}" for k, v in sorted(setup.wrapper_env().items())] + ["--"] + setup.args(argv[4:])
    except UpstreamError as e:
        print(f"bench_upstream.py: {e}", file=sys.stderr)
        return 1
    sys.stdout.write("".join(w + "\0" for w in words))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
