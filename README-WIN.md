# XeStrata-win ― the Windows build (verified on an Intel Arc B70)

This repository is the **Windows port** of [XeStrata](https://github.com/MistVVK/XeStrata) (based on upstream Strata 0.1.43 / xe0.1.43.1).
For the general picture see [README.md](README.md) and [docs/XE.md](docs/XE.md); this file covers only what is Windows-specific.

## Verified environment

| | |
| --- | --- |
| OS | Windows 11 x64 |
| GPU | **Intel Arc Pro B70 (32 GB, Xe2/Battlemage) ― verified** |
| GPU (untested) | Arc B580 / B570 / B60 share the same Xe2 architecture and will most likely work, but are **untested**. Their VRAM is under a third of the B70's, so the expert cache holds fewer slots and decode gets slower; try `--kv q4_0` in the config, or `--kv-resident` / `--kv-grow` |
| CPU | x86-64 with AVX2 or later. **Runs on CPUs without AVX-512** (the AVX-512 / VNNI kernels are chosen at run time; the build is `-DSTRATA_PORTABLE=ON`) |
| RAM | **60 GB-class or more** (the expert arena pins 46.84 GiB plus the serve process; 61.6 GB measured as the working floor) |
| SSD | NVMe recommended (a ~52 GB pack mmap plus 60-110 GB of model files) |
| Python | 3.10+ (the serve process; SETUP.bat creates the .venv, and offers to install Python itself via winget when none is found) |
| oneAPI | Intel oneAPI Base Toolkit / Toolkit 2026.x (**the 2026.x runtime is required**: the bundled engine imports `sycl9.dll`, which only the 2026.x runtime ships; without `C:\Program Files (x86)\Intel\oneAPI\setvars.bat` the engine will not start. SETUP.bat offers to install it via winget (`Intel.OneAPI.Toolkit`); icx for building comes from the same toolkit) |

## First-time setup (once)

1. **Double-click SETUP.bat** ― everything is installed by SETUP, in this order:
   - checks the GPU / RAM / oneAPI, offering to install missing ones via winget (Python too)
   - creates `.venv` (`pip install -r requirements.txt`); builds only if `build-rel\strata.exe` is missing
   - asks the model family, size, context and KV precision the way Strata-main's setup does, with
     recommendations computed from this PC's VRAM and RAM (`tools\setup_choices.py` -> `settings.json`)
   - asks for the data folder (default `%USERPROFILE%\Desktop\XeStrata-data`, your own Desktop -
     the prompt takes any other path); if it has no `packs\` yet, **SETUP downloads the model files
     from Hugging Face** (`tools\setup_models.py`: resumable download, then the pack and the MTP
     runtime are built there - you do not need to place or build anything by hand)
   - writes the serve config `strata-xe-iq3_s.json` from `strata-xe-template.json`
   - the download is ~66-84 GB (the size question shows it per quant); an NVMe is strongly
     recommended for the data folder
   - only if you already own the files from another source: put `packs\`, `models\` and `mtp\` under
     the data folder and SETUP will use them instead of downloading. Do not mix a pack written by a
     different engine version (see the `experts.bin` caution below)
2. **Double-click START-HERE.bat** ― the OpenAI-compatible server starts on :8080 (engine load takes 1-2 minutes). Closing the window stops the server.

## SETUP's choices

SETUP.bat asks four questions in the same order as Strata-main's setup (images and the experimental
speed projection are not offered - this port has neither):

1. **Which model?** ― Qwen3.8-Flash-Next (general) or Coder (half the experts, ~32 GB of RAM, faster)
2. **Which size?** ― IQ3_S (verified on this port) / IQ3_XXS / IQ2_XS / Q2_0 (untested), each line
   showing the download size, the RAM it uses, and whether your RAM is enough
3. **Context?** ― 32K / 64K / 128K / 256K, with how much of the VRAM is left for the expert cache
4. **KV cache?** ― 8-bit (what every published number was measured with) or 4-bit (half the memory,
   slightly less precise on long documents)

The recommendation is the biggest context that leaves **8 GiB or more for the expert cache**, trying
8-bit KV first and 4-bit second: a B70 lands on 256K + 8-bit (the verified config), a 12 GB card
(B580-class) on 64K + 4-bit KV (computed, not measured on real hardware). The answers go to
`settings.json`; the download and the serve config follow them. Re-run SETUP.bat to change them
(`tools\setup_choices.py` asks again; `tools\setup_config.py --force` rewrites the config).

## Scripts

- **START-HERE.bat** ― starts the server; asks for elevation once (the Monitor tab's PCIe counters need it). Prints the engine version / git HEAD / pack / ctx / kv / env before launching
- **REBUILD-AND-RESTART.bat** ― after editing C++: stop, incremental ninja build, restart (the running exe locks the link, so this order is required)
- **CONFIGURE-AND-BUILD.bat** ― the first full build (icx, `STRATA_ENABLE_XE=ON`; blosc2 comes from the bundled `third_party\blosc2\` include/lib; ~1 hour)

## Cautions (collisions and traps)

- **Do not run two XeStrata trees on one machine at the same time.** :8080 belongs to one server, so START-HERE.bat never kills anything: if another XeStrata server is running it waits up to 2 minutes for it to close, then aborts with a message. Close the other tree's window first, then start this one.
- **A pack's `experts.bin` must match the engine's geometry.** A pack written by a different engine version is refused at start ("this is not the pack this geometry came from"). Make a pack copy without `experts.bin` (`robocopy <pack> <pack>43 /E /XF experts.bin`); the new engine writes its own on its first start.
- **The arena is pinned** (`STRATA_ARENA_NO_REGISTER=1` in the config env is required: without it the driver registers the 47 GiB arena on the GPU side, the expert cache fails to allocate and the Xe runtime self-terminates at the hang limit).
- **keepwarm is built into the server** (config `"keepwarm": true`: one 1-token ping every 45 s, only while idle; no separate process).
- **expert profile**: the engine's verify path needs a profile-filled VRAM expert tier, so the tree ships a generic `data\expert-profile.bin` and SETUP puts it in the config args. The server keeps learning its own into `data\expert-profile-learned.bin` and takes over from that once it exists (same model shape).
- **engine version**: `build-rel\strata.exe --version` (only prints after setvars), or `GET :8080/metrics` -> `engine.version`.
- **the WebUI's Updates feature is removed in this build** (no Updates card, no update dot, no `/update` endpoints; check GitHub Releases yourself to update).
- **engine log**: `xe_serve_engine.log` (next to the scripts).

## Verified numbers (B70 + IQ3_S + 262k ctx + kv int8)

- start parity: `17*23=391`, an enumeration task, WebUI GPU telemetry: all 7 series populated
- decode 68-81 tok/s, generate prompt ~1,740 tok/s (serve follow ~190 ms)
- expert cache 11,648 slots / 92% hit rate (if a near-262k conversation drops the hit rate or runs out of VRAM, one config line reverts to `--kv q4_0`)

## Reproducible builds

`CMakeLists.txt` maps the builder's source paths out of the binary's `__FILE__` strings with `-Xclang -fmacro-prefix-map` (to `C:/xestrata-build`), so distributed binaries carry no personal build paths.

## Credits

- **The original software:** [XeStrata](https://github.com/MistVVK/XeStrata), by MistVVK and the XeStrata contributors; itself a port of [Strata](https://github.com/Niko1221/Strata), by Niko1221 and the Strata contributors.
- **The model:** [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), by the Qwen team; each model's own license applies to its files.
- **Parts it uses:** c-blosc2 (BSD 3-Clause), ggml / llama.cpp (MIT), oneMath (Apache-2.0), intel/llvm's DPC++ (Apache-2.0 WITH LLVM-exception) and the Outfit font (SIL OFL 1.1) - each with its license text in `third_party/` and `LICENSES/`.
- **This Windows port:** rakutuki.

## License

LGPL-3.0-or-later, same as upstream (`COPYING` / `COPYING.LESSER` / `LICENSES/` / `REUSE.toml`).
