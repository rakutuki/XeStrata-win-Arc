<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata-win-Arc

English | [日本語](README.ja.md)

> **About this copy:** this repository is a Windows port of [MistVVK/XeStrata](https://github.com/MistVVK/XeStrata) (verified on an Intel Arc B70).
> The Windows steps are in [README-WIN.md](README-WIN.md); the Linux path is upstream's own.

**XeStrata-win-Arc is the Windows port of XeStrata: it runs Qwen3.8-Flash-Next (125B) on one Intel Arc GPU (verified on an Arc Pro B70) and an ordinary Windows 11 PC. SETUP.bat checks the PC, installs what is missing (Python, the oneAPI runtime) and downloads the model; START-HERE.bat starts the server.**

> **Contents:** [What this port is](#what-this-port-is) ・ [Install (Windows)](#install-windows) ・
> [Using it](#using-it) ・ [How it works](#how-it-works) ・ [About upstream XeStrata (Linux)](#about-upstream-xestrata-linux) ・
> [Credits](#credits) ・ [License](#license)

## What this port is

XeStrata-win-Arc is XeStrata's engine and server built for **Windows** with Intel's oneAPI compilers, for **one
Intel Arc GPU** (Xe2/Battlemage). The Arc Pro B70 (32 GB) is verified; B580 / B570 / B60 share the design but are
untested. The model, the web app and the API are upstream's; the Linux packages, the AMD/NVIDIA paths and the
`xestrata` command are not part of this build.

What you need: Windows 11 x64, an Intel Arc GPU (12 GB+ of VRAM works; 32 GB verified), 60 GB-class RAM, an AVX2
x86-64 CPU, and about 70 GB of free disk. Details: [README-WIN.md](README-WIN.md).

## Install (Windows)

1. Copy this folder to the PC (anywhere).
2. Double-click **SETUP.bat**: it checks the GPU, the RAM and oneAPI, installs Python and the oneAPI runtime through
   winget when they are missing, asks four questions (the model, its size, the context, the KV) with recommendations
   computed for this PC, downloads the model from Hugging Face and writes the config.
3. Double-click **START-HERE.bat**: loading the model takes 1-3 minutes (the PC can be slow while it does), and the
   browser opens `http://127.0.0.1:8080` when it is ready.

The questions, the verified numbers, the cautions and building from source: [README-WIN.md](README-WIN.md).

## Using it

- **In the browser:** `http://127.0.0.1:8080` - Chat, the GPU/CPU/RAM Monitor, and About.
- **Apps and coding agents:** add it as an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`** (Anthropic-style apps: `/v1/messages`). Any API key and model name work.
- **Thinking depth:** Off / Low / Medium / High in the chat page's menu.
- Close the START-HERE window to stop the model. Do not run two trees at once on one PC.

## How it works

Models like this one normally run on servers with hundreds of gigabytes of GPU memory. XeStrata fits it on one GPU by
**sharing the work across the whole PC**.

- **The model is a team of 24,576 small specialists ("experts"),** and each word needs only 10 of them. So they do not
  all have to be on the GPU at once.
- **The GPU** does the work every word needs, and holds the few thousand experts used most often. It keeps learning
  which ones those are while you use it.
- **The RAM** holds every expert. When a word needs one the GPU does not have, **the CPU** computes it, at the same
  time as the GPU, so neither waits for the other.
- **The SSD** holds a large table, of which only a few rows are read per word.
- **Guess, then check:** a small helper built into the model guesses the next few words, and the big model checks
  them all at once. It keeps the ones that are right and writes the next word itself. The big model always decides,
  so the quality of the answer does not change.
- **Long texts are read in large pieces** (up to 32,768 tokens at a time).

Matrix products run on XMX on Intel GPUs, on WMMA on AMD GPUs (RDNA3 or later) and on tensor cores on NVIDIA GPUs;
a GPU without matrix engines uses integer dot-product instructions such as DP4a.
Every part is explained in [docs/DETAILS.md](docs/DETAILS.md#how-it-works), and how it is done on Intel GPUs and
what has been checked in [docs/XE.md](docs/XE.md).

## About upstream XeStrata (Linux)

Upstream [MistVVK/XeStrata](https://github.com/MistVVK/XeStrata) runs the same model on **Linux** with Intel, AMD
(RDNA2 or later) and NVIDIA GPUs: deb and rpm packages, the `xestrata` command, images, two or more GPUs and the
low-RAM mode. That Linux path is not built or checked in this repository - for it, read upstream's own README.
What this port adds and changes is in [README-WIN.md](README-WIN.md) and [docs/XE.md](docs/XE.md).

## Credits

- **This Windows port (XeStrata-win-Arc):** rakutuki.
- **The original software:** [Strata](https://github.com/Niko1221/Strata), by Niko1221 and the Strata contributors.
- **The models:**
   - [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), by the Qwen team;
   - compressed versions and the [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) by
     [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
   - [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), by UkisAI;
   - UD-IQ4_XS and UD-Q4_K_XL, by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).

  Each model's own license applies to its files.
- **Parts it uses:**
   - parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT);
   - [oneMath](https://github.com/uxlfoundation/oneMath) (Apache-2.0): the dense matrix products (the contrib-llvm and
     contrib-icpx modes, and the free mode for AMD GPUs), which CMake fetches from XeStrata's fork with XeStrata's
     changes ([MistVVK/oneMath](https://github.com/MistVVK/oneMath), under the same license);
   - [intel/llvm](https://github.com/intel/llvm)'s DPC++ (Apache-2.0 WITH LLVM-exception): the free and contrib-llvm
     modes' compiler, built with XeStrata's fixes (`third_party/main/intel-llvm/patches/`, under the same license);
     the packages carry its SYCL runtime;
   - the web app's font, [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1).
- **Ideas from:**
   - [Splash](https://github.com/incoai/splash), by Inco AI and the Splash contributors;
   - [ninfer](https://github.com/Neroued/ninfer), by Neroued;
   - [HyperQwen](https://github.com/syv-ai/HyperQwen), by syv.ai and the HyperQwen contributors;
   - [Project Maya](https://github.com/mw00/project-maya), by mw00 and the Project Maya contributors (how many of the
     experts missing from VRAM the CPU computes and how many are read over PCIe);
   - [nextsycl](https://github.com/maxfridbe/nextsycl), by maxfridbe (which kernels the decode of the Qwen3.5 family's
     models merges).
- More in [docs/DETAILS.md](docs/DETAILS.md#credits-and-licenses).

## License

XeStrata (copyright MistVVK and the XeStrata contributors) is free software; the whole of it is under the
[GNU Lesser General Public License, version 3 or later](COPYING.LESSER) (`COPYING.LESSER`, with `COPYING`).

- **Linking with the vendors' math libraries:** what is linked with a math or compute library a GPU or CPU maker
  publishes for its processors, and with the runtime libraries it needs (oneMKL and the runtime libraries of Intel's
  oneAPI compilers, cuBLAS and the CUDA runtime and driver, for example), may be distributed as well: an additional
  permission under section 7 of the GPL, version 3, worded in [NOTICE](NOTICE).
- **Code from Strata and ggml:** the code that comes from Strata and from ggml / llama.cpp, both MIT (the kernels
  transcribed into `src/` included), is part of XeStrata under the LGPL; as MIT asks, their copyright and permission
  notices are kept in [NOTICE](NOTICE).
- **Exceptions:** these, in `third_party/`, stay under their own licenses, one folder per project with its license text.
   - XeStrata's patches to intel/llvm (`third_party/main/intel-llvm/`): Apache-2.0 WITH LLVM-exception.
   - XeStrata's changes to oneMath (its fork, `third_party/main/oneMath/`): Apache-2.0.
   - ggml's `ggml-common.h` (`third_party/main/ggml/`, an unmodified copy): MIT.
   - The app's font Outfit (`third_party/main/outfit/`): SIL Open Font License 1.1.
   - c-blosc2 (`third_party/blosc2/`): BSD 3-Clause - this Windows port ships its `libblosc2.dll`,
     headers and import library there.
   - `third_party/nonfree/`: what is not free software. The original model's chat template and the experimental
     speed projection's vector, both under the Qwen Community License 1.0.
- **Without `third_party/nonfree/`,** XeStrata still builds and runs. The packages leave it out.
- **What else the packages carry:** intel/llvm's SYCL runtime (Apache-2.0 WITH LLVM-exception), llama.cpp's gguf-py
  (MIT) and oneMath (Apache-2.0), each with its license text.
  Nothing that is not free software (oneMKL, cuBLAS, NVIDIA's driver) is in them.
- **The models** are not part of this repository; each model's own license applies to its files.
