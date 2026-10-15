<!--
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeStrata-win-Arc

[English](README.md) | 日本語

> **このコピーについて:** このリポジトリは [MistVVK/XeStrata](https://github.com/MistVVK/XeStrata) の Windows ポートです（Intel Arc B70 検証済み）。
> Windows での手順は [README-WIN.ja.md](README-WIN.ja.md) にあります。Linux の経路は上流そのもので、このビルドには含まれません。

**XeStrata-win-Arc は XeStrata の Windows ポートです。Qwen3.8-Flash-Next（1,250 億パラメーター）を Intel Arc 1 枚（Arc Pro B70 検証済み）と普通の Windows 11 PC で動かします。SETUP.bat が PC を確かめて足りないもの（Python、oneAPI ランタイム）を入れ、モデルをダウンロードします。起動は START-HERE.bat です。**

> **目次:** [このポートについて](#このポートについて) ・ [インストール（Windows）](#インストールwindows) ・
> [使い方](#使い方) ・ [しくみ](#しくみ) ・ [上流 XeStrata（Linux）について](#上流-xestratalinuxについて) ・
> [クレジット](#クレジット) ・ [ライセンス](#ライセンス)

## このポートについて

XeStrata-win-Arc は XeStrata のエンジンとサーバを **Windows** 向けに Intel oneAPI コンパイラでビルドしたものです。
対応 GPU は **Intel Arc 1 枚**（Xe2/Battlemage）で、Arc Pro B70（32 GB）で検証済みです。B580 / B570 / B60 は同じ
設計ですが未検証です。モデル、画面、API は上流と同じです。Linux のパッケージ、AMD・NVIDIA の経路、`xestrata`
コマンドはこのビルドには含まれません。

必要なもの: Windows 11 x64、Intel Arc GPU（VRAM 12 GB 以上で動きます、32 GB で検証済み）、RAM 60 GB 級、
AVX2 のある x86-64 CPU、ディスク約 70 GB。詳しくは [README-WIN.ja.md](README-WIN.ja.md)。

## インストール（Windows）

1. このフォルダを PC にコピーします（どこでも構いません）。
2. **SETUP.bat** をダブルクリックします: GPU・RAM・oneAPI を確かめ、足りなければ Python と oneAPI ランタイムを
   winget で入れます。モデル・大きさ・文脈の長さ・KV の 4 つをこの PC の計算付きで尋ね、Hugging Face から
   モデルをダウンロードして設定を書きます。
3. **START-HERE.bat** をダブルクリックします: モデルの読み込みに 1〜3 分かかります（その間 PC が重くなること
   があります）、準備できるとブラウザで `http://127.0.0.1:8080` が開きます。

SETUP の質問、検証済みの数値、注意、ソースからのビルドは [README-WIN.ja.md](README-WIN.ja.md) にあります。

## 使い方

- **ブラウザ**: `http://127.0.0.1:8080` - Chat、GPU・CPU・RAM の Monitor、About があります。
- **アプリやコーディングエージェント**: 「OpenAI 互換」の接続先としてベース URL **`http://127.0.0.1:8080/v1`** を
  指定します（Anthropic 形式のアプリは `/v1/messages`）。API キーとモデル名は何でも構いません。
- **思考の深さ**: チャットの画面のメニューで Off / Low / Medium / High が選べます。
- START-HERE のウィンドウを閉じるとモデルが止まります。同じ PC で 2 つのツリーを同時に起動しないでください。

## しくみ

このようなモデルは、ふつう数百 GB の GPU のメモリを持つサーバーで動かします。XeStrata は、**PC 全体で仕事を分け合う**ことで、
1 枚の GPU に収めています。

- **モデルは 24,576 個の小さな専門家（エキスパート）の集まり**で、1 語を書くのに使うのはそのうち 10 個だけです。
  そのため、すべてを一度に GPU に置く必要はありません。
- **GPU** は、どの語にも必要な計算と、よく使われる数千個のエキスパートを受け持ちます。どれがよく使われるかは、
  使っている間も学び続けます。
- **RAM** にはすべてのエキスパートを置きます。GPU にないエキスパートが要るときは、**CPU** がそれを計算します。
  GPU と同時に計算するので、互いに待ちません。
- **SSD** には大きな表を置き、1 語ごとにその数行だけを読みます。
- **推測してから確かめる**: モデルに組み込まれた小さな助手が次の数語を推測し、大きなモデルがそれをまとめて確かめます。
  合っているものは残し、次の 1 語は自分で書きます。決めるのはいつも大きなモデルなので、答えの品質は変わりません。
- **長い文章は大きな塊で読みます**（一度に最大 32,768 トークン）。

行列積は、Intel の GPU では XMX、AMD の GPU では WMMA（RDNA3 以降）、NVIDIA の GPU では Tensor Core で計算し、
行列エンジンのない GPU では DP4a などの整数の内積の命令を使います。
各部分の詳しい説明は [docs/DETAILS.ja.md](docs/DETAILS.ja.md#しくみ)、Intel の GPU での実装と確かめたことは
[docs/XE.ja.md](docs/XE.ja.md) にあります。

## 上流 XeStrata（Linux）について

上流の [MistVVK/XeStrata](https://github.com/MistVVK/XeStrata) は、同じモデルを **Linux** で Intel・AMD（RDNA2
以降）・NVIDIA の GPU で動かします: deb と rpm のパッケージ、`xestrata` コマンド、画像、2 枚以上の GPU、
省 RAM モードがあります。その Linux の経路はこのリポジトリではビルドも確認もしていません。Linux については
上流自身の README を読んでください。このポートが足したものと変えたものは
[README-WIN.ja.md](README-WIN.ja.md) と [docs/XE.ja.md](docs/XE.ja.md) にあります。

## クレジット

- **この Windows ポート（XeStrata-win-Arc）:** rakutuki。
- **元のソフトウェア**: Niko1221 と Strata の貢献者による [Strata](https://github.com/Niko1221/Strata)。
- **モデル**:
   - Qwen チームの [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。
   - [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) による圧縮版と、
     [Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)。
   - UkisAI による [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)。
   - [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) による UD-IQ4_XS と UD-Q4_K_XL。

  モデルのファイルには、それぞれのライセンスが適用されます。
- **使っている部品**:
   - [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)（MIT）の一部。
   - [oneMath](https://github.com/uxlfoundation/oneMath)（Apache-2.0）: 密な行列積（contrib-llvm と contrib-icpx、free では AMD の GPU）。
     CMake が XeStrata の変更の入ったフォーク（[MistVVK/oneMath](https://github.com/MistVVK/oneMath)、同じライセンス）から取得します。
   - [intel/llvm](https://github.com/intel/llvm) の DPC++（Apache-2.0 WITH LLVM-exception）: free と contrib-llvm のコンパイラ。
     XeStrata の修正（`third_party/main/intel-llvm/patches/`、同じライセンス）を当ててビルドし、パッケージにはその SYCL の実行時を入れます。
   - 画面のフォント [Outfit](https://github.com/Outfitio/Outfit-Fonts)（SIL Open Font License 1.1）。
- **参考にした考え方**:
   - Inco AI と Splash の貢献者による [Splash](https://github.com/incoai/splash)。
   - Neroued による [ninfer](https://github.com/Neroued/ninfer)。
   - syv.ai と HyperQwen の貢献者による [HyperQwen](https://github.com/syv-ai/HyperQwen)。
   - mw00 と Project Maya の貢献者による [Project Maya](https://github.com/mw00/project-maya)（VRAM にないエキスパートを CPU と PCIe に分ける数の決め方）。
   - maxfridbe による [nextsycl](https://github.com/maxfridbe/nextsycl)（Qwen3.5 系のモデルの decode でカーネルをまとめるやり方）。
- 詳しくは [docs/DETAILS.ja.md](docs/DETAILS.ja.md#クレジットとライセンス) にあります。

## ライセンス

XeStrata（著作権は MistVVK と XeStrata の貢献者）は自由ソフトウェアで、全体を
[GNU Lesser General Public License の第 3 版以降](COPYING.LESSER)（`COPYING.LESSER` と `COPYING`）で配布します。

- **メーカーの計算ライブラリとのリンク**: GPU や CPU のメーカーが自社の製品向けに出している計算ライブラリとその実行時ライブラリ
  （oneMKL や Intel oneAPI のコンパイラの実行時ライブラリ、cuBLAS や CUDA のランタイムとドライバなど）とリンクしたものも配布できます。
  GPL 第 3 版の第 7 条による追加の許可で、文面は [NOTICE](NOTICE) にあります。
- **Strata と ggml から来たコード**: MIT ライセンスの Strata と ggml / llama.cpp から来たコード（`src/` に書き写したカーネルを含む）も、
  XeStrata の一部として LGPL です。MIT の条件どおり、著作権表示と許諾表示を [NOTICE](NOTICE) に残しています。
- **例外**: `third_party/` の次のものは、元のライセンスのままです。プロジェクトごとのフォルダーに、ライセンスの文書と一緒に置いています。
   - intel/llvm への XeStrata の修正のパッチ（`third_party/main/intel-llvm/`）: Apache-2.0 WITH LLVM-exception。
   - oneMath への XeStrata の変更（フォーク、`third_party/main/oneMath/`）: Apache-2.0。
   - ggml の `ggml-common.h`（`third_party/main/ggml/`、変えていない写し）: MIT。
   - 画面のフォント Outfit（`third_party/main/outfit/`）: SIL Open Font License 1.1。
   - c-blosc2（`third_party/blosc2/`）: BSD 3-Clause。この Windows ポートはそこにその `libblosc2.dll`、
     ヘッダー、インポートライブラリを同梱しています。
   - `third_party/nonfree/`: 自由ソフトウェアでないもの。元のモデルのチャットテンプレートと、実験的な速度向上用の射影のベクトルで、
     どちらも Qwen Community License 1.0 です。
- **`third_party/nonfree/` がなくても**、XeStrata はビルドでき、動きます。パッケージには入れていません。
- **パッケージに入れているほかのもの**: intel/llvm の SYCL の実行時（Apache-2.0 WITH LLVM-exception）、llama.cpp の gguf-py（MIT）、
  oneMath（Apache-2.0）を、それぞれのライセンスの文書と一緒に入れています。
  自由ソフトウェアでないもの（oneMKL、cuBLAS、NVIDIA のドライバー）は入れていません。
- **モデル**: このリポジトリには含みません。モデルのファイルには、それぞれのライセンスが適用されます。
