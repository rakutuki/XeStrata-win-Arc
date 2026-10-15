# XeStrata-win ― Windows版（Intel Arc B70 検証済み）

このリポジトリは [XeStrata](https://github.com/MistVVK/XeStrata) の **Windows ポート**です（upstream Strata 0.1.43 / xe0.1.43.1 ベース）。
Linux 版の全体像は [README.ja.md](README.ja.md) と [docs/XE.ja.md](docs/XE.ja.md) を参照してください。ここでは Windows 固有のことだけを書きます。

## 動作確認済み環境

| | |
| --- | --- |
| OS | Windows 11 x64 |
| GPU | **Intel Arc Pro B70（32 GB、Xe2/Battlemage）― 検証済み** |
| GPU（未検証） | Arc B580 / B570 / B60 は同じ Xe2 アーキテクチャなので動く可能性が高いが**未検証**。VRAM が B70 の 3 分の 1 以下になるため expert cache のヒット率が下がり decode が遅くなる見込み。その場合は config の `--kv` を `q4_0` に戻すか `--kv-resident` / `--kv-grow` を検討してください |
| CPU | x86-64 で AVX2 以上。**AVX-512 非対応の CPU でも動作可能（未検証）**（AVX-512 / VNNI のカーネルは実行時に選択される。ビルドは `-DSTRATA_PORTABLE=ON` 基準） |
| RAM | **60 GB 級以上必須**（expert arena が 46.84 GiB を pinned 確保 + serve 本体。61.6 GB 実機が動作下限の実測値） |
| SSD | NVMe 推奨（pack 約 52 GB の mmap + モデル合計 60〜110 GB） |
| Python | 3.10 以上（serve 本体。SETUP.bat が .venv を作ります。Python 自体が見つからない場合は winget での導入を提案します） |
| oneAPI | Intel oneAPI Base Toolkit / Toolkit 2026.x（**2026.x ランタイム必須**: 同梱エンジンは 2026.x ランタイムだけが同梱する `sycl9.dll` を参照します。`C:\Program Files (x86)\Intel\oneAPI\setvars.bat` がないとエンジンは起動しません。SETUP.bat が winget（`Intel.OneAPI.Toolkit`）での導入を提案します。ビルドに使う icx も同じ toolkit に含まれます） |

## 初回のセットアップ（1 回だけ）

1. **SETUP.bat をダブルクリック** ― 必要なものはすべて SETUP が入れます。この順に進みます:
   - GPU・RAM・oneAPI を検査（未導入なら winget での導入を提案。Python も同じ）
   - `.venv` 作成（`pip install -r requirements.txt`）。ビルドは `build-rel\strata.exe` が
     ない時だけ（通常は同梱済みなので不要）
   - Strata-main の setup と同じ流れでモデル family・size・コンテキスト長・KV 量子化を選択
     （この PC の VRAM と RAM から計算した推奨付き、`tools\setup_choices.py` → `settings.json`）
   - データフォルダを尋ねます（既定 `%USERPROFILE%\Desktop\XeStrata-data`＝自分のデスクトップ。
     別のパスも入力可）。そのフォルダに `packs\` がない場合、**SETUP が Hugging Face から
     モデルファイルをダウンロードします**（`tools\setup_models.py`: 中断しても再開できる
     ダウンロード後、pack と MTP ランタイムまでその場で構築。手作業で配置・構築する必要は
     ありません）
   - serve config `strata-xe-iq3_s.json` を `strata-xe-template.json` から生成
   - ダウンロード容量は約 66〜84 GB（size の質問の各行に表示）。データフォルダは NVMe 推奨
   - 別の入手元で既にファイルを持っている場合だけ: データフォルダの下に `packs\`、`models\`、
     `mtp\` を置けば SETUP はダウンロードの代わりにそれを使います。別バージョンのエンジンが
     書いた pack を混ぜないでください（下記の `experts.bin` の注意参照）
2. **START-HERE.bat をダブルクリック** ― :8080 に OpenAI 互換サーバーが起動します（エンジンロードに 1〜2 分）。このウィンドウを閉じるとサーバーが止まります

## SETUP の選択

SETUP.bat は Strata-main の setup と同じ順で 4 つを尋ねます（images と実験的 speed projection は
このポートに機能がないため聞きません）:

1. **Which model?** ― Qwen3.8-Flash-Next（汎用）か Coder（expert 半分、RAM 約 32 GB、速い）
2. **Which size?** ― IQ3_S（このポートで検証済み）/ IQ3_XXS / IQ2_XS / Q2_0（未検証）。各行に
   ダウンロード容量・使用 RAM・自分の RAM で足りるかどうかが表示されます
3. **Context?** ― 32K / 64K / 128K / 256K。expert cache に残る VRAM の量も併記されます
4. **KV cache?** ― 8-bit（公開されている数値はすべてこの設定の実測）か 4-bit（メモリ半分、
   長い文書でやや精度が落ちる）

推奨は「**expert cache に 8 GiB 以上残る最大の context**」を 8-bit KV → 4-bit KV の順で探します。
B70 なら 256K + 8-bit（検証済みの構成）、VRAM 12 GB（B580 級）なら 64K + 4-bit KV が推奨に
なります（12 GB 級は実機未検証の計算値）。答えは `settings.json` に保存され、ダウンロードと
serve config はそれに従います。後から変えたい場合は SETUP.bat を再実行してください
（`tools\setup_choices.py` が聞き直し、`tools\setup_config.py --force` で config を書き直せます）。

## 起動スクリプト

- **START-HERE.bat** ― サーバー起動。UAC 昇格を 1 回尋ねます（Monitor タブの PCIe スループット counter に必要）。起動前にエンジン version・git HEAD・pack・ctx・kv・env を表示します
- **REBUILD-AND-RESTART.bat** ― C++ を編集した後: 停止 → ninja 増分ビルド → 再起動（起動中の exe はリンクでロックされるためこの順序が必須）
- **CONFIGURE-AND-BUILD.bat** ― 最初のフルビルド（icx、`STRATA_ENABLE_XE=ON`。blosc2 は `third_party\blosc2\` の include/lib を使用。約 1 時間）

## 注意（衝突と罠）

- **同じマシンで XeStrata のツリーを 2 つ同時に起動しないでください**。:8080 は 1 つのサーバーだけが使えるため、START-HERE.bat は他を kill しません。他の XeStrata サーバーが起動中の場合は終了を最大 2 分待ち、それでも終わらなければ起動を取りやめます。先に他のツリーのウィンドウを閉じてから起動してください
- **pack の `experts.bin` はエンジン版の geometry に一致する必要があります**。別バージョンのエンジンが書いた pack を指すと起動が拒否されます（「this is not the pack this geometry came from」）。その場合は pack のコピーから `experts.bin` を除いてください（`robocopy <pack> <pack>43 /E /XF experts.bin`）。新しいエンジンが初回起動時に自分で書き直します
- **arena は pinned**（config の env に `STRATA_ARENA_NO_REGISTER=1` 必須。無いと 47 GiB の arena を GPU ドライバが VRAM 側に登録し、expert cache の確保に失敗して GPU ハング制限で自死します）
- **keepwarm はサーバー内蔵**（config `"keepwarm": true`。アイドル中だけ 45 秒ごとに 1 トークン ping します。別プロセスは不要です）
- **expert profile**: エンジンの verify 経路は profile 済みの VRAM expert tier を必要とするため、汎用の `data\expert-profile.bin` をツリーに同梱し、SETUP が config の args に追加します。サーバーは独自の学習を `data\expert-profile-learned.bin` に続け、それが生成されたら（同一モデル形状なら）そちらに切り替えます
- **エンジン version の確認**: `build-rel\strata.exe --version`（setvars 呼び出し後でないと何も表示しません）、または `GET :8080/metrics` の `engine.version`
- **WebUI の Updates 機能はこのビルドでは除去済み**（About タブの Updates カード・更新通知ドット・`/update` エンドポイントはありません。更新は GitHub の Releases を自分で確認します）
- **エンジンログ**: `xe_serve_engine.log`（bat と同じフォルダにできます）

## 検証済み数値（B70 + IQ3_S + 262k ctx + kv int8）

- 起動 parity: `17*23=391`、列挙課題、WebUI GPU telemetry 7 項目すべて表示
- decode 68〜81 tok/s、generate prompt 約 1,740 tok/s（serve follow 約 190 ms）
- expert cache 11,648 slots / ヒット率 92%（262k 級の会話でヒット率が落ちるか VRAM が尽きたら、config 1 行で `--kv q4_0` に戻せます）

## ビルドの再現性

`CMakeLists.txt` は `-Xclang -fmacro-prefix-map` でビルドマシンのソースパスをバイナリ内の `__FILE__` 文字列から中立化します（`C:/xestrata-build` にマップ）。配布バイナリにビルドマシンの個人パスは残りません。

## クレジット

- **元のソフトウェア:** MistVVK と XeStrata の貢献者による [XeStrata](https://github.com/MistVVK/XeStrata)。その元は Niko1221 と Strata の貢献者による [Strata](https://github.com/Niko1221/Strata) です。
- **モデル:** Qwen チームの [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。モデルのファイルにはそれぞれのライセンスが適用されます。
- **使っている部品:** c-blosc2（BSD 3-Clause）、ggml / llama.cpp（MIT）、oneMath（Apache-2.0）、intel/llvm の DPC++（Apache-2.0 WITH LLVM-exception）、Outfit フォント（SIL OFL 1.1）。それぞれのライセンス文書は `third_party/` と `LICENSES/` にあります。
- **この Windows ポート:** rakutuki。

## ライセンス

upstream と同じ LGPL-3.0-or-later（`COPYING` / `COPYING.LESSER` / `LICENSES/` / `REUSE.toml` 参照）。
