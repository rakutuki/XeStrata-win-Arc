"""tests/tel_windows_probe.py - read-only check of serve/telemetry.py's Windows GPU path (no server restart).

Run from the repo root:  python tests\\tel_windows_probe.py
Prints three samples 1s apart (one during a real generation) and what the web app would show."""
import sys
import time

sys.path.insert(0, ".")
from serve.telemetry import Telemetry  # noqa: E402

tel = Telemetry(extra=lambda: {})
print("static:", tel.static)
g = tel.gpu
print("gpu class:", type(g).__name__, "ok:", g.ok())
print("gpu_name:", g.name())
print("counters:", len(g.counters) if getattr(g, "counters", None) else 0)

import subprocess  # noqa: E402


def load():
    subprocess.run(
        ["curl", "-s", "-m", "60", "http://127.0.0.1:8080/v1/chat/completions",
         "-H", "Content-Type: application/json",
         "-d", '{"model":"xe-strata-iq3-s","messages":[{"role":"user","content":"Write a very long detailed essay about GPUs, at least 800 words."}],"max_tokens":900,"temperature":0}'],
        capture_output=True, timeout=70)


th = __import__("threading").Thread(target=load, daemon=True)
th.start()
for i in range(8):
    time.sleep(1.0)
    s = tel.sample()
    gb = lambda v: f"{v/2**30:.2f}GiB" if v else "-"  # noqa: E731
    print(f"s{i}: util={s.get('gpu_util')}% vram={gb(s.get('gpu_mem_used'))}/{gb(s.get('gpu_mem_total'))} "
          f"pcie=gen{s.get('gpu_pcie_gen')}x{s.get('gpu_pcie_width')} temp={s.get('gpu_temp')} "
          f"power={s.get('gpu_power')} limit={s.get('gpu_power_limit')} "
          f"pcie_rx={s.get('gpu_pcie_rx_mb')} pcie_tx={s.get('gpu_pcie_tx_mb')}")
th.join(timeout=70)
print("done")
