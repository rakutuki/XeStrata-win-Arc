# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: the Intel GPU on the xe driver, from Linux (temperature and power from hwmon, the PCIe link, the load of this
  user's processes from their DRM fdinfo) and its VRAM from Level Zero's sysman through ctypes, so no pip package is
  needed.  The same Intel GPU on Windows: its VRAM and static PCIe link from Level Zero's sysman (ze_loader.dll),
  its load from the Performance Counters' GPU Engine set, its name from WMI, and its temperature and power draw
  from Intel's control library (IntelControlLib.dll, the DriverStore's IGCL runtime); the link's GPU->host traffic
  comes from sysman's PCI stats when the process is elevated (the Intel Windows driver leaves the host->GPU
  counter at 0, so that direction stays None).  An NVIDIA GPU (the contrib build): NVIDIA's NVML
  library (libnvidia-ml.so.1, installed with the driver) through ctypes: load, VRAM, temperature, power, PCIe link
  and throughput.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  Linux /proc and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import glob
import os
import platform
import sys
import threading
import time

HISTORY = 60


# ------------------------------------------------------------------------------------------------ the Intel GPU
class _ZesPciAddress(ctypes.Structure):
    _fields_ = [("domain", ctypes.c_uint32), ("bus", ctypes.c_uint32), ("device", ctypes.c_uint32),
                ("function", ctypes.c_uint32)]


class _ZesPciSpeed(ctypes.Structure):
    _fields_ = [("gen", ctypes.c_int32), ("width", ctypes.c_int32), ("max_bandwidth", ctypes.c_int64)]


class _ZesPciProps(ctypes.Structure):    # zes_pci_properties_t
    _fields_ = [("stype", ctypes.c_uint32), ("pnext", ctypes.c_void_p), ("address", _ZesPciAddress),
                ("max_speed", _ZesPciSpeed), ("bw", ctypes.c_uint8), ("pkt", ctypes.c_uint8),
                ("replay", ctypes.c_uint8)]


class _ZesMemState(ctypes.Structure):    # zes_mem_state_t
    _fields_ = [("stype", ctypes.c_uint32), ("pnext", ctypes.c_void_p), ("health", ctypes.c_int32),
                ("free", ctypes.c_uint64), ("size", ctypes.c_uint64)]


class _Zes:
    """The device's VRAM through Level Zero's management API (sysman, libze_loader.so.1, which the GPU runtime
    installs) through ctypes: the reading Linux's sysfs does not give for the xe and i915 drivers."""

    STYPE_PCI_PROPERTIES, STYPE_MEM_STATE = 0x2, 0x1E

    def __init__(self, pci):
        self.mems = []
        try:
            self.lib = ctypes.CDLL("libze_loader.so.1")
            if self.lib.zesInit(ctypes.c_uint32(0)) != 0:
                return
            n = ctypes.c_uint32(0)
            if self.lib.zesDriverGet(ctypes.byref(n), None) != 0 or n.value == 0:
                return
            drivers = (ctypes.c_void_p * n.value)()
            self.lib.zesDriverGet(ctypes.byref(n), drivers)
            for drv in drivers:
                m = ctypes.c_uint32(0)
                if self.lib.zesDeviceGet(ctypes.c_void_p(drv), ctypes.byref(m), None) != 0:
                    continue
                devs = (ctypes.c_void_p * m.value)()
                self.lib.zesDeviceGet(ctypes.c_void_p(drv), ctypes.byref(m), devs)
                for dev in devs:
                    props = _ZesPciProps(stype=self.STYPE_PCI_PROPERTIES)
                    if self.lib.zesDevicePciGetProperties(ctypes.c_void_p(dev), ctypes.byref(props)) != 0:
                        continue
                    a = props.address
                    if f"{a.domain:04x}:{a.bus:02x}:{a.device:02x}.{a.function:x}" != pci:
                        continue
                    k = ctypes.c_uint32(0)
                    self.lib.zesDeviceEnumMemoryModules(ctypes.c_void_p(dev), ctypes.byref(k), None)
                    mods = (ctypes.c_void_p * k.value)()
                    self.lib.zesDeviceEnumMemoryModules(ctypes.c_void_p(dev), ctypes.byref(k), mods)
                    self.mems = list(mods)
        except (OSError, AttributeError):
            self.mems = []

    def read(self):
        """(used, total) bytes over the device's memory modules, or (None, None)."""
        free = total = 0
        for mod in self.mems:
            st = _ZesMemState(stype=self.STYPE_MEM_STATE)
            try:
                if self.lib.zesMemoryGetState(ctypes.c_void_p(mod), ctypes.byref(st)) != 0:
                    return None, None
            except (OSError, AttributeError):
                return None, None
            free += st.free
            total += st.size
        return (total - free, total) if total else (None, None)


def _read(path, conv=int):
    try:
        with open(path, encoding="ascii") as f:
            return conv(f.read().strip())
    except (OSError, ValueError):
        return None


PCI_IDS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids")


def _pci_ids_name(dev):
    """The card's name in the system's PCI ID list ("Battlemage G31 [Intel Graphics]", "Navi 44 [Radeon RX 9050 /
    9060 XT]") from its sysfs folder's vendor and device IDs, or None: neither the xe driver nor sysman names a card
    the driver does not know by its product name, and amdgpu gives product_name for few cards."""
    vendor = _read(os.path.join(dev or "", "vendor"), str)
    device = _read(os.path.join(dev or "", "device"), str)
    if not vendor or not device:
        return None
    vendor, device = vendor.lower().replace("0x", ""), device.lower().replace("0x", "")
    for path in PCI_IDS:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                in_vendor = False
                for line in f:
                    if line.startswith("#") or not line.strip():   # comments sit inside a vendor's list too
                        continue
                    if not line.startswith("\t"):
                        in_vendor = line.startswith(vendor + " ")
                    elif in_vendor and line.startswith(f"\t{device} "):
                        return line.strip()[len(device):].strip() or None
        except OSError:
            continue
    return None


def _gen(text):
    """The PCIe generation of a link-speed file's contents ("16.0 GT/s PCIe" -> 4), or None."""
    try:
        return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}[float(str(text).split()[0])]
    except (ValueError, TypeError, IndexError, KeyError):
        return None


def _bdf(s):
    """True for a sysfs pci device name ("0000:03:00.0")."""
    return (len(s) == 12 and s[4] == ":" and s[7] == ":" and s[10] == "." and s[11] in "01234567"
            and all(c in "0123456789abcdef" for c in s[0:4] + s[5:7] + s[8:10]))


def _pci_hops(dev):
    """The sysfs folders of the PCI devices from the root port down to the card `dev`."""
    hops, prefix = [], []
    for part in os.path.realpath(dev or "").split("/"):
        prefix.append(part)
        if _bdf(part):
            hops.append("/".join(prefix))
    return hops


def _path_link(dev, hops=None):
    """The PCIe link a card actually gets (upstream 7f3c0da2): the narrowest and slowest hop between the root port and
    the card, from each hop's max_link_speed / max_link_width (a card that is Gen4 on its own hop behind a Gen3 root
    port runs at Gen3).  pcie_own_gen keeps the card's own hop aside.  The bottleneck is claimed only when every hop
    could be read (pcie_path: how many were, "4/4"); else the card's own negotiated link.  `hops`: _pci_hops(dev)."""
    hops = _pci_hops(dev) if hops is None else hops
    gens, widths, seen = [], [], 0
    for d in hops:
        g = _gen(_read(os.path.join(d, "max_link_speed"), str))
        w = _read(os.path.join(d, "max_link_width"))
        seen += 1 if (g is not None or w is not None) else 0
        if g:
            gens.append(g)
        if w:
            widths.append(w)
    own_gen = _gen(_read(os.path.join(dev or "", "current_link_speed"), str))
    whole = bool(hops) and seen == len(hops)
    gen = min(gens) if (whole and gens) else own_gen
    width = min(widths) if (whole and widths) else _read(os.path.join(dev or "", "current_link_width"))
    return {"pcie_gen": gen, "pcie_gen_max": gen, "pcie_own_gen": own_gen, "pcie_width": width,
            "pcie_path": "%d/%d" % (seen, len(hops))}


class _XeGpu:
    """An Intel GPU on the xe or i915 driver: temperature and power from Linux's sysfs (hwmon), the PCIe link from the
    PCI path, the load from the DRM clients' fdinfo, the VRAM through sysman.  `pci` is the card's address as the
    config gives it (setup's gpu_pci); without one, the first card on one of `drivers`."""
    LINK_GEN = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}

    def __init__(self, pci=None, drivers=("xe", "i915")):
        self.dev = None
        for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
            dev = os.path.realpath(os.path.join(card, "device"))
            driver = os.path.basename(os.path.realpath(os.path.join(dev, "driver")))
            if driver not in drivers:
                continue
            if pci is None or os.path.basename(dev) == pci:
                self.dev, self.driver = dev, driver
                break
        if self.dev is None:
            return
        self.pci = os.path.basename(self.dev)
        hw = glob.glob(os.path.join(self.dev, "hwmon", "hwmon*"))
        self.hwmon = hw[0] if hw else None
        self.zes = _Zes(self.pci)
        self._prev = None   # (time, energy uJ)
        self._cycles = {}   # DRM client id -> (engine cycles, total cycles); i915: (engine ns, sample time ns)

    def ok(self):
        return self.dev is not None

    def name(self):
        if not self.ok():
            return None
        device = (_read(os.path.join(self.dev, 'device'), str) or '').replace('0x', '')
        name = _pci_ids_name(self.dev) or f"GPU {device}"
        return f"Intel {name} at {self.pci}"

    def _link(self):
        """The card's widest link up its PCI path: a card with a switch of its own reports x1 on its own port.  Its
        generation now, its width, and the highest generation the path above it allows: the lowest max_link_speed from
        that link up to the root port (the B70's own switch can do Gen5 behind a Gen4 root port)."""
        best, tops, p = None, [], self.dev
        while p.startswith("/sys/devices/pci"):
            speed = _read(os.path.join(p, "current_link_speed"), str)
            width = _read(os.path.join(p, "current_link_width"))
            top = _read(os.path.join(p, "max_link_speed"), str)
            tops.append(self.LINK_GEN.get(top.split()[0]) if top else None)
            if speed and width:
                g = self.LINK_GEN.get(speed.split()[0])
                if g and (best is None or (g * width) > best[0] * best[1]):
                    best = (g, width, len(tops) - 1)
            p = os.path.dirname(p)
        if best is None:
            return None, None, None
        above = [t for t in tops[best[2]:] if t]
        return best[0], best[1], min(above) if above else None

    def _busy(self):
        """The render and compute engines' load from this user's processes on the card: the xe driver counts each
        DRM client's engine cycles in /proc/<pid>/fdinfo, i915 its engine time in ns (the card-wide load needs root
        on these drivers)."""
        now = {}
        t_ns = time.monotonic_ns()
        for f in glob.glob("/proc/[0-9]*/fdinfo/*"):
            try:
                with open(f, encoding="ascii", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if f"drm-driver:\t{self.driver}\n" not in text or f"drm-pdev:\t{self.pci}" not in text:
                continue
            kv = {}
            for line in text.splitlines():
                k, _, v = line.partition(":")
                kv.setdefault(k, v.strip())
            cid = kv.get("drm-client-id")
            try:
                if self.driver == "i915":
                    busy = sum(int(kv.get(f"drm-engine-{e}", "0 ns").split()[0]) for e in ("render", "compute"))
                    total = t_ns
                else:   # a class's cycles count all its engines (drm-engine-capacity-*, 1 when not given)
                    busy = sum(int(kv.get(f"drm-cycles-{e}", "0")) / int(kv.get(f"drm-engine-capacity-{e}", "1") or 1)
                               for e in ("rcs", "ccs"))
                    total = int(kv.get("drm-total-cycles-ccs") or kv.get("drm-total-cycles-rcs") or 0)
            except (ValueError, IndexError):
                continue
            if cid and total:
                now[cid] = (busy, total)
        prev, self._cycles = self._cycles, now
        spans = [now[c][1] - prev[c][1] for c in now if c in prev and now[c][1] > prev[c][1]]
        if not spans:
            return 0.0 if not now else None
        used = sum(max(0, now[c][0] - prev[c][0]) for c in now if c in prev)
        return max(0.0, min(100.0, 100.0 * used / max(spans)))

    def read(self):
        out = {}
        t = time.time()
        energy = _read(os.path.join(self.hwmon, "energy1_input")) if self.hwmon else None
        prev, self._prev = self._prev, (t, energy)
        out["util"] = self._busy()
        if prev and t > prev[0] and energy is not None and prev[1] is not None and energy >= prev[1]:
            out["power"] = (energy - prev[1]) / 1e6 / (t - prev[0])
        if self.hwmon:
            # the GPU's own sensor (label "pkg") as the GPU temperature, as NVIDIA's core and AMD's edge readings
            # are; the VRAM's sensors (labels vram, vram_ch_N) apart.  A hwmon without labels: the hottest, as before.
            temps = {}
            for f in glob.glob(os.path.join(self.hwmon, "temp*_input")):
                v = _read(f)
                if v:
                    temps[_read(f.replace("_input", "_label"), str) or ""] = v / 1000.0
            vram = [v for k, v in temps.items() if k.startswith("vram")]
            out["temp"] = temps.get("pkg") if "pkg" in temps else (max(temps.values()) if temps else None)
            out["mem_temp"] = max(vram) if vram else None
            cap = _read(os.path.join(self.hwmon, "power1_cap")) or _read(os.path.join(self.hwmon, "power1_max"))
            out["power_limit"] = cap / 1e6 if cap else None
        # the xe driver's hwmon can keep a sleeping GT's last temperatures (a Battlemage card that has cooled shows the
        # load's; github.com/hugs7/intel-xe-battlemage-hwmon-fix): the web app marks the reading while nothing runs
        out["temp_may_be_stale"] = self.driver == "xe" and out["util"] is not None and out["util"] < 1.0
        out["mem_used"], out["mem_total"] = self.zes.read()
        out["pcie_gen"], out["pcie_width"], out["pcie_gen_max"] = self._link()
        out["pcie_rx_mb"] = out["pcie_tx_mb"] = None   # no counters for the link's traffic
        return out


# ------------------------------------------------------------------------------------------------ an AMD GPU
SYSFS = "/sys"


class _AmdGpu:
    """An AMD GPU on the amdgpu driver, from its sysfs files (no ROCm library needed; upstream #301): load
    (gpu_busy_percent), VRAM (mem_info_vram_used / _total), and from its hwmon folder the temperature (temp1_input, the
    edge sensor, m°C), power (power1_average or power1_input, µW) and its cap (power1_cap).  `pci` is the card's
    address as the config gives it; without one, the first card on the amdgpu driver."""

    def __init__(self, pci=None, sysfs=None):
        self.dev = None
        for card in sorted(glob.glob(os.path.join(sysfs or SYSFS, "class", "drm", "card[0-9]*"))):
            dev = os.path.realpath(os.path.join(card, "device"))
            if os.path.basename(os.path.realpath(os.path.join(dev, "driver"))) != "amdgpu":
                continue
            if pci is None or os.path.basename(dev) == pci:
                self.dev = dev
                break
        self.hwmon = None
        if self.dev:
            hw = sorted(glob.glob(os.path.join(self.dev, "hwmon", "hwmon*")))
            self.hwmon = hw[0] if hw else None

    def ok(self):
        return self.dev is not None

    def name(self):
        if not self.ok():
            return None
        listed = _pci_ids_name(self.dev)
        return _read(os.path.join(self.dev, "product_name"), str) or (f"AMD {listed}" if listed else "AMD Radeon")

    _gen = staticmethod(_gen)
    _bdf = staticmethod(_bdf)

    def _hops(self):
        return _pci_hops(self.dev)

    def link(self):
        return _path_link(self.dev, self._hops())

    def read(self):
        out = {"util": _read(os.path.join(self.dev, "gpu_busy_percent")),
               "mem_used": _read(os.path.join(self.dev, "mem_info_vram_used")),
               "mem_total": _read(os.path.join(self.dev, "mem_info_vram_total"))}
        if self.hwmon:
            t = _read(os.path.join(self.hwmon, "temp1_input"))
            out["temp"] = t / 1000.0 if t is not None else None
            for f in glob.glob(os.path.join(self.hwmon, "temp*_label")):   # the VRAM's sensor ("mem"), apart
                if _read(f, str) == "mem":
                    m = _read(f.replace("_label", "_input"))
                    out["mem_temp"] = m / 1000.0 if m is not None else None
            p = _read(os.path.join(self.hwmon, "power1_average"))
            if p is None:
                p = _read(os.path.join(self.hwmon, "power1_input"))
            out["power"] = p / 1e6 if p is not None else None
            cap = _read(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap is not None else None
        out.update(self.link())
        out["pcie_rx_mb"] = out["pcie_tx_mb"] = None   # no counters for the link's traffic
        return out


# ------------------------------------------------------------------------------------------------ an NVIDIA GPU
class _NvGpu:
    """An NVIDIA GPU through NVML, with _XeGpu's interface.  `pci` is the card's address as the config gives it
    (setup's gpu_pci); without one, NVML's first card."""

    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, pci=None):
        self.lib = self.dev = None
        self.pci = pci
        for n in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            if self.lib.nvmlInit_v2() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            r = self.lib.nvmlDeviceGetHandleByPciBusId_v2(pci.encode(), ctypes.byref(h)) if pci else \
                self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0), ctypes.byref(h))
            if r != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        if not self.ok():
            return None
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace") + (f" at {self.pci}" if self.pci else "")
        except (AttributeError, OSError):
            pass
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        own = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")       # drops at idle (power saving)
        link = _path_link(os.path.join(SYSFS, "bus", "pci", "devices", self.pci)) if self.pci else {}
        if link.get("pcie_gen"):   # the link the card gets, as for AMD: the same meaning in every maker's panel
            out.update(link)
            out["pcie_own_gen"] = own
        else:
            out["pcie_gen"] = own
            out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
            out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out


# ------------------------------------------------------------------------------ an Intel GPU on Windows
class _WinZesMemState(ctypes.Structure):   # zes_mem_state_t (Windows header: stype 0x1b)
    _fields_ = [("stype", ctypes.c_uint32), ("pnext", ctypes.c_void_p), ("health", ctypes.c_int32),
                ("free", ctypes.c_uint64), ("size", ctypes.c_uint64)]


class _WinPowerProps(ctypes.Structure):    # zes_power_properties_t (Windows header: stype 0x13)
    _fields_ = [("stype", ctypes.c_uint32), ("pnext", ctypes.c_void_p),
                ("onSubdevice", ctypes.c_uint8), ("_pad", ctypes.c_uint8 * 3),
                ("engine", ctypes.c_uint32), ("limit", ctypes.c_int32),
                ("maxLimit", ctypes.c_int32), ("minLimit", ctypes.c_int32)]


class _PdhCounterValue(ctypes.Structure):
    class U(ctypes.Union):
        _fields_ = [("Large", ctypes.c_uint64), ("Double", ctypes.c_double),
                    ("Single", ctypes.c_float), ("Long", ctypes.c_int32)]
    _fields_ = [("CounterType", ctypes.c_uint32), ("u", U)]


class _CtlAppId(ctypes.Structure):         # ctl_application_id_t: a GUID, 16 bytes
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_uint8 * 8)]


class _CtlInitArgs(ctypes.Structure):      # ctl_init_args_t: 36 bytes (the ApplicationUID GUID is what makes it)
    _fields_ = [("Size", ctypes.c_uint32), ("Version", ctypes.c_uint8),
                ("AppVersion", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("SupportedVersion", ctypes.c_uint32), ("ApplicationUID", _CtlAppId)]


class _CtlTempProps(ctypes.Structure):     # ctl_temp_properties_t
    _fields_ = [("Size", ctypes.c_uint32), ("Version", ctypes.c_uint8),
                ("type", ctypes.c_uint32), ("maxTemperature", ctypes.c_double)]


class _CtlEnergy(ctypes.Structure):        # ctl_power_energy_counter_t: microjoules + microsecond timestamp
    _fields_ = [("Size", ctypes.c_uint32), ("Version", ctypes.c_uint8),
                ("energy", ctypes.c_uint64), ("timestamp", ctypes.c_uint64)]


class _Igcl:
    """The GPU's temperature and power draw through Intel's Graphics Control Library (IntelControlLib.dll, which
    the Intel graphics driver installs into the DriverStore): the readings Level Zero's sysman keeps at 0 on
    Windows.  Power is the difference of the hardware's monotonic energy counter between samples, the same method
    the Linux hwmon path uses.  Initializing it next to a running Level Zero server is safe; a machine without the
    library (or an IGCL that refuses us) simply leaves both readings None."""

    def __init__(self):
        self.lib = self.ctx = self.temp = self.pwr = None
        self._energy = None                # (timestamp_us, energy_uJ) of the previous sample
        dlls = glob.glob(r"C:\Windows\System32\DriverStore\FileRepository\iigd_dch*.inf*\IntelControlLib.dll")
        if not dlls:
            return
        try:
            lib = ctypes.CDLL(dlls[0])
            for name in ("ctlInit", "ctlEnumerateDevices", "ctlEnumTemperatureSensors",
                         "ctlTemperatureGetProperties", "ctlTemperatureGetState",
                         "ctlEnumPowerDomains", "ctlPowerGetEnergyCounter"):
                getattr(lib, name).restype = ctypes.c_uint32
            args = _CtlInitArgs()
            args.Size = ctypes.sizeof(args)
            args.Version = 0               # the struct's own header version: 0, not 1 (the driver refuses 1)
            args.AppVersion = (1 << 16) | 1
            args.flags = 1                 # CTL_INIT_FLAG_USE_LEVEL_ZERO: the telemetry surface
            ctx = ctypes.c_void_p()
            if lib.ctlInit(ctypes.byref(args), ctypes.byref(ctx)) != 0 or not ctx:
                return
            self.lib, self.ctx = lib, ctx
            n = ctypes.c_uint32(0)
            if lib.ctlEnumerateDevices(ctx, ctypes.byref(n), None) != 0 or not n.value:
                return
            devs = (ctypes.c_void_p * n.value)()
            if lib.ctlEnumerateDevices(ctx, ctypes.byref(n), devs) != 0 or not n.value:
                return
            dh = ctypes.c_void_p(devs[0])  # IGCL enumerates only Intel adapters: on this engine's PC, the Arc
            k = ctypes.c_uint32(0)
            if lib.ctlEnumTemperatureSensors(dh, ctypes.byref(k), None) == 0 and k.value:
                ts = (ctypes.c_void_p * k.value)()
                lib.ctlEnumTemperatureSensors(dh, ctypes.byref(k), ts)
                gpu = glob_min = None
                for t in ts:
                    tp = _CtlTempProps()
                    tp.Size = ctypes.sizeof(tp)
                    if lib.ctlTemperatureGetProperties(ctypes.c_void_p(t), ctypes.byref(tp)) == 0:
                        if tp.type == 1 and gpu is None:
                            gpu = ctypes.c_void_p(t)      # CTL_TEMP_SENSORS_GPU
                        if tp.type == 0 and glob_min is None:
                            glob_min = ctypes.c_void_p(t)  # CTL_TEMP_SENSORS_GLOBAL
                self.temp = gpu or glob_min
            k = ctypes.c_uint32(0)
            if lib.ctlEnumPowerDomains(dh, ctypes.byref(k), None) == 0 and k.value:
                ps = (ctypes.c_void_p * k.value)()
                lib.ctlEnumPowerDomains(dh, ctypes.byref(k), ps)
                self.pwr = ctypes.c_void_p(ps[0])
        except (OSError, AttributeError):
            self.lib = self.ctx = self.temp = self.pwr = None

    def temperature(self):
        if self.temp is None:
            return None
        v = ctypes.c_double(0.0)
        try:
            return v.value if self.lib.ctlTemperatureGetState(self.temp, ctypes.byref(v)) == 0 else None
        except OSError:
            return None

    def power(self):
        """Watts: the energy counter's difference over its own timestamp's difference, both in microseconds."""
        if self.pwr is None:
            return None
        ec = _CtlEnergy()
        ec.Size = ctypes.sizeof(ec)
        try:
            if self.lib.ctlPowerGetEnergyCounter(self.pwr, ctypes.byref(ec)) != 0:
                return None
        except OSError:
            return None
        prev, self._energy = self._energy, (ec.timestamp, ec.energy)
        if prev is None or ec.timestamp <= prev[0] or ec.energy < prev[1]:
            return None
        return (ec.energy - prev[1]) / (ec.timestamp - prev[0])   # uJ / us = W


class _ZesPciStats(ctypes.Structure):      # zes_pci_stats_t
    _fields_ = [("output", ctypes.c_uint64), ("input", ctypes.c_uint64),
                ("maxBandwidthOutput", ctypes.c_uint64), ("maxBandwidthInput", ctypes.c_uint64),
                ("bandwidthOutput", ctypes.c_uint64), ("bandwidthInput", ctypes.c_uint64),
                ("packetsTotal", ctypes.c_uint64), ("packetsErrors", ctypes.c_uint64)]


class _WinXeGpu:
    """An Intel GPU on Windows (the engine's card under Level Zero): its VRAM and its static PCIe link through
    sysman (ze_loader.dll, which the GPU runtime installs), its load through the Performance Counters' GPU Engine
    set, its name through WMI, and its temperature and power through the driver's IGCL runtime.  `pci` is the
    card's address as the config gives it (setup's gpu_pci); without one, the first Level Zero device.

    The GPU Engine counters are per process and engine, and a wildcard query through ctypes returns no data, so
    the instance names are listed once at startup (PowerShell's Get-Counter, the one path that enumerates them)
    and one single-instance counter is added per name; the card they belong to is the LUID whose Dedicated Usage
    matches the VRAM sysman reads, which picks the engine's card on a PC with several GPUs."""

    def __init__(self, pci=None):
        self.lib = self.dev = self.pdh = None
        self.mems = []
        self.pcie = (None, None, None)      # (gen, width, gen_max): the static link, the only one Windows reports
        self.power_limit = None
        self.counters = []                  # (engtype, counter handle) of the engine's card
        self.name_ = None
        self.pci_stats = None               # zesDevicePciGetStats when it works (it needs an elevated process)
        self._pci_prev = None               # (time, output bytes): the GPU->host counter's previous sample
        # IGCL before Level Zero: once this process has opened the sysman device, a non-elevated IGCL enumerates
        # no sensors and refuses the energy counter (probe-verified on the engine's PC) — the order decides it.
        self.igcl = _Igcl()                 # temperature and power, which sysman keeps at 0 on Windows
        try:
            self.lib = ctypes.CDLL("ze_loader.dll")
        except OSError:
            return
        try:
            if self.lib.zesInit(ctypes.c_uint32(0)) != 0:
                self.lib = None
                return
            n = ctypes.c_uint32(0)
            if self.lib.zesDriverGet(ctypes.byref(n), None) != 0 or n.value == 0:
                self.lib = None
                return
            drivers = (ctypes.c_void_p * n.value)()
            self.lib.zesDriverGet(ctypes.byref(n), drivers)
            for drv in drivers:
                m = ctypes.c_uint32(0)
                if self.lib.zesDeviceGet(ctypes.c_void_p(drv), ctypes.byref(m), None) != 0:
                    continue
                devs = (ctypes.c_void_p * m.value)()
                self.lib.zesDeviceGet(ctypes.c_void_p(drv), ctypes.byref(m), devs)
                for dev in devs:
                    dh = ctypes.c_void_p(dev)
                    if pci is not None:
                        props = _ZesPciProps(stype=3)
                        if self.lib.zesDevicePciGetProperties(dh, ctypes.byref(props)) == 0:
                            a = props.address
                            if f"{a.domain:04x}:{a.bus:02x}:{a.device:02x}.{a.function:x}" != pci:
                                continue
                            self.pcie = (props.max_speed.gen, props.max_speed.width, props.max_speed.gen)
                    self.dev = dh
                    break
                if self.dev is not None:
                    break
            if self.dev is None:
                self.lib = None
                return
            k = ctypes.c_uint32(0)
            if self.lib.zesDeviceEnumMemoryModules(self.dev, ctypes.byref(k), None) == 0 and k.value:
                mods = (ctypes.c_void_p * k.value)()
                self.lib.zesDeviceEnumMemoryModules(self.dev, ctypes.byref(k), mods)
                self.mems = list(mods)
            props = _ZesPciProps(stype=3)
            if self.pcie[0] is None and self.lib.zesDevicePciGetProperties(self.dev, ctypes.byref(props)) == 0:
                self.pcie = (props.max_speed.gen, props.max_speed.width, props.max_speed.gen)
            k = ctypes.c_uint32(0)
            if self.lib.zesDeviceEnumPowerDomains(self.dev, ctypes.byref(k), None) == 0 and k.value:
                pds = (ctypes.c_void_p * k.value)()
                self.lib.zesDeviceEnumPowerDomains(self.dev, ctypes.byref(k), pds)
                fn, two = None, False
                for name in ("zesDevicePowerGetProperties", "zesPowerGetProperties"):
                    try:
                        fn = getattr(self.lib, name)
                        two = "Device" in name
                        break
                    except AttributeError:
                        continue
                if fn is not None:
                    for p in pds:
                        pr = _WinPowerProps(stype=0x13)
                        r = fn(self.dev, ctypes.c_void_p(p), ctypes.byref(pr)) if two \
                            else fn(ctypes.c_void_p(p), ctypes.byref(pr))
                        if r == 0:
                            lim = pr.maxLimit / 1e3     # maxLimit is in mW on this header
                            if 0 < lim < 1e4:
                                self.power_limit = max(self.power_limit or 0.0, lim)
        except (OSError, AttributeError):
            self.lib = self.dev = None
            return
        luid = self._engine_luid()
        if luid:
            self._open_pdh(luid)
        self.name_ = self._wmi_name()
        try:                                    # the link's traffic counters: they need an elevated process
            fn = getattr(self.lib, "zesDevicePciGetStats", None)
            if fn is not None:
                st = _ZesPciStats()
                if fn(self.dev, ctypes.byref(st)) == 0:
                    self.pci_stats = fn
        except (OSError, AttributeError):
            self.pci_stats = None

    def ok(self):
        return self.dev is not None

    @staticmethod
    def _powershell(script, timeout=60):
        import base64
        import subprocess
        try:
            # Python 3.15 defaults to UTF-8 mode: text=True would decode this pipe as UTF-8 while
            # PowerShell writes the OEM code page (cp932 on a Japanese Windows) - the reader thread
            # then dies on a UnicodeDecodeError.  Ask PowerShell itself for UTF-8 output and decode
            # the bytes here (utf-8-sig: PS 5.1 prefixes a BOM; errors=replace: never crash).
            script = "[Console]::OutputEncoding=[Text.Encoding]::UTF8;" + script
            enc = base64.b64encode(script.encode("utf-16-le")).decode()
            r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", enc],
                               capture_output=True, timeout=timeout)
            return r.stdout.decode("utf-8-sig", errors="replace") or ""
        except (OSError, subprocess.TimeoutExpired):
            return ""

    def _engine_luid(self):
        """The GPU Adapter Memory instance (a LUID) whose Dedicated Usage matches the VRAM sysman reads: the card
        the engine runs on, which on a PC with several GPUs is not necessarily the one NVML or WMI list first."""
        used, total = self._mem()
        if used is None:
            return None
        out = self._powershell(
            "(Get-Counter -Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -SampleInterval 1 -MaxSamples 1)"
            ".CounterSamples | ForEach-Object { $_.InstanceName + '=' + [Math]::Round($_.CookedValue/1GB,2) }")
        best, best_gap = None, None
        for line in out.splitlines():
            if "=" not in line or not line.strip().startswith("luid_"):
                continue
            inst, _, val = line.strip().rpartition("=")
            try:
                gap = abs(float(val) - used / 2**30)
            except ValueError:
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = inst, gap
        return best if best_gap is not None and best_gap < 2.0 else None

    def _open_pdh(self, luid):
        """One Utilization Percentage counter per engine of this card: the instance names come from PowerShell
        once (a ctypes wildcard query returns PDH_NO_DATA), the values are read per second from ctypes."""
        out = self._powershell(
            "(Get-Counter -Counter '\\GPU Engine(*)\\Utilization Percentage' -SampleInterval 1 -MaxSamples 1)"
            ".CounterSamples | ForEach-Object { $_.InstanceName } | Sort-Object -Unique", timeout=120)
        insts = [s.strip() for s in out.splitlines() if luid in s and "engtype_" in s]
        if not insts:
            return
        try:
            pdh = ctypes.windll.pdh
            pdh.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            pdh.PdhOpenQueryW.restype = ctypes.c_long
            pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_void_p,
                                                  ctypes.POINTER(ctypes.c_void_p)]
            pdh.PdhAddEnglishCounterW.restype = ctypes.c_long
            pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            pdh.PdhCollectQueryData.restype = ctypes.c_long
            pdh.PdhGetFormattedCounterValue.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                        ctypes.POINTER(ctypes.c_uint32),
                                                        ctypes.POINTER(_PdhCounterValue)]
            pdh.PdhGetFormattedCounterValue.restype = ctypes.c_long
            h = ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, None, ctypes.byref(h)) != 0:
                return
            counters = []
            for inst in insts:
                c = ctypes.c_void_p()
                if pdh.PdhAddEnglishCounterW(h, f"\\GPU Engine({inst})\\Utilization Percentage",
                                             None, ctypes.byref(c)) == 0:
                    counters.append((inst.rsplit("engtype_", 1)[-1], c))
            if not counters:
                return
            self.counters = counters
            self.pdh = (pdh, h, counters)
        except (OSError, AttributeError):
            self.pdh = None

    @staticmethod
    def _wmi_name():
        out = _WinXeGpu._powershell(
            "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.PNPDeviceID }")
        intel = [line.split("|")[0] for line in out.splitlines() if "|PCI\\VEN_8086" in line]
        return intel[0] if len(intel) == 1 else None

    def _mem(self):
        """(used, total) bytes over the device's memory modules, or (None, None)."""
        free = total = 0
        for mod in self.mems:
            st = _WinZesMemState(stype=0x1b)
            try:
                r = self.lib.zesMemoryGetState(ctypes.c_void_p(mod), ctypes.byref(st))
            except OSError:
                return None, None
            if r != 0:
                st = _WinZesMemState(stype=0x1e)
                if self.lib.zesMemoryGetState(ctypes.c_void_p(mod), ctypes.byref(st)) != 0:
                    return None, None
            free += st.free
            total += st.size
        return (total - free, total) if total else (None, None)

    def name(self):
        return self.name_

    def read(self):
        out = {}
        out["mem_used"], out["mem_total"] = self._mem()
        out["pcie_gen"], out["pcie_width"], out["pcie_gen_max"] = self.pcie
        out["power_limit"] = self.power_limit
        out["util"] = self._util()
        out["temp"] = self.igcl.temperature() if self.igcl else None
        out["power"] = self.igcl.power() if self.igcl else None
        out["pcie_rx_mb"], out["pcie_tx_mb"] = self._pci()
        return out

    def _pci(self):
        """(to GPU, from GPU) MB/s from the link's byte counters.  Intel's Windows driver fills only the
        GPU->host counter (input stays 0, compute-runtime#932), so the host->GPU direction is None there."""
        if self.pci_stats is None:
            return None, None
        st = _ZesPciStats()
        try:
            if self.pci_stats(self.dev, ctypes.byref(st)) != 0:
                return None, None
        except OSError:
            return None, None
        t = time.time()
        prev, self._pci_prev = self._pci_prev, (t, st.output, st.input)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        rx = (st.input - prev[2]) / dt / 2**20 if st.input >= prev[2] and st.input else None
        tx = (st.output - prev[1]) / dt / 2**20 if st.output >= prev[1] else None
        return rx, tx

    def _util(self):
        """The card's load: the sum of its engines' utilization, per engine type, the busiest type's sum - the
        engines of one type share the card's compute units, so their sums are the card's 0-100% scale."""
        if self.pdh is None:
            return None
        pdh, h, counters = self.pdh
        if pdh.PdhCollectQueryData(h) != 0:
            return None
        by_type = {}
        for engtype, c in counters:
            cv = _PdhCounterValue()
            cs = ctypes.c_uint32()
            if pdh.PdhGetFormattedCounterValue(c, ctypes.c_uint32(0x200), ctypes.byref(cs), ctypes.byref(cv)) == 0:
                by_type[engtype] = by_type.get(engtype, 0.0) + cv.u.Double
        if not by_type:
            return None
        return max(0.0, min(100.0, max(by_type.values())))


def _gpu(pci=None):
    """The card the engine runs on: on Windows the Intel GPU via sysman/PDH/WMI/IGCL, else the Intel GPU at
    `pci` (or the first on the xe driver), else the AMD one (amdgpu), else the NVIDIA one, else (without `pci`)
    the first on the i915 driver.  Without `pci` an i915 card comes last: it is most often the processor's
    graphics beside the card the engine runs on."""
    if os.name == "nt":
        w = _WinXeGpu(pci)
        if w.ok():
            return w
    g = _XeGpu(pci, ("xe", "i915") if pci else ("xe",))
    if g.ok():
        return g
    a = _AmdGpu(pci)
    if a.ok():
        return a
    n = _NvGpu(pci)
    if n.ok():
        return n
    i = _XeGpu(None, ("i915",)) if pci is None else g
    return i if i.ok() else g


def free_vram_mib(gpu_pci=None):
    """Free VRAM of the card (by PCI address, None: as _gpu chooses) in MiB, or None
    when it cannot be read."""
    g = _gpu(gpu_pci)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_name():
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra=None, gpu_pci=None, split_pci=None):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_pci`: the PCI address
        of the card the engine runs on (setup's gpu_pci), or None for the card _gpu chooses; `split_pci`: a layer
        split's other cards (setup's split_pci), read beside it."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        self.gpus = [(i, _gpu(pci)) for i, pci in enumerate([gpu_pci, *(split_pci or [])])]
        self.gpus = [(i, g) for i, g in self.gpus if g.ok()] or self.gpus[:1]
        self.gpu = self.gpus[0][1]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        self.names = [g.name() for _, g in self.gpus]
        self.static = {
            "gpu_name": " + ".join(g.name() or "?" for _, g in self.gpus) if self.gpu.ok() else None,
            "gpu_count": len(self.gpus),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
        self._disk_prev = None
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _disk(self):
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        if c is None:   # psutil found no disk (a gVisor container, Windows with its disk counters off)
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def sample(self):
        s = {}
        if self.gpu.ok():
            reads = [(i, g.read()) for i, g in self.gpus]
            g = dict(reads[0][1])
            if len(reads) > 1:
                def vals(k):
                    return [r[k] for _, r in reads if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                u = vals("util")
                g["util"] = sum(u) / len(u) if u else None
                t = vals("temp")
                g["temp"] = max(t) if t else None
                t = vals("mem_temp")
                g["mem_temp"] = max(t) if t else None
                s["gpus"] = [{"index": i, "name": n, "util": r.get("util"), "mem_used": r.get("mem_used"),
                              "mem_total": r.get("mem_total"), "temp": r.get("temp"), "mem_temp": r.get("mem_temp"),
                              "power": r.get("power")}
                             for (i, r), n in zip(reads, self.names)]
            s.update({f"gpu_{k}": v for k, v in g.items()})
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["disk_read_mb"], s["disk_write_mb"] = self._disk()
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def close(self):
        """Ends the sampler thread (a server that stops, a test's service): it used to run for the life of the
        process."""
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            s = self.sample()
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb",
                          "gpu_pcie_tx_mb", "cpu", "ram_used", "disk_read_mb", "tok_s",
                          "prefill_tok_s_mean"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            self._stop.wait(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
