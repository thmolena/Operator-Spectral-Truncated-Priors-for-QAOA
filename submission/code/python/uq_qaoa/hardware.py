from __future__ import annotations

import platform
import subprocess


def collect_hardware_info() -> dict[str, str]:
    info = {"os": platform.platform(), "machine": platform.machine(), "processor": platform.processor()}
    try:
        info["cpu_model"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        info["cpu_model"] = "unknown"
    return info
